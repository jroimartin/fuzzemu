//! Parsing library for the ELF executable format.

pub mod defs;
mod endianread;

use std::convert::TryInto;
use std::fmt;
use std::fs;
use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

use crate::defs::*;
use crate::endianread::{EndianRead, FromBytes};

/// Error related to ELF parsing.
#[derive(Debug)]
pub enum Error {
    /// ELF parsing error.
    ParseError,

    /// Read error.
    ReadError,

    /// IO error when reading file.
    IoError(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::ParseError => write!(f, "parse error"),
            Error::ReadError => write!(f, "read error"),
            Error::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}

/// ELF parser. It specifies how to interpret the file and provides access to
/// the parsing methods.
#[derive(Debug, PartialEq, Eq)]
pub struct Parser {
    /// Contents of the ELF file.
    data: Cursor<Vec<u8>>,

    /// Architecture of the ELF binary.
    class: Class,

    /// Data encoding of the processor-specific data in the file.
    data_encoding: DataEncoding,

    /// Version number of the ELF specification.
    version: Version,

    /// Operating system and ABI to which the object is targeted.
    os_abi: OsAbi,

    /// ABI version byte index. This field is used to distinguish among
    /// incompatible versions of an ABI. The interpretation of this version
    /// number is dependent on the ABI identified by the EI_OSABI field.
    abi_version: u8,
}

impl Parser {
    /// Returns an `Parser` initialized with the data in `path`. It returns
    /// error if the `ident` field of the ELF header is not valid.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Parser, Error> {
        let data = fs::read(path)?;
        Parser::from_bytes(&data)
    }

    /// Returns an `Parser` initialized with the provided `data`. It returns
    /// error if the `ident` field of the ELF header is not valid.
    pub fn from_bytes(data: &[u8]) -> Result<Parser, Error> {
        // Check ELF magic.
        let magic = data.get(..4).ok_or(Error::ParseError)?;
        if magic != b"\x7fELF" {
            return Err(Error::ParseError);
        }

        // Parse ELF class and validate value.
        let &class = data.get(4).ok_or(Error::ParseError)?;

        // Parse ELF data encoding and validate value.
        let &data_encoding = data.get(5).ok_or(Error::ParseError)?;

        // Parse ELF version and validate value.
        let &version = data.get(6).ok_or(Error::ParseError)?;

        // Parse ELF OS ABI.
        let &os_abi = data.get(7).ok_or(Error::ParseError)?;

        // Parse ELF ABI version.
        let &abi_version = data.get(8).ok_or(Error::ParseError)?;

        let elf_parser = Parser {
            data: Cursor::new(data.to_vec()),
            class: class.try_into()?,
            data_encoding: data_encoding.try_into()?,
            version: version.try_into()?,
            os_abi: os_abi.into(),
            abi_version,
        };
        Ok(elf_parser)
    }

    /// Returns the contents of the underlying ELF file.
    pub fn data(&self) -> &[u8] {
        self.data.get_ref()
    }

    /// Returns the class of the underlying ELF file.
    pub fn class(&self) -> Class {
        self.class
    }

    /// Returns the data encoding of the underlying ELF file.
    pub fn data_encoding(&self) -> DataEncoding {
        self.data_encoding
    }

    /// Returns the version of the underlying ELF file.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns the OS ABI of the underlying ELF file.
    pub fn os_abi(&self) -> OsAbi {
        self.os_abi
    }

    /// Returns the ABI version of the underlying ELF file.
    pub fn abi_version(&self) -> u8 {
        self.abi_version
    }

    /// Reads a value from the ELF file. Endianness is automatically chosen
    /// based on `ident`.
    fn read_val<T: FromBytes>(&mut self) -> Result<T::Target, Error> {
        match self.data_encoding {
            DataEncoding::None => Err(Error::ParseError),
            DataEncoding::LittleEndian => Ok(self.data.read_le::<T>()?),
            DataEncoding::BigEndian => Ok(self.data.read_be::<T>()?),
        }
    }

    /// Returns an `Elf` structure containing the parsed data.
    pub fn parse(&mut self) -> Result<Elf, Error> {
        let ehdr = self.parse_ehdr()?;
        let mut phdrs = Vec::new();

        for i in 0..ehdr.phnum as u64 {
            let offset = ehdr.phoff + i * (ehdr.phentsize as u64);
            let phdr = self.parse_phdr(offset)?;
            phdrs.push(phdr);
        }

        Ok(Elf { ehdr, phdrs })
    }

    /// Returns an `Ehdr` structure containing the parsed ELF header.
    fn parse_ehdr(&mut self) -> Result<Ehdr, Error> {
        self.data.seek(SeekFrom::Start(0))?;

        match self.class {
            Class::None => Err(Error::ParseError),
            Class::Elf32 => {
                let mut ident = [0u8; 16];
                self.data.read_exact(&mut ident)?;

                let ehdr = Ehdr {
                    ident,
                    file_type: self.read_val::<u16>()?.into(),
                    machine: self.read_val::<u16>()?.into(),
                    version: self.read_val::<u32>()?.try_into()?,
                    entry: self.read_val::<u32>()? as u64,
                    phoff: self.read_val::<u32>()? as u64,
                    shoff: self.read_val::<u32>()? as u64,
                    flags: self.read_val::<u32>()?,
                    ehsize: self.read_val::<u16>()?,
                    phentsize: self.read_val::<u16>()?,
                    phnum: self.read_val::<u16>()?,
                    shentsize: self.read_val::<u16>()?,
                    shnum: self.read_val::<u16>()?,
                    shstrndx: self.read_val::<u16>()?,
                };

                Ok(ehdr)
            }
            Class::Elf64 => {
                let mut ident = [0u8; 16];
                self.data.read_exact(&mut ident)?;

                let ehdr = Ehdr {
                    ident,
                    file_type: self.read_val::<u16>()?.into(),
                    machine: self.read_val::<u16>()?.into(),
                    version: self.read_val::<u32>()?.try_into()?,
                    entry: self.read_val::<u64>()?,
                    phoff: self.read_val::<u64>()?,
                    shoff: self.read_val::<u64>()?,
                    flags: self.read_val::<u32>()?,
                    ehsize: self.read_val::<u16>()?,
                    phentsize: self.read_val::<u16>()?,
                    phnum: self.read_val::<u16>()?,
                    shentsize: self.read_val::<u16>()?,
                    shnum: self.read_val::<u16>()?,
                    shstrndx: self.read_val::<u16>()?,
                };

                Ok(ehdr)
            }
        }
    }

    /// Returns a `Phdr` structure containing the parsed program header.
    fn parse_phdr(&mut self, offset: u64) -> Result<Phdr, Error> {
        self.data.seek(SeekFrom::Start(offset))?;

        match self.class {
            Class::None => Err(Error::ParseError),
            Class::Elf32 => {
                let phdr = Phdr {
                    segment_type: self.read_val::<u32>()?.into(),
                    offset: self.read_val::<u32>()? as u64,
                    vaddr: self.read_val::<u32>()? as u64,
                    paddr: self.read_val::<u32>()? as u64,
                    filesz: self.read_val::<u32>()? as u64,
                    memsz: self.read_val::<u32>()? as u64,
                    flags: self.read_val::<u32>()?,
                    align: self.read_val::<u32>()? as u64,
                };

                Ok(phdr)
            }
            Class::Elf64 => {
                let phdr = Phdr {
                    segment_type: self.read_val::<u32>()?.into(),
                    flags: self.read_val::<u32>()?,
                    offset: self.read_val::<u64>()?,
                    vaddr: self.read_val::<u64>()?,
                    paddr: self.read_val::<u64>()?,
                    filesz: self.read_val::<u64>()?,
                    memsz: self.read_val::<u64>()?,
                    align: self.read_val::<u64>()?,
                };

                Ok(phdr)
            }
        }
    }
}

/// Parsed ELF executable.
#[derive(Debug, PartialEq, Eq)]
pub struct Elf {
    /// ELF header.
    ehdr: Ehdr,

    /// Program headers.
    phdrs: Vec<Phdr>,
}

impl Elf {
    /// Returns the ELF header of the parsed file.
    pub fn ehdr(&self) -> Ehdr {
        self.ehdr
    }

    /// Returns the program headers.
    pub fn phdrs(&self) -> Vec<Phdr> {
        self.phdrs.clone()
    }
}

/// ELF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ehdr {
    /// Specifies how to interpret the file, independent of the processor or
    /// the file's remaining contents.
    ident: [u8; 16],

    /// Object file type.
    file_type: FileType,

    /// Specifies the required architecture for an individual file.
    machine: Machine,

    /// File version.
    version: Version,

    /// Virtual address to which the system first transfers control, thus
    /// starting the process.
    entry: u64, // Elf32: u32

    /// Program header table's file offset in bytes.
    phoff: u64, // Elf32: u32

    /// Section header table's file offset in bytes.
    shoff: u64, // Elf32: u32

    /// Processor-specific flags associated with the file.
    flags: u32,

    /// ELF header's size in bytes.
    ehsize: u16,

    /// Size in bytes of one entry in the file's program header table.
    phentsize: u16,

    /// Number of entries in the program header table.
    phnum: u16,

    /// Sections header's size in bytes.
    shentsize: u16,

    /// Number of entries in the section header table.
    shnum: u16,

    /// Section header table index of the entry associated with the section
    /// name string table.
    shstrndx: u16,
}

impl Ehdr {
    /// Returns the `ident` field of the ELF header, which specifies how to
    /// interpret the file, independent of the processor or the file's
    /// remaining contents.
    pub fn ident(&self) -> [u8; 16] {
        self.ident
    }

    /// Returns the object file type.
    pub fn file_type(&self) -> FileType {
        self.file_type
    }

    /// Returns the required architecture for an individual file.
    pub fn machine(&self) -> Machine {
        self.machine
    }

    /// Returns the file version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// Returns the virtual address to which the system first transfers
    /// control, thus starting the process.
    pub fn entry(&self) -> u64 {
        self.entry
    }

    /// Returns the program header table's file offset in bytes.
    pub fn phoff(&self) -> u64 {
        self.phoff
    }

    /// Returns the section header table's file offset in bytes.
    pub fn shoff(&self) -> u64 {
        self.shoff
    }

    /// Returns the processor-specific flags associated with the file.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Returns the ELF header's size in bytes.
    pub fn ehsize(&self) -> u16 {
        self.ehsize
    }

    /// Returns the size in bytes of one entry in the file's program header
    /// table.
    pub fn phentsize(&self) -> u16 {
        self.phentsize
    }

    /// Returns the number of entries in the program header table.
    pub fn phnum(&self) -> u16 {
        self.phnum
    }

    /// Returns the sections header's size in bytes.
    pub fn shentsize(&self) -> u16 {
        self.shentsize
    }

    /// Returns the number of entries in the section header table.
    pub fn shnum(&self) -> u16 {
        self.shnum
    }

    /// Returns the section header table index of the entry associated with the
    /// section name string table.
    pub fn shstrndx(&self) -> u16 {
        self.shstrndx
    }
}

/// Program header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phdr {
    /// Type of segment.
    segment_type: SegmentType,

    /// Bit mask of flags relevant to the segment.
    flags: u32,

    /// Offset from the beginning of the file at which the first byte of the
    /// segment resides.
    offset: u64, // Elf32: u32

    /// Virtual address at which the first byte of the segment resides in
    /// memory.
    vaddr: u64, // Elf32: u32

    /// On systems for which physical addressing is relevant, this member
    /// is reserved for the segment's physical address.
    paddr: u64, // Elf32: u32

    /// Number of bytes in the file image of the segment.
    filesz: u64, // Elf32: u32

    /// Number of bytes in the memory image of the segment.
    memsz: u64, // Elf32: u32

    /// value to which the segments are aligned in memory and in the file.
    align: u64, // Elf32: u32
}

impl Phdr {
    /// Returns the type of segment.
    pub fn segment_type(&self) -> SegmentType {
        self.segment_type
    }

    /// Returns a bit mask of flags relevant to the segment.
    pub fn flags(&self) -> u32 {
        self.flags
    }

    /// Returns the offset from the beginning of the file at which the first
    /// byte of the segment resides.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the virtual address at which the first byte of the segment
    /// resides in memory.
    pub fn vaddr(&self) -> u64 {
        self.vaddr
    }

    /// Returns the segment's physical address. This only applies to systems
    /// for which physical addressing is relevant.
    pub fn paddr(&self) -> u64 {
        self.paddr
    }

    /// Returns the number of bytes in the file image of the segment.
    pub fn filesz(&self) -> u64 {
        self.filesz
    }

    /// Returns the number of bytes in the memory image of the segment.
    pub fn memsz(&self) -> u64 {
        self.memsz
    }

    /// Returns the value to which the segments are aligned in memory and in
    /// the file.
    pub fn align(&self) -> u64 {
        self.align
    }
}

/// Parses `data` as an ELF file and returns an `Elf` structure.
pub fn parse(data: &[u8]) -> Result<Elf, Error> {
    Parser::from_bytes(&data)?.parse()
}

/// Parses an ELF file and returns an `Elf` structure.
pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Elf, Error> {
    Parser::from_file(path)?.parse()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ident() {
        let filename = Path::new("testdata").join("hello");
        let contents = fs::read(filename).unwrap();
        let elf_parser = Parser::from_bytes(&contents).unwrap();

        let want = Parser {
            data: Cursor::new(contents),
            class: Class::Elf64,
            data_encoding: DataEncoding::LittleEndian,
            version: Version::Current,
            os_abi: OsAbi::SysV,
            abi_version: 0,
        };

        assert_eq!(elf_parser, want);
    }

    #[test]
    fn parse_ehdr() {
        let filename = Path::new("testdata").join("hello");
        let contents = fs::read(filename).unwrap();

        let mut elf_parser = Parser::from_bytes(&contents).unwrap();
        let headers = elf_parser.parse().unwrap();
        let ehdr = headers.ehdr();

        let want = Ehdr {
            ident: contents[..16].try_into().unwrap(),
            file_type: FileType::Exec,
            machine: Machine::RiscV,
            version: Version::Current,
            entry: 0x100c8,
            phoff: 64,
            shoff: 160760,
            flags: 0,
            ehsize: 64,
            phentsize: 56,
            phnum: 2,
            shentsize: 64,
            shnum: 23,
            shstrndx: 22,
        };

        assert_eq!(ehdr, want);
    }

    #[test]
    fn parse_phdrs() {
        let filename = Path::new("testdata").join("hello");
        let contents = fs::read(filename).unwrap();

        let mut elf_parser = Parser::from_bytes(&contents).unwrap();
        let headers = elf_parser.parse().unwrap();
        let phdrs = headers.phdrs();

        let want = vec![
            Phdr {
                segment_type: SegmentType::Load,
                flags: PF_R | PF_X,
                offset: 0x0000000000000000,
                vaddr: 0x0000000000010000,
                paddr: 0x0000000000010000,
                filesz: 0x0000000000012888,
                memsz: 0x0000000000012888,
                align: 0x1000,
            },
            Phdr {
                segment_type: SegmentType::Load,
                flags: PF_R | PF_W,
                offset: 0x0000000000012888,
                vaddr: 0x0000000000023888,
                paddr: 0x0000000000023888,
                filesz: 0x0000000000001178,
                memsz: 0x0000000000001208,
                align: 0x1000,
            },
        ];

        assert_eq!(phdrs, want);
    }

    #[test]
    fn read_le_val() {
        let mut elf_parser = Parser {
            data: Cursor::new(vec![0x41, 0x42, 0x43, 0x44, 0x45]),
            class: Class::Elf64,
            data_encoding: DataEncoding::LittleEndian,
            version: Version::Current,
            os_abi: OsAbi::SysV,
            abi_version: 0,
        };

        let value = elf_parser.read_val::<u8>().unwrap();
        assert_eq!(value, 0x41u8);
        let value = elf_parser.read_val::<u32>().unwrap();
        assert_eq!(value, 0x45444342u32);
    }

    #[test]
    fn read_be_val() {
        let mut elf_parser = Parser {
            data: Cursor::new(vec![0x41, 0x42, 0x43, 0x44, 0x45]),
            class: Class::Elf64,
            data_encoding: DataEncoding::BigEndian,
            version: Version::Current,
            os_abi: OsAbi::SysV,
            abi_version: 0,
        };

        let value = elf_parser.read_val::<u8>().unwrap();
        assert_eq!(value, 0x41u8);
        let value = elf_parser.read_val::<u32>().unwrap();
        assert_eq!(value, 0x42434445);
    }
}
