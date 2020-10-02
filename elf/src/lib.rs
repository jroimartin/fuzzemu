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
    pub fn from_bytes<D: AsRef<[u8]>>(data: D) -> Result<Parser, Error> {
        let data = data.as_ref();

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

        let shdr_0 = self.parse_shdr(ehdr.shoff)?;

        let mut shdrs = vec![shdr_0];

        // If the number of entries in the program header table is larger than
        // or equal to `PN_XNUM` (0xffff), this member holds `PN_XNUM` (0xffff)
        // and the real number of entries in the program header table is held
        // in the `info` member of the initial entry in section header table.
        // Otherwise, the `info` member of the initial entry contains the value
        // zero.
        let phnum = if ehdr.phnum == PN_XNUM {
            shdr_0.info
        } else {
            ehdr.phnum as u32
        };

        if phnum == 0 {
            return Err(Error::ParseError);
        }

        for i in 0..phnum as u64 {
            let offset = ehdr
                .phoff
                .checked_add(i * (ehdr.phentsize as u64))
                .ok_or(Error::ParseError)?;
            let phdr = self.parse_phdr(offset)?;
            phdrs.push(phdr);
        }

        // If the number of entries in the section header table is larger than
        // or equal to `SHN_LORESERVE` (0xff00), `shnum` holds the value zero
        // and the real number of entries in the section header table is held
        // in the `size` member of the initial entry in section header table.
        // Otherwise, the `size` member of the initial entry in the section
        // header table holds the value zero.
        let shnum = if ehdr.shnum == 0 {
            shdr_0.size
        } else {
            ehdr.shnum as u64
        };

        if shnum == 0 {
            return Err(Error::ParseError);
        }

        for i in 1..shnum as u64 {
            let offset = ehdr
                .shoff
                .checked_add(i * (ehdr.shentsize as u64))
                .ok_or(Error::ParseError)?;
            let shdr = self.parse_shdr(offset)?;
            shdrs.push(shdr);
        }

        Ok(Elf { ehdr, phdrs, shdrs })
    }

    /// Returns an `Ehdr` structure containing the parsed ELF header.
    fn parse_ehdr(&mut self) -> Result<Ehdr, Error> {
        self.data.seek(SeekFrom::Start(0))?;

        match self.class {
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
            Class::None => Err(Error::ParseError),
        }
    }

    /// Returns a `Phdr` structure containing the parsed program header.
    fn parse_phdr(&mut self, offset: u64) -> Result<Phdr, Error> {
        self.data.seek(SeekFrom::Start(offset))?;

        match self.class {
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
            Class::None => Err(Error::ParseError),
        }
    }

    /// Returns a `Shdr` structure containing the parsed section header.
    fn parse_shdr(&mut self, offset: u64) -> Result<Shdr, Error> {
        self.data.seek(SeekFrom::Start(offset))?;

        match self.class {
            Class::Elf32 => {
                let shdr = Shdr {
                    name: self.read_val::<u32>()?,
                    section_type: self.read_val::<u32>()?.into(),
                    flags: self.read_val::<u32>()? as u64,
                    addr: self.read_val::<u64>()?,
                    offset: self.read_val::<u64>()?,
                    size: self.read_val::<u32>()? as u64,
                    link: self.read_val::<u32>()?,
                    info: self.read_val::<u32>()?,
                    addralign: self.read_val::<u32>()? as u64,
                    entsize: self.read_val::<u32>()? as u64,
                };

                Ok(shdr)
            }
            Class::Elf64 => {
                let shdr = Shdr {
                    name: self.read_val::<u32>()?,
                    section_type: self.read_val::<u32>()?.into(),
                    flags: self.read_val::<u64>()?,
                    addr: self.read_val::<u64>()?,
                    offset: self.read_val::<u64>()?,
                    size: self.read_val::<u64>()?,
                    link: self.read_val::<u32>()?,
                    info: self.read_val::<u32>()?,
                    addralign: self.read_val::<u64>()?,
                    entsize: self.read_val::<u64>()?,
                };

                Ok(shdr)
            }
            Class::None => Err(Error::ParseError),
        }
    }
}

/// Parses `data` as an ELF file and returns an `Elf` structure.
pub fn parse<D: AsRef<[u8]>>(data: D) -> Result<Elf, Error> {
    Parser::from_bytes(data)?.parse()
}

/// Parses an ELF file and returns an `Elf` structure.
pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Elf, Error> {
    Parser::from_file(path)?.parse()
}

/// Parsed ELF executable.
#[derive(Debug, PartialEq, Eq)]
pub struct Elf {
    /// ELF header.
    ehdr: Ehdr,

    /// Program headers.
    phdrs: Vec<Phdr>,

    /// Section headers.
    shdrs: Vec<Shdr>,
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

    /// Returns the section headers.
    pub fn shdrs(&self) -> Vec<Shdr> {
        self.shdrs.clone()
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

    /// Returns a bit mask of flags relevant to the segment. See `PF_` consts.
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

/// Section header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shdr {
    /// Name of the section. Its value is an index into the section header
    /// string table section, giving the location of a null-terminated string.
    name: u32,

    /// Categorizes the section's contents and semantics.
    section_type: SectionType,

    /// Sections support one-bit flags that describe miscellaneous attributes.
    /// See `SHF_` consts.
    flags: u64, // Elf32: u32

    /// If this section appears in the memory image of a process, this member
    /// holds the address at which the section's first byte should reside.
    /// Otherwise, the member contains zero.
    addr: u64,

    /// Byte offset from the beginning of the file to the first byte in the
    /// section.
    offset: u64,

    /// Section's size in bytes.
    size: u64, // Elf32: u32

    /// Section header table index link.
    link: u32,

    /// Extra information.
    info: u32,

    /// Some sections have address alignment constraints. The value 0 or 1
    /// means that the section has no alignment constraints.
    addralign: u64, // Elf32: u32

    /// Some sections hold a table of fixed-sized entries, such as a symbol
    /// table. For such a section, this member gives the size in bytes for
    /// each entry. This member contains zero if the section does not hold a
    /// table of fixed-size entries.
    entsize: u64, // Elf32: u32
}

impl Shdr {
    /// Returns the name of the section. Its value is an index into the section
    /// header string table section, giving the location of a null-terminated
    /// string.
    pub fn name(&self) -> u32 {
        self.name
    }

    /// Returns the section type, which categorizes the section's contents and
    /// semantics.
    pub fn section_type(&self) -> SectionType {
        self.section_type
    }

    /// Returns the flags of the section. Sections support one-bit flags that
    /// describe miscellaneous attributes. See `SHF_` consts.
    pub fn flags(&self) -> u64 {
        self.flags
    }

    /// If this section appears in the memory image of a process, this function
    /// returns the address at which the section's first byte should reside.
    /// Otherwise, zero is returned.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// Returns the byte offset from the beginning of the file to the first
    /// byte in the section.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the section's size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Returns the section header table index link.
    pub fn link(&self) -> u32 {
        self.link
    }

    /// Returns extra information about the section.
    pub fn info(&self) -> u32 {
        self.info
    }

    /// Returns the alignment of the section. Some sections have address
    /// alignment constraints. The value 0 or 1 means that the section has no
    /// alignment constraints.
    pub fn addralign(&self) -> u64 {
        self.addralign
    }

    /// Some sections hold a table of fixed-sized entries, such as a symbol
    /// table. For such a section, this function returns the size in bytes for
    /// each entry. If the section does not hold a table of fixed-size entries,
    /// zero is returned.
    pub fn entsize(&self) -> u64 {
        self.entsize
    }
}

#[cfg(test)]
mod tests {
    // TODO(rm): Add binaries using `shdrs[0]` for `shnum` and `phnum`.

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
    fn parse_shdrs() {
        let filename = Path::new("testdata").join("hello");
        let contents = fs::read(filename).unwrap();

        let mut elf_parser = Parser::from_bytes(&contents).unwrap();
        let headers = elf_parser.parse().unwrap();
        let shdrs = headers.shdrs();

        let want = vec![
            Shdr {
                name: 0,
                section_type: SectionType::Null,
                addr: 0x0000000000000000,
                offset: 0x00000000,
                size: 0x0000000000000000,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 0,
            },
            Shdr {
                name: 0x1b, // ".text",
                section_type: SectionType::ProgBits,
                addr: 0x00000000000100b0,
                offset: 0x000000b0,
                size: 0x00000000000119b0,
                entsize: 0x0000000000000000,
                flags: SHF_ALLOC | SHF_EXECINSTR,
                link: 0,
                info: 0,
                addralign: 4,
            },
            Shdr {
                name: 0x21, // ".rodata",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000021a60,
                offset: 0x00011a60,
                size: 0x0000000000000e28,
                entsize: 0x0000000000000000,
                flags: SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 16,
            },
            Shdr {
                name: 0x29, // ".eh_frame",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000023888,
                offset: 0x00012888,
                size: 0x0000000000000004,
                entsize: 0x0000000000000000,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 4,
            },
            Shdr {
                name: 0x33, // ".init_array",
                section_type: SectionType::InitArray,
                addr: 0x0000000000023890,
                offset: 0x00012890,
                size: 0x0000000000000010,
                entsize: 0x0000000000000008,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x3f, // ".fini_array",
                section_type: SectionType::FiniArray,
                addr: 0x00000000000238a0,
                offset: 0x000128a0,
                size: 0x0000000000000008,
                entsize: 0x0000000000000008,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x4b, // ".data",
                section_type: SectionType::ProgBits,
                addr: 0x00000000000238a8,
                offset: 0x000128a8,
                size: 0x0000000000001100,
                entsize: 0x0000000000000000,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x51, // ".sdata",
                section_type: SectionType::ProgBits,
                addr: 0x00000000000249a8,
                offset: 0x000139a8,
                size: 0x0000000000000058,
                entsize: 0x0000000000000000,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x58, // ".sbss",
                section_type: SectionType::NoBits,
                addr: 0x0000000000024a00,
                offset: 0x00013a00,
                size: 0x0000000000000030,
                entsize: 0x0000000000000000,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x5e, // ".bss",
                section_type: SectionType::NoBits,
                addr: 0x0000000000024a30,
                offset: 0x00013a00,
                size: 0x0000000000000060,
                entsize: 0x0000000000000000,
                flags: SHF_WRITE | SHF_ALLOC,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0x63, // ".comment",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00013a00,
                size: 0x0000000000000012,
                entsize: 0x0000000000000001,
                flags: SHF_MERGE | SHF_STRINGS,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0x6c, // ".riscv.attributes",
                section_type: SectionType::RiscVAttributes,
                addr: 0x0000000000000000,
                offset: 0x00013a12,
                size: 0x000000000000001c,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0x7e, // ".debug_aranges",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00013a30,
                size: 0x00000000000002c0,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 16,
            },
            Shdr {
                name: 0x8d, // ".debug_info",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00013cf0,
                size: 0x0000000000002f10,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0x99, // ".debug_abbrev",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00016c00,
                size: 0x000000000000119a,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0xa7, // ".debug_line",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00017d9a,
                size: 0x0000000000004a11,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0xb3, // ".debug_frame",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x0001c7b0,
                size: 0x00000000000002e0,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 8,
            },
            Shdr {
                name: 0xc0, // ".debug_str",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x0001ca90,
                size: 0x0000000000000b02,
                entsize: 0x0000000000000001,
                flags: SHF_MERGE | SHF_STRINGS,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0xcb, // ".debug_loc",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x0001d592,
                size: 0x0000000000005f8e,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0xd6, // ".debug_ranges",
                section_type: SectionType::ProgBits,
                addr: 0x0000000000000000,
                offset: 0x00023520,
                size: 0x0000000000001130,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0x1, // ".symtab",
                section_type: SectionType::SymTab,
                addr: 0x0000000000000000,
                offset: 0x00024650,
                size: 0x0000000000002040,
                entsize: 0x0000000000000018,
                flags: 0,
                link: 21,
                info: 158,
                addralign: 8,
            },
            Shdr {
                name: 0x9, // ".strtab",
                section_type: SectionType::StrTab,
                addr: 0x0000000000000000,
                offset: 0x00026690,
                size: 0x0000000000000c7d,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
            Shdr {
                name: 0x11, // ".shstrtab",
                section_type: SectionType::StrTab,
                addr: 0x0000000000000000,
                offset: 0x0002730d,
                size: 0x00000000000000e4,
                entsize: 0x0000000000000000,
                flags: 0,
                link: 0,
                info: 0,
                addralign: 1,
            },
        ];

        assert_eq!(shdrs, want);
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
