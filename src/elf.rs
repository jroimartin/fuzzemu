//! ELF 64-bit parser able to extract the PT_LOAD program headers of a program.

use std::convert::TryInto;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use crate::mmu::{Perm, VirtAddr, PERM_EXEC, PERM_READ, PERM_WRITE};

/// Error related to ELF parsing.
#[derive(Debug)]
pub enum Error {
    /// Malformed file
    MalformedFile,

    /// No loadable segments found.
    NoLoadHeaders,

    /// IO error when reading file.
    IoError(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MalformedFile => write!(f, "malformed file"),
            Error::NoLoadHeaders => write!(f, "no LOAD program headers"),
            Error::IoError(err) => write!(f, "{}", err),
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}

/// ELF program header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phdr {
    offset: usize,
    virt_addr: VirtAddr,
    file_size: usize,
    mem_size: usize,
    perms: Perm,
    align: usize,
}

impl Phdr {
    /// Offset from the beginning of the file at which the first byte of the
    /// segment resides.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Virtual address at which the first byte of the segment resides in
    /// memory.
    pub fn virt_addr(&self) -> VirtAddr {
        self.virt_addr
    }

    /// Number of bytes in the file image of the segment.
    pub fn file_size(&self) -> usize {
        self.file_size
    }

    /// Number of bytes in the memory image of the segment.
    pub fn mem_size(&self) -> usize {
        self.mem_size
    }

    /// Memory permissions of the segment.
    pub fn perms(&self) -> Perm {
        self.perms
    }

    /// Value to which the segment is aligned in memory and in the file.
    pub fn align(&self) -> usize {
        self.align
    }
}

/// ELF executable.
#[derive(Debug, PartialEq, Eq)]
pub struct Elf {
    entry: VirtAddr,
    phdrs: Vec<Phdr>,
}

impl Elf {
    /// Virtual address to which the system first tranfers control, thus
    /// starting the process.
    pub fn entry(&self) -> VirtAddr {
        self.entry
    }

    /// Program headers table.
    pub fn phdrs(&self) -> Vec<Phdr> {
        self.phdrs.clone()
    }

    /// Parses an ELF file and returns an `Elf` structure.
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Elf, Error> {
        let contents = fs::read(path)?;

        Elf::parse(&contents)
    }

    /// Parses a slice of bytes with the contents of an ELF file and returns an
    /// `Elf` structure.
    pub fn parse(contents: &[u8]) -> Result<Elf, Error> {
        // Check ELF magic.
        if &contents[0..4] != b"\x7fELF" {
            return Err(Error::MalformedFile);
        }

        // Get entrypoint.
        let e_entry =
            usize::from_le_bytes(contents[24..32].try_into().unwrap());

        // Get program headers offset.
        let e_phoff =
            usize::from_le_bytes(contents[32..40].try_into().unwrap());

        // Get number of program headers.
        let e_phnum =
            u16::from_le_bytes(contents[56..58].try_into().unwrap()) as usize;

        // Parse PT_LOAD program headers.
        let mut phdrs = Vec::with_capacity(e_phnum);

        for i in 0..e_phnum {
            let off = e_phoff + i * 56;

            // Get header type and skip non PT_LOAD headers.
            let p_type =
                u32::from_le_bytes(contents[off..off + 4].try_into().unwrap());
            if p_type != 1 {
                continue;
            }

            // Get the relevant fields of the program header.
            let p_flags = u32::from_le_bytes(
                contents[off + 4..off + 8].try_into().unwrap(),
            );
            let p_offset = usize::from_le_bytes(
                contents[off + 8..off + 16].try_into().unwrap(),
            );
            let p_vaddr = usize::from_le_bytes(
                contents[off + 16..off + 24].try_into().unwrap(),
            );
            let p_filesz = usize::from_le_bytes(
                contents[off + 32..off + 40].try_into().unwrap(),
            );
            let p_memsz = usize::from_le_bytes(
                contents[off + 40..off + 48].try_into().unwrap(),
            );
            let p_align = usize::from_le_bytes(
                contents[off + 48..off + 56].try_into().unwrap(),
            );

            // Convert flags to MMU permissions
            let mut perms = 0;
            // PF_X = 0x1
            if p_flags & 1 != 0 {
                perms |= PERM_EXEC;
            }
            // PF_W = 0x2
            if p_flags & 2 != 0 {
                perms |= PERM_WRITE;
            }
            // PF_R = 0x4
            if p_flags & 4 != 0 {
                perms |= PERM_READ;
            }

            let phdr = Phdr {
                offset: p_offset,
                virt_addr: VirtAddr(p_vaddr),
                file_size: p_filesz,
                mem_size: p_memsz,
                perms: Perm(perms),
                align: p_align,
            };

            phdrs.push(phdr);
        }

        // Return an error if no PT_LOAD headers were found.
        if phdrs.is_empty() {
            return Err(Error::NoLoadHeaders);
        }

        Ok(Elf {
            entry: VirtAddr(e_entry),
            phdrs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::mmu::{PERM_EXEC, PERM_READ, PERM_WRITE};

    #[test]
    fn elf_parse_file() {
        let want = Elf {
            phdrs: vec![
                Phdr {
                    offset: 0x0000000000000000,
                    virt_addr: VirtAddr(0x0000000000010000),
                    file_size: 0x0000000000012888,
                    mem_size: 0x0000000000012888,
                    perms: Perm(PERM_READ | PERM_EXEC),
                    align: 0x1000,
                },
                Phdr {
                    offset: 0x0000000000012888,
                    virt_addr: VirtAddr(0x0000000000023888),
                    file_size: 0x0000000000001178,
                    mem_size: 0x0000000000001208,
                    perms: Perm(PERM_READ | PERM_WRITE),
                    align: 0x1000,
                },
            ],
            entry: VirtAddr(0x100c8),
        };

        assert_eq!(Elf::parse_file("testdata/hello").unwrap(), want);
    }
}
