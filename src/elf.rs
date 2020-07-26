//! ELF 64-bit parser able to extract the PT_LOAD program headers of a program.

use std::convert::TryInto;
use std::fs;
use std::io;
use std::path::Path;

use crate::mmu::{Perm, VirtAddr};

/// ELF executable.
#[derive(Debug, PartialEq, Eq)]
pub struct Elf {
    phdrs: Vec<Phdr>,
    entry: VirtAddr,
}

/// ELF program header.
#[derive(Debug, PartialEq, Eq)]
pub struct Phdr {
    offset: usize,
    virt_addr: VirtAddr,
    file_size: usize,
    mem_size: usize,
    perms: Perm,
    align: usize,
}

/// Error related to ELF parsing.
#[derive(Debug)]
pub enum Error {
    IoError(io::Error),
    MalformedFile,
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}

impl Elf {
    /// Parses an ELF file and returs an `Elf` structure.
    pub fn load_file<T: AsRef<Path>>(path: T) -> Result<Elf, Error> {
        let contents = fs::read(path)?;

        Elf::load(&contents)
    }

    /// Parses a slice of bytes with the contents of an ELF file and returs an
    /// `Elf` structure.
    pub fn load(contents: &[u8]) -> Result<Elf, Error> {
        // Check ELF magic.
        if &contents[0..4] != b"\x7fELF" {
            return Err(Error::MalformedFile);
        }

        // Get entrypoint.
        let e_entry =
            usize::from_le_bytes(contents[24..24 + 8].try_into().unwrap());

        // Get program headers offset.
        let e_phoff =
            usize::from_le_bytes(contents[32..32 + 8].try_into().unwrap());

        // Get number of program headers.
        let e_phnum =
            u16::from_le_bytes(contents[56..56 + 2].try_into().unwrap())
                as usize;

        // Parse PT_LOAD program headers.
        let mut phdrs: Vec<Phdr> = Vec::with_capacity(e_phnum);

        for i in 0..e_phnum {
            let off = e_phoff + i * 56;

            // Get header type and skip non PT_LOAD headers.
            let p_type =
                u32::from_le_bytes(contents[off..off + 4].try_into().unwrap());
            if p_type != 1 {
                continue;
            }

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

            let phdr = Phdr {
                offset: p_offset,
                virt_addr: VirtAddr(p_vaddr),
                file_size: p_filesz,
                mem_size: p_memsz,
                perms: Perm(p_flags),
                align: p_align,
            };

            phdrs.push(phdr);
        }

        // Return an error if no PT_LOAD headers were found.
        if phdrs.len() == 0 {
            return Err(Error::MalformedFile);
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
    fn load_file() {
        let want = Elf {
            phdrs: vec![
                Phdr {
                    offset: 0x0000000000000000,
                    virt_addr: VirtAddr(0x0000000000010000),
                    file_size: 0x0000000000012868,
                    mem_size: 0x0000000000012868,
                    perms: Perm(PERM_READ | PERM_EXEC),
                    align: 0x1000,
                },
                Phdr {
                    offset: 0x0000000000012868,
                    virt_addr: VirtAddr(0x0000000000023868),
                    file_size: 0x0000000000001178,
                    mem_size: 0x0000000000001208,
                    perms: Perm(PERM_READ | PERM_WRITE),
                    align: 0x1000,
                },
            ],
            entry: VirtAddr(0x100c8),
        };

        assert_eq!(Elf::load_file("testdata/hello").unwrap(), want);
    }
}
