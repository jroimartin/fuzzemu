//! Emulated RISC-V architecture (rv64i).

use std::cmp;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use crate::elf::{self, Elf};
use crate::mmu::{self, Mmu, VirtAddr};

/// A RISC-V emulator that supports rv64i (Base Integer Insturction Set for
/// 64-bit).
pub struct Emulator {
    mmu: Mmu,
    alloc_base: VirtAddr,
}

#[derive(Debug)]
pub enum Error {
    ElfError(elf::Error),
    MmuError(mmu::Error),

    IoError(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::MmuError(e) => write!(f, "MMU error: {}", e),
            Error::ElfError(e) => write!(f, "ELF error: {}", e),
            Error::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl From<mmu::Error> for Error {
    fn from(error: mmu::Error) -> Error {
        Error::MmuError(error)
    }
}

impl From<elf::Error> for Error {
    fn from(error: elf::Error) -> Error {
        Error::ElfError(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}

impl Emulator {
    /// Creates a new emulator. `mem_size` defines the memory size of the
    /// emulator.
    ///
    /// # Panics
    ///
    /// This function panics if `mem_size` is 0.
    pub fn new<P: AsRef<Path>>(
        program: P,
        mem_size: usize,
    ) -> Result<Emulator, Error> {
        let contents = fs::read(program)?;
        let elf = Elf::parse(&contents)?;

        let mut mmu = Mmu::new(mem_size);

        println!("entry = {:#x}", *elf.entry());

        let mut alloc_base = 0;
        for phdr in elf.phdrs() {
            let segment_bytes = contents
                .get(phdr.offset()..phdr.offset() + phdr.file_size())
                .ok_or(elf::Error::MalformedFile)?;

            mmu.poke(phdr.virt_addr(), segment_bytes)?;
            mmu.set_perms(phdr.virt_addr(), phdr.mem_size(), phdr.perms())?;

            alloc_base =
                cmp::max(alloc_base, *phdr.virt_addr() + phdr.mem_size());
        }
        println!("{:2x?}", mmu.peek(elf.entry(), 4).unwrap());

        Ok(Emulator {
            mmu,
            alloc_base: VirtAddr(alloc_base),
        })
    }

    /// Returns a copy of the Emulator, including its internal state.
    pub fn fork(&self) -> Emulator {
        Emulator {
            mmu: self.mmu.fork(),
            alloc_base: self.alloc_base,
        }
    }

    /// Resets the internal state of the emulator to the given state `other`.
    pub fn reset(&mut self, other: &Emulator) {
        self.mmu.reset(&other.mmu);
        self.alloc_base = other.alloc_base;
    }
}
