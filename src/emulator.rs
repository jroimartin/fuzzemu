//! Emulated RISC-V architecture (rv64i).

use crate::mmu::{Mmu, VirtAddr};

/// A RISC-V emulator that supports rv64i (Base Integer Insturction Set for
/// 64-bit).
pub struct Emulator {
    mmu: Mmu,
}

impl Emulator {
    /// Creates a new emulator. `mem_size` defines the memory size of the
    /// emulator. `alloc_base` sets the initial base address for memory
    /// allocations.
    ///
    /// # Panics
    ///
    /// This function panics if `mem_size` is 0 or below `alloc_base`.
    pub fn new(mem_size: usize, alloc_base: VirtAddr) -> Emulator {
        Emulator {
            mmu: Mmu::new(mem_size, alloc_base),
        }
    }

    /// Returns a copy of the Emulator, including its internal state.
    pub fn fork(&self) -> Emulator {
        Emulator {
            mmu: self.mmu.fork(),
        }
    }

    /// Resets the internal state of the emulator to the given state `other`.
    pub fn reset(&mut self, other: &Emulator) {
        self.mmu.reset(&other.mmu);
    }
}
