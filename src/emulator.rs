use crate::mmu::Mmu;

pub struct Emulator {
    mmu: Mmu,
}

impl Emulator {
    pub fn new(memsize: usize) -> Emulator {
        Emulator {
            mmu: Mmu::new(memsize),
        }
    }

    pub fn fork(&self) -> Emulator {
        Emulator {
            mmu: self.mmu.fork(),
        }
    }

    pub fn reset(&mut self, other: &Emulator) {
        self.mmu.reset(&other.mmu);
    }
}
