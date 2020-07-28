//! Emulated MMU with byte-level memory permissions able to detect unitialized
//! memory accesses.

use std::fmt;
use std::ops::Deref;

/// Executable memory. Aimed to be used with `Perm`.
pub const PERM_EXEC: u8 = 1;

/// Writable memory. Aimed to be used with `Perm`.
pub const PERM_WRITE: u8 = 1 << 1;

/// Readable memory. Aimed to be used with `Perm`.
pub const PERM_READ: u8 = 1 << 2;

/// Read-after-write memory. Aimed to be used with `Perm`.
///
/// This permission should be set when allocating writable memory. If a memory
/// position has this flag and is written, the READ permission will be
/// automatically assigned afterwards. This allows us to detect accesses to
/// unitialized memory.
pub const PERM_RAW: u8 = 1 << 3;

/// Block size used for resetting and tracking memory which has been modified.
/// Memory is considered dirty after writing to it and after changing its
/// permissions.
const DIRTY_BLOCK_SIZE: usize = 1024;

/// Memory error.
#[derive(Debug)]
pub enum Error {
    /// Memory address is out of range `(vaddr, size)`.
    InvalidAddress(VirtAddr, usize),

    /// Integer overflow when computing address `(vaddr, size)`.
    AddressIntegerOverflow(VirtAddr, usize),

    /// Read access to unitialized memory `(vaddr)`.
    UnitializedMemory(VirtAddr),

    /// Permissions do not allow memory access `(vaddr, expected, current)`.
    NotAllowed(VirtAddr, Perm, Perm),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InvalidAddress(vaddr, size) => {
                write!(f, "invalid address: vaddr={} size={}", vaddr, size)
            }
            Error::AddressIntegerOverflow(vaddr, size) => {
                write!(f, "integer overflow: vaddr={} size={}", vaddr, size)
            }
            Error::UnitializedMemory(vaddr) => {
                write!(f, "unitialized memory: vaddr={}", vaddr)
            }
            Error::NotAllowed(vaddr, exp, cur) => write!(
                f,
                "not allowed: vaddr={} exp={} cur={}",
                vaddr, exp, cur
            ),
        }
    }
}

/// Memory permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perm(pub u8);

impl fmt::Display for Perm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut disp = String::from("");

        if self.0 & PERM_READ != 0 {
            disp.push_str("R");
        } else {
            disp.push_str("-");
        }

        if self.0 & PERM_WRITE != 0 {
            disp.push_str("W");
        } else {
            disp.push_str("-");
        }

        if self.0 & PERM_EXEC != 0 {
            disp.push_str("X");
        } else {
            disp.push_str("-");
        }

        write!(f, "{}", disp)
    }
}

impl Deref for Perm {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Virtual address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct VirtAddr(pub usize);

impl Deref for VirtAddr {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Display for VirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

/// Emulated memory management unit.
#[derive(Debug, PartialEq, Eq)]
pub struct Mmu {
    /// Memory size.
    size: usize,

    /// Memory contents.
    memory: Vec<u8>,

    /// Byte-level memory permissions.
    perms: Vec<Perm>,

    /// Block indices in `memory` which are dirty.
    dirty: Vec<usize>,

    /// Tracks which parts of memory have been dirtied.
    dirty_bitmap: Vec<u64>,
}

impl Mmu {
    /// Returns a new Mmu with a given memory `size`.
    ///
    /// # Panics
    ///
    /// This function panics if `size` is 0.
    pub fn new(size: usize) -> Mmu {
        assert!(size > 0, "invalid size");

        Mmu {
            size,
            memory: vec![0; size],
            perms: vec![Perm(0); size],
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0; size / DIRTY_BLOCK_SIZE / 64 + 1],
        }
    }

    /// Returns a copy of the MMU. It marks all memory as clean in the new
    /// copy.
    pub fn fork(&self) -> Mmu {
        Mmu {
            size: self.size,
            memory: self.memory.clone(),
            perms: self.perms.clone(),
            dirty: Vec::with_capacity(self.size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0; self.size / DIRTY_BLOCK_SIZE / 64 + 1],
        }
    }

    /// Restores memory to the original state `other`.
    pub fn reset(&mut self, other: &Mmu) {
        // Restore memory and set as clean.
        for &block in &self.dirty {
            let start = block * DIRTY_BLOCK_SIZE;
            let end = (block + 1) * DIRTY_BLOCK_SIZE;

            self.dirty_bitmap[block / 64] = 0;
            self.memory[start..end].copy_from_slice(&other.memory[start..end]);
            self.perms[start..end].copy_from_slice(&other.perms[start..end]);
        }
        self.dirty.clear();
    }

    /// Set memory permissions in the given range.
    pub fn set_perms(
        &mut self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<(), Error> {
        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        self.perms
            .get_mut(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))?
            .iter_mut()
            .for_each(|p| *p = perms);

        self.update_dirty(addr, size);

        Ok(())
    }

    /// Given a memory range and the expected permissions, this function will
    /// return true if every byte in the specified region satisfies those
    /// permissions. Otherwise, the function will return false.
    pub fn check_perms(
        &self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<(), Error> {
        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        let range = self
            .perms
            .get(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))?;

        for (i, p) in range.iter().enumerate() {
            // If we reach this point, we know that addr + size does not
            // overflow. Given that i < size, we don't need to use
            // checked_add to calculate the following virtual addresses.

            if (*perms & PERM_READ != 0) && (**p & PERM_RAW != 0) {
                return Err(Error::UnitializedMemory(VirtAddr(*addr + i)));
            }

            if **p & *perms != *perms {
                return Err(Error::NotAllowed(VirtAddr(*addr + i), perms, *p));
            }
        }

        Ok(())
    }

    /// Copy the bytes in `src` to the given memory address. This function will
    /// fail if the destination memory is not writable.
    pub fn write(&mut self, addr: VirtAddr, src: &[u8]) -> Result<(), Error> {
        let size = src.len();

        // Check if the destination memory range is writable.
        self.check_perms(addr, size, Perm(PERM_WRITE))?;

        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        // Update memory contents
        self.memory
            .get_mut(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))?
            .copy_from_slice(src);

        // Add PERM_READ and remove PERM_RAW in case of RAW.
        self.perms
            .get_mut(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))?
            .iter_mut()
            .filter(|p| ***p & PERM_RAW != 0)
            .for_each(|p| *p = Perm((**p | PERM_READ) & !PERM_RAW));

        self.update_dirty(addr, size);

        Ok(())
    }

    /// Returns a slice with the data stored in the specified memory range.
    /// This function will fail if the source memory is not readable.
    pub fn read(&self, addr: VirtAddr, size: usize) -> Result<&[u8], Error> {
        self.read_with_perms(addr, size, Perm(PERM_READ))
    }

    /// Returns a slice with the data stored in the specified memory range.
    /// This function will fail if the source memory does not satisfy the
    /// expected permissions.
    pub fn read_with_perms(
        &self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<&[u8], Error> {
        // Check if the source memory range is readable.
        self.check_perms(addr, size, perms)?;

        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        self.memory
            .get(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))
    }

    /// Copy the bytes in `src` to the given memory address. This function
    /// does not check memory permissions and does not mark memory as dirty.
    pub fn poke(&mut self, addr: VirtAddr, src: &[u8]) -> Result<(), Error> {
        let size = src.len();

        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        self.memory
            .get_mut(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))?
            .copy_from_slice(src);
        Ok(())
    }

    /// Returns a slice with the data stored in the specified memory range.
    /// This function does not check memory permissions.
    pub fn peek(&self, addr: VirtAddr, size: usize) -> Result<&[u8], Error> {
        let end = addr
            .checked_add(size)
            .ok_or(Error::AddressIntegerOverflow(addr, size))?;

        self.memory
            .get(*addr..end)
            .ok_or(Error::InvalidAddress(addr, size))
    }

    /// Compute dirty blocks and bitmap. It does not check if the memory range
    /// is valid.
    fn update_dirty(&mut self, addr: VirtAddr, size: usize) {
        let block_start = *addr / DIRTY_BLOCK_SIZE;
        // Calculate the start of the next block. It takes into account corner
        // cases like `end` being equal to the start of the next block.
        let block_end =
            (*addr + size + (DIRTY_BLOCK_SIZE - 1)) / DIRTY_BLOCK_SIZE;

        for block in block_start..block_end {
            let idx = block / 64;
            let bit = block % 64;

            if self.dirty_bitmap[idx] & (1 << bit) == 0 {
                self.dirty_bitmap[idx] |= 1 << bit;
                self.dirty.push(block);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmu_new() {
        let mmu = Mmu::new(16);
        let want = Mmu {
            size: 16,
            memory: vec![0; 16],
            perms: vec![Perm(0); 16],
            dirty: vec![],
            dirty_bitmap: vec![0; 1],
        };

        assert_eq!(mmu, want);
    }

    #[test]
    #[should_panic]
    fn mmu_new_zero_size() {
        Mmu::new(0);
    }

    #[test]
    fn mmu_check_perms() {
        let mut mmu = Mmu::new(16);
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap();

        mmu.check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn mmu_check_perms_subset() {
        let mut mmu = Mmu::new(16);
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE)).unwrap();

        mmu.check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap();
    }

    #[test]
    fn mmu_check_perms_oob() {
        let mut mmu = Mmu::new(16);
        match mmu.set_perms(VirtAddr(5), 16, Perm(PERM_WRITE)) {
            Err(Error::InvalidAddress(..)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_check_perms_integer_overflow() {
        let mut mmu = Mmu::new(16);
        match mmu.set_perms(VirtAddr(usize::MAX), 1, Perm(PERM_WRITE)) {
            Err(Error::AddressIntegerOverflow(..)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_poke_peek() {
        let mut mmu = Mmu::new(4);
        mmu.poke(VirtAddr(0), &[1, 2, 3, 4]).unwrap();
        let got = mmu.peek(VirtAddr(0), 4).unwrap();

        assert_eq!(got, &[1, 2, 3, 4]);
    }

    #[test]
    fn mmu_write_read() {
        let mut mmu = Mmu::new(4);
        mmu.set_perms(VirtAddr(0), 4, Perm(PERM_READ | PERM_WRITE))
            .unwrap();
        mmu.write(VirtAddr(0), &[1, 2, 3, 4]).unwrap();
        let got = mmu.read(VirtAddr(0), 4).unwrap();

        assert_eq!(got, &[1, 2, 3, 4]);
    }

    #[test]
    fn mmu_write_not_allowed() {
        let mut mmu = Mmu::new(4);
        match mmu.write(VirtAddr(0), &[1, 2, 3, 4]) {
            Err(Error::NotAllowed(..)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_read_not_allowed() {
        let mmu = Mmu::new(4);
        match mmu.read(VirtAddr(0), 2) {
            Err(Error::NotAllowed(..)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_raw_after_write() {
        let mut mmu = Mmu::new(4);
        mmu.set_perms(VirtAddr(0), 3, Perm(PERM_WRITE | PERM_RAW))
            .unwrap();
        mmu.write(VirtAddr(0), &[1, 2]).unwrap();

        assert_eq!(&mmu.memory[..4], &[1, 2, 0, 0]);
        assert_eq!(
            &mmu.perms[..4],
            &[
                Perm(PERM_WRITE | PERM_READ),
                Perm(PERM_WRITE | PERM_READ),
                Perm(PERM_WRITE | PERM_RAW),
                Perm(0)
            ]
        );
    }

    #[test]
    fn mmu_raw_ok() {
        let mut mmu = Mmu::new(4);
        mmu.set_perms(VirtAddr(0), 2, Perm(PERM_READ | PERM_WRITE))
            .unwrap();
        mmu.set_perms(VirtAddr(2), 2, Perm(PERM_WRITE | PERM_RAW))
            .unwrap();
        mmu.write(VirtAddr(0), &[1, 2, 3, 4]).unwrap();
        let got = mmu.read(VirtAddr(0), 4).unwrap();

        assert_eq!(got, &[1, 2, 3, 4]);
    }

    #[test]
    fn mmu_raw_unitialized() {
        let mut mmu = Mmu::new(4);
        mmu.set_perms(VirtAddr(0), 2, Perm(PERM_READ)).unwrap();
        mmu.set_perms(VirtAddr(2), 2, Perm(PERM_WRITE | PERM_RAW))
            .unwrap();
        match mmu.read(VirtAddr(1), 2) {
            Err(Error::UnitializedMemory(_)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_raw_not_allowed() {
        let mut mmu = Mmu::new(4);
        mmu.set_perms(VirtAddr(0), 2, Perm(PERM_WRITE)).unwrap();
        mmu.set_perms(VirtAddr(2), 2, Perm(PERM_WRITE | PERM_RAW))
            .unwrap();
        match mmu.read(VirtAddr(1), 2) {
            Err(Error::NotAllowed(..)) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_reset() {
        let mmu = Mmu::new(1024 * 1024);
        let mut mmu_fork = mmu.fork();

        mmu_fork
            .set_perms(VirtAddr(1028), 4, Perm(PERM_WRITE))
            .unwrap();
        mmu_fork.write(VirtAddr(1028), &[1, 2, 3, 4]).unwrap();

        let got = mmu_fork.peek(VirtAddr(1028), 4).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);

        let got = mmu_fork.peek(VirtAddr(1028), 4).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }

    #[test]
    fn mmu_reset_two_blocks() {
        let mmu = Mmu::new(1024 * 1024);
        let mut mmu_fork = mmu.fork();

        mmu_fork
            .set_perms(VirtAddr(1022), 4, Perm(PERM_WRITE))
            .unwrap();
        mmu_fork.write(VirtAddr(1022), &[1, 2, 3, 4]).unwrap();

        let got = mmu_fork.peek(VirtAddr(1022), 4).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);

        let got = mmu_fork.peek(VirtAddr(1022), 4).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }

    #[test]
    fn mmu_reset_one_of_two_blocks() {
        let mmu = Mmu::new(1024 * 1024);
        let mut mmu_fork = mmu.fork();

        mmu_fork.poke(VirtAddr(1022), &[1, 2, 3, 4]).unwrap();
        mmu_fork
            .set_perms(VirtAddr(1024), 2, Perm(PERM_WRITE))
            .unwrap();

        let got = mmu_fork.peek(VirtAddr(1022), 4).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);

        let got = mmu_fork.peek(VirtAddr(1022), 4).unwrap();
        assert_eq!(got, &[1, 2, 0, 0]);
    }

    #[test]
    fn mmu_reset_all() {
        let mmu = Mmu::new(1024 * 1024);
        let mut mmu_fork = mmu.fork();

        mmu_fork
            .set_perms(VirtAddr(0), 1024 * 1024, Perm(PERM_WRITE | PERM_RAW))
            .unwrap();
        mmu_fork.write(VirtAddr(1028), &[1, 2, 3, 4]).unwrap();
        let got = mmu_fork.read(VirtAddr(1028), 4).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);
        let got = mmu_fork.peek(VirtAddr(4), 4).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }
}
