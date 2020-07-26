//! Emulated MMU with byte-level memory permissions able to detect unitialized
//! memory accesses.

use std::ops::Deref;

/// Executable memory. Aimed to be used with `Perm`.
pub const PERM_EXEC: u8 = 1;

/// Writable memory. Aimed to be used with `Perm`.
pub const PERM_WRITE: u8 = 1 << 1;

/// Readable memory. Aimed to be used with `Perm`.
pub const PERM_READ: u8 = 1 << 2;

/// Read-after-write memory. Aimed to be used with `Perm`.
///
/// This permission is assigned automatically when allocating memory. If a
/// memory position has this flag and is written, the READ permission will be
/// automatically assigned afterwards. This allows us to detect accesses to
/// unitialized memory.
pub const PERM_RAW: u8 = 1 << 3;

/// Block sized used for resetting and tracking memory which has been modified.
const DIRTY_BLOCK_SIZE: usize = 1024;

/// Memory permissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Perm(pub u8);

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

/// Memory errors.
#[derive(Debug)]
pub enum MemoryError {
    /// There is not enough free memory to allocate the requested size.
    OutOfMemory,

    /// Memory address is out of range.
    InvalidAddress,

    /// Read access to unitialized memory.
    UnitializedMemory,

    /// Integer overflow happened when operating with a memory address.
    IntegerOverflow,

    /// Invalid permissions.
    NotAllowed,
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

    /// End address of the allocated memory (non-inclusive).
    alloc_end: VirtAddr,
}

impl Mmu {
    /// Returns a new Mmu with a given memory `size`. `alloc_base` sets the
    /// initial base address for memory allocations
    ///
    /// # Panics
    ///
    /// This function panics if `size` is 0 or below `alloc_base`.
    pub fn new(size: usize, alloc_base: VirtAddr) -> Mmu {
        assert!(size > 0 && size >= *alloc_base, "invalid size");

        Mmu {
            size,
            memory: vec![0; size],
            perms: vec![Perm(0); size],
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0; size / DIRTY_BLOCK_SIZE / 64 + 1],
            alloc_end: alloc_base,
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
            alloc_end: self.alloc_end,
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

        self.alloc_end = other.alloc_end;
    }

    /// Allocate `size` bytes as WRITE. They will be set as RAW in order to
    /// detect accesses to unitialized memory. Returns the new top address.
    pub fn allocate(&mut self, size: usize) -> Result<VirtAddr, MemoryError> {
        if size == 0 {
            return Ok(self.alloc_end);
        }

        // We calculate the new allocation end, making sure that it does not
        // overflow. Otherwise, there could be problems when checking memory
        // boundaries.
        let new_alloc_end = match self.alloc_end.checked_add(size) {
            Some(addr) => addr,
            None => return Err(MemoryError::IntegerOverflow),
        };

        if new_alloc_end > self.size {
            return Err(MemoryError::OutOfMemory);
        }

        // Set the permissions of the allocated memory to WR and RAW. This will
        // allow us to detect futures accesses to unitialized memory. We use
        // the `set_perms_unchecked()` function here because the memory
        // boundaries have been previously checked.
        self.set_perms_unchecked(
            self.alloc_end,
            size,
            Perm(PERM_WRITE | PERM_RAW),
        );

        self.alloc_end = VirtAddr(new_alloc_end);

        Ok(self.alloc_end)
    }

    /// Set memory permissions in the given range without checking memory
    /// boundaries.
    fn set_perms_unchecked(
        &mut self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) {
        self.perms[*addr..*addr + size].iter_mut().for_each(|p| *p = perms);
        self.compute_dirty(addr, size);
    }

    /// Set memory permissions in the given range.
    pub fn set_perms(
        &mut self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<(), MemoryError> {
        // We must ensure that calculating the end adddress does not overflow.
        // Otherwise, there could be problems when checking memory boundaries.
        let end = match addr.checked_add(size) {
            Some(addr) => addr,
            None => return Err(MemoryError::IntegerOverflow),
        };

        if end > self.size {
            return Err(MemoryError::InvalidAddress);
        }

        self.set_perms_unchecked(addr, size, perms);
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
    ) -> Result<bool, MemoryError> {
        // We must ensure that calculating the end adddress does not overflow.
        // Otherwise, there could be problems when checking memory boundaries.
        let end = match addr.checked_add(size) {
            Some(addr) => addr,
            None => return Err(MemoryError::IntegerOverflow),
        };

        if end > self.size {
            return Err(MemoryError::InvalidAddress);
        }

        let result = self.perms[*addr..end]
            .iter()
            .all(|p| **p & *perms == *perms);
        Ok(result)
    }

    /// Copy the bytes in `src` to the given memory address. This function will
    /// fail if the destination memory is not writable.
    pub fn write(
        &mut self,
        addr: VirtAddr,
        src: &[u8],
    ) -> Result<(), MemoryError> {
        self.write_if_perms(addr, src, Perm(PERM_WRITE))
    }

    /// Copy the bytes in `src` to the given memory address. If the expected
    /// perms include `PERM_WRITE` and the memory position is marked as RAW,
    /// `PERM_READ` will be set.
    pub fn write_if_perms(
        &mut self,
        addr: VirtAddr,
        src: &[u8],
        perms: Perm,
    ) -> Result<(), MemoryError> {
        let size = src.len();

        // Check if expected permissions are compatible with memory
        // permissions. This also checks memory boundaries.
        if !self.check_perms(addr, size, perms)? {
            return Err(MemoryError::NotAllowed);
        }

        let end = *addr + size;

        self.memory[*addr..end].copy_from_slice(src);

        // Add PERM_READ in case of RAW.
        self.perms[*addr..end].iter_mut().for_each(|p| {
            if (*perms & PERM_WRITE != 0) && (**p & PERM_RAW != 0) {
                *p = Perm(**p | PERM_READ);
            }
        });

        self.compute_dirty(addr, size);

        Ok(())
    }

    /// Returns a slice with the data stored in the specified memory range.
    /// This function will fail if the source memory is not readable.
    pub fn read(
        &self,
        addr: VirtAddr,
        size: usize,
    ) -> Result<&[u8], MemoryError> {
        self.read_if_perms(addr, size, Perm(PERM_READ))
    }

    /// Returns a slice with the data stored in the specified memory range.
    pub fn read_if_perms(
        &self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<&[u8], MemoryError> {
        // Check if expected permissions are compatible with memory
        // permissions. This also checks memory boundaries.
        if !self.check_perms(addr, size, perms)? {
            return Err(MemoryError::NotAllowed);
        }

        // Return the requested memory range.
        let end = *addr + size;
        Ok(&self.memory[*addr..end])
    }

    /// Compute dirty blocks and bitmap. It does not check if the memory range
    /// is valid.
    fn compute_dirty(&mut self, addr: VirtAddr, size: usize) {
        let block_start = *addr / DIRTY_BLOCK_SIZE;
        // Calculate the start of the next block. It takes into account corner
        // cases like `end` being equal to the start of the next block.
        let block_end = (*addr + size + (DIRTY_BLOCK_SIZE - 1)) / DIRTY_BLOCK_SIZE;

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
        let mmu = Mmu::new(16, VirtAddr(0));
        let want = Mmu {
            size: 16,
            memory: vec![0; 16],
            perms: vec![Perm(0); 16],
            dirty: vec![],
            dirty_bitmap: vec![0; 1],
            alloc_end: VirtAddr(0),
        };

        assert_eq!(mmu, want);
    }

    #[test]
    #[should_panic]
    fn mmu_new_zero_size() {
        Mmu::new(0, VirtAddr(0));
    }

    #[test]
    #[should_panic]
    fn mmu_new_size_lt_alloc_init() {
        Mmu::new(0xffff, VirtAddr(0x10000));
    }

    #[test]
    fn mmu_check_perms() {
        let mut mmu = Mmu::new(16, VirtAddr(0));
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap();

        assert!(mmu
            .check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap());
    }

    #[test]
    fn mmu_check_perms_subset() {
        let mut mmu = Mmu::new(16, VirtAddr(0));
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE)).unwrap();

        assert!(!mmu
            .check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap());
    }

    #[test]
    fn mmu_check_perms_oob() {
        let mut mmu = Mmu::new(16, VirtAddr(0));
        match mmu.set_perms(VirtAddr(5), 16, Perm(PERM_WRITE)) {
            Err(MemoryError::InvalidAddress) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_write_read() {
        let mut mmu = Mmu::new(4, VirtAddr(0));
        mmu.allocate(4).unwrap();
        mmu.write(VirtAddr(0), &[1, 2, 3, 4]).unwrap();
        let got = mmu.read(VirtAddr(0), 4).unwrap();

        assert_eq!(got, &[1, 2, 3, 4]);
    }

    #[test]
    fn mmu_write_not_allowed() {
        let mut mmu = Mmu::new(4, VirtAddr(0));
        match mmu.write(VirtAddr(0), &[1, 2, 3, 4]) {
            Err(MemoryError::NotAllowed) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_read_not_allowed() {
        let mmu = Mmu::new(4, VirtAddr(0));
        match mmu.read(VirtAddr(0), 2) {
            Err(MemoryError::NotAllowed) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_raw_after_write() {
        let mut mmu = Mmu::new(4, VirtAddr(0));
        mmu.allocate(3).unwrap();
        mmu.write(VirtAddr(0), &[1, 2]).unwrap();

        assert_eq!(&mmu.memory[..4], &[1, 2, 0, 0]);
        assert_eq!(
            &mmu.perms[..4],
            &[
                Perm(PERM_WRITE | PERM_READ | PERM_RAW),
                Perm(PERM_WRITE | PERM_READ | PERM_RAW),
                Perm(PERM_WRITE | PERM_RAW),
                Perm(0)
            ]
        );
    }

    #[test]
    fn mmu_reset() {
        let mmu = Mmu::new(1024 * 1024, VirtAddr(0));
        let mut mmu_fork = mmu.fork();

        mmu_fork
            .write_if_perms(VirtAddr(1028), &[1, 2, 3, 4], Perm(0))
            .unwrap();
        let got = mmu_fork.read_if_perms(VirtAddr(1028), 4, Perm(0)).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);
        let got = mmu_fork.read_if_perms(VirtAddr(4), 4, Perm(0)).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }

    #[test]
    fn mmu_reset_two_blocks() {
        let mmu = Mmu::new(1024 * 1024, VirtAddr(0));
        let mut mmu_fork = mmu.fork();

        mmu_fork
            .write_if_perms(VirtAddr(1022), &[1, 2, 3, 4], Perm(0))
            .unwrap();
        let got = mmu_fork.read_if_perms(VirtAddr(1022), 4, Perm(0)).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);
        let got = mmu_fork.read_if_perms(VirtAddr(4), 4, Perm(0)).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }

    #[test]
    fn mmu_reset_allocate_all() {
        let mmu = Mmu::new(1024 * 1024, VirtAddr(0));
        let mut mmu_fork = mmu.fork();

        mmu_fork.allocate(1024 * 1024).unwrap();
        mmu_fork.write(VirtAddr(1028), &[1, 2, 3, 4]).unwrap();
        let got = mmu_fork.read(VirtAddr(1028), 4).unwrap();
        assert_eq!(got, &[1, 2, 3, 4]);

        mmu_fork.reset(&mmu);
        let got = mmu_fork.read_if_perms(VirtAddr(4), 4, Perm(0)).unwrap();
        assert_eq!(got, &[0, 0, 0, 0]);
    }
}
