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
    /// There is not room to allocate the requested size.
    OutOfMemory,

    /// Memory address is out of range.
    InvalidAddress,

    /// Read access to unitialized memory.
    UnitializedMemory,

    /// Integer overflow happened when operating with a memory address.
    IntegerOverflow,

    /// Access permissions are not compatible with memory permissions.
    NotAllowed,
}

/// Emulated memory management unit.
#[derive(Debug, PartialEq, Eq)]
pub struct Mmu {
    /// End address of the memory space (non-inclusive).
    memory_end: VirtAddr,

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
    /// Returns a new Mmu with a given maximum memory size.
    ///
    /// # Panics
    ///
    /// This function may panic if size is 0.
    pub fn new(size: usize) -> Mmu {
        // Size cannot be 0.
        assert!(size > 0, "invalid size");

        Mmu {
            memory_end: VirtAddr(size),
            memory: vec![0; size],
            perms: vec![Perm(0); size],
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0; size / DIRTY_BLOCK_SIZE / 64 + 1],
            alloc_end: VirtAddr(0),
        }
    }

    /// Returns a copy of the MMU. It marks all memory as clean in the new
    /// copy.
    pub fn fork(&self) -> Mmu {
        let size = self.memory.len();

        Mmu {
            memory_end: self.memory_end,
            memory: self.memory.clone(),
            perms: self.perms.clone(),
            dirty: Vec::with_capacity(size / DIRTY_BLOCK_SIZE + 1),
            dirty_bitmap: vec![0; size / DIRTY_BLOCK_SIZE / 64 + 1],
            alloc_end: self.alloc_end,
        }
    }

    /// Restores memory to the original state `other`, inclusing dirty marks.
    pub fn reset(&mut self, other: &Mmu) {
        for &block in &self.dirty {
            // Compute start and end indices of the dirty block.
            let start = block * DIRTY_BLOCK_SIZE;
            let end = (block + 1) * DIRTY_BLOCK_SIZE;

            // Zero bitmap.
            self.dirty_bitmap[block / 64] = 0;

            // Restore memory.
            self.memory[start..end].copy_from_slice(&other.memory[start..end]);

            // Restore permissions.
            self.perms[start..end].copy_from_slice(&other.perms[start..end]);
        }

        // Clear dirty marks.
        self.dirty.clear();

        // Restore allocation's end address.
        self.alloc_end = other.alloc_end;
    }

    /// Allocate `size` bytes as RW. They will be marked initially as RAW in
    /// order to detect accesses to unitialized memory. Returns the new
    /// allocation limit.
    pub fn allocate(&mut self, size: usize) -> Result<VirtAddr, MemoryError> {
        // If size is 0, we just return the current allocation limit.
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
        let new_alloc_end = VirtAddr(new_alloc_end);

        // the new range must be within the hard memory boundaries imposed by
        // the MMU.
        if new_alloc_end > self.memory_end {
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

        // Update alloc_end with the new value.
        self.alloc_end = new_alloc_end;

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
        // Set permissions for every byte in the memory range.
        let end = *addr + size;
        self.perms[*addr..end].iter_mut().for_each(|p| *p = perms);

        // Compute dirty blocks and bitmap.
        let block_start = *addr / DIRTY_BLOCK_SIZE;
        let block_end = if end % DIRTY_BLOCK_SIZE == 0 {
            end / DIRTY_BLOCK_SIZE
        } else {
            end / DIRTY_BLOCK_SIZE + 1
        };

        for block in block_start..block_end {
            let idx = block / 64;
            let bit = block % 64;

            if self.dirty_bitmap[idx] & (1 << bit) == 0 {
                self.dirty_bitmap[idx] |= 1 << bit;
                self.dirty.push(block);
            }
        }
    }

    /// Set memory permissions in the given range. This function may return a
    /// `MemoryError` if the range is not valid.
    pub fn set_perms(
        &mut self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<(), MemoryError> {
        // Check that the range is within the limits of the memory.
        let new_memory_end = match addr.checked_add(size) {
            Some(x) => x,
            None => return Err(MemoryError::IntegerOverflow),
        };
        if new_memory_end > *self.memory_end {
            return Err(MemoryError::InvalidAddress);
        }

        // Now that we know that the address range is valid, we can set the
        // specified permissions.
        self.set_perms_unchecked(addr, size, perms);
        Ok(())
    }

    /// Given a memory address, a size and the expected permissions, this
    /// function will return true if every byte in the specified region
    /// satisfies those permissions. Otherwise, the function will return false.
    /// A `MemoryError` may be returned if the memory range is out of bounds.
    pub fn check_perms(
        &self,
        addr: VirtAddr,
        size: usize,
        perms: Perm,
    ) -> Result<bool, MemoryError> {
        // Check that the range is within the limits of the memory.
        let new_memory_end = match addr.checked_add(size) {
            Some(x) => x,
            None => return Err(MemoryError::IntegerOverflow),
        };
        if new_memory_end > *self.memory_end {
            return Err(MemoryError::InvalidAddress);
        }

        // Check that every byte in the range has at least the expected
        // permissions.
        let end = *addr + size;
        let result = self.perms[*addr..end]
            .iter()
            .all(|p| **p & *perms == *perms);
        Ok(result)
    }

    /// Write bytes in memory range.
    pub fn write(
        &mut self,
        addr: VirtAddr,
        src: &[u8],
    ) -> Result<(), MemoryError> {
        // Ensure that the memory region is writable.
        self.write_if_perms(addr, src, Perm(PERM_WRITE))
    }

    /// Write bytes in memory range, checking the permissions first. If the
    /// expected perms include `PERM_WRITE` and the memory positions is marked
    /// as RAW, the perm `PERM_READ` will be set.
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

        // Update memory.
        self.memory[*addr..end].copy_from_slice(src);

        // Add PERM_READ in case of RAW.
        self.perms[*addr..end].iter_mut().for_each(|p| {
            if (*perms & PERM_WRITE != 0) && (**p & PERM_RAW != 0) {
                *p = Perm(**p | PERM_READ);
            }
        });

        // Compute dirty blocks and bitmap.
        let block_start = *addr / DIRTY_BLOCK_SIZE;
        let block_end = if end % DIRTY_BLOCK_SIZE == 0 {
            end / DIRTY_BLOCK_SIZE
        } else {
            end / DIRTY_BLOCK_SIZE + 1
        };

        for block in block_start..block_end {
            let idx = block / 64;
            let bit = block % 64;

            // Push block to the dirty list if it's not already there.
            if self.dirty_bitmap[idx] & (1 << bit) == 0 {
                self.dirty_bitmap[idx] |= 1 << bit;
                self.dirty.push(block);
            }
        }

        Ok(())
    }

    /// Returns bytes in memory range.
    pub fn read(
        &self,
        addr: VirtAddr,
        size: usize,
    ) -> Result<&[u8], MemoryError> {
        // Ensure that the memory region is readable.
        self.read_if_perms(addr, size, Perm(PERM_READ))
    }

    /// Returns bytes in memory range, checking the permissions first.
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mmu_new() {
        let mmu = Mmu::new(16);
        let want = Mmu {
            memory_end: VirtAddr(16),
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
        Mmu::new(0);
    }

    #[test]
    fn mmu_check_perms() {
        let mut mmu = Mmu::new(16);
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap();

        assert!(mmu
            .check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap());
    }

    #[test]
    fn mmu_check_perms_subset() {
        let mut mmu = Mmu::new(16);
        mmu.set_perms(VirtAddr(0), 8, Perm(PERM_WRITE)).unwrap();

        assert!(!mmu
            .check_perms(VirtAddr(0), 8, Perm(PERM_WRITE | PERM_READ))
            .unwrap());
    }

    #[test]
    fn mmu_check_perms_oob() {
        let mut mmu = Mmu::new(16);
        match mmu.set_perms(VirtAddr(5), 16, Perm(PERM_WRITE)) {
            Err(MemoryError::InvalidAddress) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_write_read() {
        let mut mmu = Mmu::new(4);
        mmu.allocate(4).unwrap();
        mmu.write(VirtAddr(0), &[1, 2, 3, 4]).unwrap();
        let got = mmu.read(VirtAddr(0), 4).unwrap();

        assert_eq!(got, &[1, 2, 3, 4]);
    }

    #[test]
    fn mmu_write_not_allowed() {
        let mut mmu = Mmu::new(4);
        match mmu.write(VirtAddr(0), &[1, 2, 3, 4]) {
            Err(MemoryError::NotAllowed) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_read_not_allowed() {
        let mmu = Mmu::new(4);
        match mmu.read(VirtAddr(0), 2) {
            Err(MemoryError::NotAllowed) => return,
            Err(err) => panic!("Wrong error {:?}", err),
            _ => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn mmu_raw_after_write() {
        let mut mmu = Mmu::new(4);
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
        let mmu = Mmu::new(1024 * 1024);
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
        let mmu = Mmu::new(1024 * 1024);
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
        let mmu = Mmu::new(1024 * 1024);
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
