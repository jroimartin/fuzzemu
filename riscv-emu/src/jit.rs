//! Useful JIT compilation primitives.

use std::collections::HashMap;
use std::fmt;

use crate::mmu::VirtAddr;

/// Error due to JIT cache operations.
#[derive(Debug)]
pub enum Error {
    InvalidAddress,
    OutOfMemory,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::InvalidAddress => write!(f, "invalid address"),
            Error::OutOfMemory => write!(f, "out of memory"),
        }
    }
}

/// Memory map used to store the compiled code.
pub struct JitMemory {
    /// Allocated RWX memory map containing the compiled code.
    memory: &'static mut [u8],

    /// Cursor pointing to the next free area.
    cursor: usize,

    /// Used for block deduplication. Mapping between a given slice of bytes
    /// and the pointer to the corresponding address in the JIT memory map.
    dedup: HashMap<Vec<u8>, usize>,
}

/// Jit cache.
pub struct JitCache {
    /// Mapping between a program address and the pointer to the corresponding
    /// compiled code. A null pointer means that the block has not been lifted
    /// yet.
    lookup_table: Vec<usize>,

    /// Memory map containing the compiled code.
    jit_memory: JitMemory,
}

/// Creates a memory map of size `size` with RWX permissions.
///
/// TODO(rm): Port to other OS without mmap (i.e. MS Windows).
fn alloc_rwx(size: usize) -> &'static mut [u8] {
    unsafe {
        let rwx_ptr = libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
            libc::MAP_ANONYMOUS | libc::MAP_PRIVATE,
            -1,
            0,
        );

        std::slice::from_raw_parts_mut(rwx_ptr as *mut u8, size)
    }
}

impl JitCache {
    /// Returns a new JIT cache. `exec_size` is the size of the executable
    /// memory. `jit_size` is the size of the memory allocated to store the
    /// compiled code.
    pub fn new(exec_size: usize, jit_size: usize) -> JitCache {
        // The internal cache will have as many entries as `mem_size / 4`,
        // given that all the instructions in RISC-V rv64i are 4 bytes long.
        let size = exec_size / 4;

        let jit_memory = JitMemory {
            memory: alloc_rwx(jit_size),
            cursor: 0,
            dedup: HashMap::new(),
        };

        JitCache {
            lookup_table: vec![0; size],
            jit_memory,
        }
    }

    /// Returns a pointer to the lifted block corresponding to the virtual
    /// address `addr`. If the block is not in the chache or the address is not
    /// valid, None is returned.
    ///
    /// The address `addr` must be 4-byte aligned.
    pub fn lookup(&self, addr: VirtAddr) -> Option<*const u8> {
        if *addr & 3 != 0 {
            return None;
        }

        match self.lookup_table.get(*addr / 4) {
            Some(ptr) if *ptr != 0 => Some(*ptr as *const u8),
            _ => None,
        }
    }

    /// Inserts a new block in the cache. The function returns a pointer to
    /// this new block. If the block was already present, the function returns
    /// a pointer to the already existing one.
    ///
    /// If the address is out of bounds or it is not 4-byte aligned or there is
    /// no more space in the JIT memory area, the block won't be inserted into
    /// the cache and an `Error` is returned.
    pub fn insert(
        &mut self,
        addr: VirtAddr,
        block: Vec<u8>,
    ) -> Result<*const u8, Error> {
        if *addr & 3 != 0 {
            return Err(Error::InvalidAddress);
        }

        let idx = *addr / 4;

        // Check if the block already exists.
        let ptr = self.lookup_table.get(idx).ok_or(Error::InvalidAddress)?;

        if *ptr != 0 {
            return Ok(*ptr as *const u8);
        }

        // If the block does not exist, create a new mapping.
        if let Some(ptr) = self.jit_memory.dedup.get(&block) {
            // If the dedup hash map contains the key, map the vaddr with the
            // already existing block.
            self.lookup_table[idx] = *ptr;

            Ok(*ptr as *const u8)
        } else {
            // New block.
            let size = block.len();

            let start = self.jit_memory.cursor;
            let end = start + size;

            // Check that there is enough free memory for the new block.
            if end > self.jit_memory.memory.len() {
                return Err(Error::OutOfMemory);
            }

            // Copy the new block into the JIT memory map.
            self.jit_memory.memory[start..end].copy_from_slice(&block);

            // Update the cursor
            self.jit_memory.cursor += size;

            // Get a pointer to the new block.
            let ptr = self.jit_memory.memory[start..end].as_ptr();

            // Update the dedup hash map.
            self.jit_memory.dedup.insert(block, ptr as usize);

            // Update the lookup table.
            self.lookup_table[idx] = ptr as usize;

            Ok(ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitcache_insert_exec() {
        let code = r#"
            BITS 64

            mov rbx, 0x1337
            ret
        "#;
        let block = nasm::assemble(code).unwrap();

        let mut cache = JitCache::new(0x10, 0x1000);
        let block_ptr = cache.insert(VirtAddr(0), block).unwrap();

        let result: u64;

        unsafe {
            asm!(
                "call rax",
                in("rax") block_ptr,
                out("rbx") result,
            );
        }

        assert_eq!(result, 0x1337);
    }

    #[test]
    fn jitcache_insert_dedup() {
        let mut cache = JitCache::new(0x10, 0x1000);

        let block_ptr = cache.insert(VirtAddr(0x0), vec![0x90]).unwrap();
        let block2_ptr = cache.insert(VirtAddr(0x4), vec![0x90]).unwrap();

        assert_eq!(block_ptr, block2_ptr);
    }

    #[test]
    fn jitcache_insert_invalid_address() {
        let mut cache = JitCache::new(0x10, 0x1000);

        match cache.insert(VirtAddr(0x3), vec![0x90]) {
            Err(Error::InvalidAddress) => return,
            Err(_) => panic!("Wrong error"),
            Ok(_) => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn jitcache_insert_oom() {
        let mut cache = JitCache::new(0x10, 0x2);

        match cache.insert(VirtAddr(0x0), vec![0x90; 3]) {
            Err(Error::OutOfMemory) => return,
            Err(_) => panic!("Wrong error"),
            Ok(_) => panic!("The function didn't return an error"),
        }
    }

    #[test]
    fn jitcache_use_all_memory() {
        let mut cache = JitCache::new(0x10, 0x4);
        cache
            .insert(VirtAddr(0x0), vec![0x00, 0x01, 0x02, 0x03])
            .unwrap();
    }

    #[test]
    fn jitcache_lookup() {
        let mut cache = JitCache::new(0x10, 0x1000);

        let block_ptr = cache.insert(VirtAddr(0x4), vec![0x90]).unwrap();

        assert_eq!(Some(block_ptr), cache.lookup(VirtAddr(0x4)));
        assert_eq!(None, cache.lookup(VirtAddr(0x0)));
        assert_eq!(None, cache.lookup(VirtAddr(0x3)));
        assert_eq!(None, cache.lookup(VirtAddr(0x20)));
    }

    #[test]
    fn jitcache_insert_lookup_exec() {
        let mut cache = JitCache::new(0x10, 0x1000);

        let code = r#"
            BITS 64

            mov rcx, 0x1337
            ret
        "#;
        let block = nasm::assemble(code).unwrap();
        cache.insert(VirtAddr(0), block).unwrap();

        let code = r#"
            BITS 64

            mov rdx, 0xc4f3
            ret
        "#;
        let block = nasm::assemble(code).unwrap();
        cache.insert(VirtAddr(4), block).unwrap();

        let result_1337: u64;
        let result_c4f3: u64;

        let block_rcx_1337_ptr = cache.lookup(VirtAddr(0)).unwrap();
        let block_rdx_c4f3_ptr = cache.lookup(VirtAddr(4)).unwrap();

        unsafe {
            asm!(
                "call rax",
                "call rbx",
                in("rax") block_rcx_1337_ptr,
                in("rbx") block_rdx_c4f3_ptr,
                out("rcx") result_1337,
                out("rdx") result_c4f3,
            );
        }
        assert_eq!(result_1337, 0x1337);
        assert_eq!(result_c4f3, 0xc4f3);
    }
}
