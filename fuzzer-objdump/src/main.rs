//! Fuzzer based on a RISC-V emulator.

use std::cmp;
use std::fmt;
use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::jit::JitCache;
use riscv_emu::mmu::{
    self, Mmu, Perm, VirtAddr, PERM_RAW, PERM_READ, PERM_WRITE,
};

/// If `true`, print debug messages.
const DEBUG: bool = false;

/// If `true`, run one fuzz case using one thread and panic afterwards.
const DEBUG_ONE: bool = false;

/// If `true`, print stdout/stderr output.
const DEBUG_OUTPUT: bool = false;

/// Number of threads to spawn.
const NUM_THREADS: usize = 1;

/// Memory size of the VM.
const VM_MEM_SIZE: usize = 32 * 1024 * 1024;

/// Memory size of the JIT cache.
const JIT_CACHE_SIZE: usize = 32 * 1024 * 1024;

/// Amount of memory reserved for the stack at the end of the VM memory. Note
/// that 1024 bytes will be reserved to store the program arguments, so this
/// value must be bigger than 1024.
const STACK_SIZE: usize = 1024 * 1024;

/// If `true`, allocate memory with WRITE|RAW permissions, so unitialized
/// memory accesses are detected. Otherwise, allocate memory with WRITE|READ
/// permissions.
const CHECK_RAW: bool = false;

/// If `true`, execute the target program using JIT compilation.
const USE_JIT: bool = true;

/// Fuzzer's exit reason.
#[derive(Debug)]
enum FuzzExit {
    ProgramExit(u64),
    SyscallError,

    MmuError(mmu::Error),
    VmExit(VmExit),
}

impl fmt::Display for FuzzExit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuzzExit::ProgramExit(code) => write!(f, "program exit: {}", code),
            FuzzExit::SyscallError => write!(f, "syscall error"),
            FuzzExit::MmuError(err) => write!(f, "MMU error: {}", err),
            FuzzExit::VmExit(vmexit) => write!(f, "VM exit: {}", vmexit),
        }
    }
}

impl From<mmu::Error> for FuzzExit {
    fn from(error: mmu::Error) -> FuzzExit {
        FuzzExit::MmuError(error)
    }
}

impl From<VmExit> for FuzzExit {
    fn from(vmexit: VmExit) -> FuzzExit {
        FuzzExit::VmExit(vmexit)
    }
}

/// Statistics recorded during the fuzzing session.
#[derive(Default)]
struct Stats {
    /// Total number of fuzz cases.
    fuzz_cases: u64,

    /// Total number of crashes.
    crashes: u64,

    /// Total of executed instructions.
    total_inst: u64,

    /// Total number of CPU cycles.
    total_cycles: u64,

    /// Total number of CPU cylles spent resetting the guest.
    reset_cycles: u64,

    /// Total number of CPU cycles spent in the VM.
    vm_cycles: u64,

    /// Total number of CPU cycles spent in handling syscalls.
    syscall_cycles: u64,
}

/// Returns the current value of the Timestamp Counter.
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// InputFile represents the file opened by objdump.
struct InputFile {
    /// Contents of the file.
    contents: &'static [u8],

    /// Current read/write position in the file.
    cursor: usize,
}

/// A Fuzzer represents a instance of our fuzzer. It links everything together,
/// for instance, memory allocation, file handling, statistics, etc.
struct Fuzzer {
    /// Initial state of the emulator. Shared among all fuzzer instances.
    emu_init: Arc<Emulator>,

    /// Current state of the emulator.
    emu: Emulator,

    /// Global statistics. Shared among all fuzzer instances.
    stats: Arc<Mutex<Stats>>,

    /// The initial program break.
    brk_addr_init: VirtAddr,

    /// The current program break.
    brk_addr: VirtAddr,

    /// Input file. The fuzzer only supports one open file, which is enough for
    /// our use case.
    input_file: Option<InputFile>,
}

impl Fuzzer {
    /// Returns a new fuzzer instance.
    fn new(
        emu_init: Arc<Emulator>,
        brk_addr_init: VirtAddr,
        stats: Arc<Mutex<Stats>>,
    ) -> Fuzzer {
        let emu = emu_init.fork();

        Fuzzer {
            emu_init,
            emu,
            stats,
            brk_addr_init,
            brk_addr: brk_addr_init,
            input_file: None,
        }
    }

    /// Start a fuzzer worker. Normally, one fuzzer per core is spawned.
    fn go(mut self) {
        loop {
            let batch_start = rdtsc();

            let mut local_stats = Stats::default();

            // Update global stats every 500M cycles.
            while rdtsc() - batch_start < 500_000_000 {
                // Reset memory to the initial state before starting fuzzing.
                let reset_start = rdtsc();
                self.reset();
                local_stats.reset_cycles += rdtsc() - reset_start;

                // Run the target and handle the result.
                let fcexit = self.run_fc(&mut local_stats);
                self.handle_fcexit(fcexit, &mut local_stats);

                if DEBUG_ONE {
                    panic!("DEBUG_ONE is enabled");
                }

                // Update local stats.
                local_stats.fuzz_cases += 1;
            }

            // Update global stats.
            let mut stats = self.stats.lock().unwrap();

            stats.fuzz_cases += local_stats.fuzz_cases;
            stats.crashes += local_stats.crashes;
            stats.total_inst += local_stats.total_inst;
            stats.reset_cycles += local_stats.reset_cycles;
            stats.vm_cycles += local_stats.vm_cycles;
            stats.syscall_cycles += local_stats.syscall_cycles;

            stats.total_cycles += rdtsc() - batch_start;
        }
    }

    /// Runs a single fuzz_case.
    fn run_fc(&mut self, stats: &mut Stats) -> FuzzExit {
        loop {
            let mut total_inst = 0;

            let vm_start = rdtsc();
            let run_result = self.emu.run(&mut total_inst);
            stats.vm_cycles += rdtsc() - vm_start;

            stats.total_inst += total_inst;

            match run_result {
                Err(VmExit::Ecall) => {
                    let syscall_start = rdtsc();
                    let syscall_result = self.syscall_dispatcher();
                    stats.syscall_cycles += rdtsc() - syscall_start;

                    if let Err(err) = syscall_result {
                        break err;
                    } else {
                        continue;
                    }
                }
                Err(err) => break err.into(),
                _ => unreachable!(),
            }
        }
    }

    /// Handle the results obtained by the fuzz case.
    fn handle_fcexit(&mut self, fcexit: FuzzExit, stats: &mut Stats) {
        // TODO(rm): Handle crashes as well as unexpected errors.

        match fcexit {
            FuzzExit::ProgramExit(code) => {
                if DEBUG {
                    eprintln!("program exited with {}", code);
                }
            }
            FuzzExit::VmExit(vmexit) => {
                if DEBUG {
                    let pc = self.emu.reg(RegAlias::Pc).unwrap();
                    eprintln!("[{:#010x}] {}\n{}", pc, vmexit, self.emu);
                }

                if vmexit_is_crash(&vmexit) {
                    stats.crashes += 1;
                }
            }
            err => todo!("unhandled error: {}", err),
        }
    }

    /// Restore the fuzzer to its initial state.
    fn reset(&mut self) {
        self.emu.reset(&self.emu_init);
        self.brk_addr = self.brk_addr_init;
        self.input_file = None;
    }

    /// Dispatches syscalls when the emulator exits with `VmExit::Ecall`,
    /// redirecting the execution to the relevant syscall handler.
    fn syscall_dispatcher(&mut self) -> Result<(), FuzzExit> {
        let syscall_number = self.emu.reg(RegAlias::A7)?;
        let pc = self.emu.reg(RegAlias::Pc)?;

        match syscall_number {
            57 => self.syscall_close()?,
            62 => self.syscall_lseek()?,
            63 => self.syscall_read()?,
            64 => self.syscall_write()?,
            80 => self.syscall_fstat()?,
            93 => self.syscall_exit()?,
            214 => self.syscall_brk()?,
            1024 => self.syscall_open()?,
            1038 => self.syscall_stat()?,
            _ => unimplemented!("[{:#010x}] unknown syscall", pc),
        };

        // The syscall dispatcher is in charge of advancing PC.
        self.emu.set_reg(RegAlias::Pc, pc.wrapping_add(4))?;

        Ok(())
    }

    /// close syscall handle.
    fn syscall_close(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.reg(RegAlias::A0)?;

        if DEBUG {
            eprintln!("close: fd={}", fd);
        }

        // The file descriptor 1337 is assigned to the input file.
        // After close, None is assigned. For any other file
        // descriptor, we just return 0 (success).
        if fd == 1337 {
            self.input_file = None;
        }
        self.emu.set_reg(RegAlias::A0, 0)?;

        Ok(())
    }

    /// lseek syscall handle.
    fn syscall_lseek(&mut self) -> Result<(), FuzzExit> {
        const SEEK_SET: u64 = 0;
        const SEEK_CUR: u64 = 1;
        const SEEK_END: u64 = 2;

        let fd = self.emu.reg(RegAlias::A0)?;
        let offset = self.emu.reg(RegAlias::A1)?;
        let whence = self.emu.reg(RegAlias::A2)?;

        if DEBUG {
            eprintln!("lseek: fd={} offset={} whence={}", fd, offset, whence);
        }

        // lseek is only allowed in the input file, for any other fd, we return
        // error.
        if fd != 1337 {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        // lseek also returns error if the input_file is not open (i.e.
        // self.input_file == None).
        if let Some(input_file) = &mut self.input_file {
            let contents_len = input_file.contents.len();

            let new_cursor = match whence {
                SEEK_SET => offset as usize,
                SEEK_CUR => input_file.cursor + offset as usize,
                SEEK_END => (contents_len as i64 + offset as i64) as usize,
                _ => return Err(FuzzExit::SyscallError),
            };

            // Although it should be possible, we do not support seeking beyond
            // the end of the file, so we return with error in that case.
            if new_cursor >= contents_len {
                self.emu.set_reg(RegAlias::A0, !0)?;
                return Ok(());
            }

            input_file.cursor = new_cursor;
            self.emu.set_reg(RegAlias::A0, input_file.cursor as u64)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// read syscall handle.
    fn syscall_read(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.reg(RegAlias::A0)?;
        let buf = self.emu.reg(RegAlias::A1)?;
        let count = self.emu.reg(RegAlias::A2)? as usize;

        let buf = VirtAddr(buf as usize);

        if DEBUG {
            eprintln!("read: fd={} count={}", fd, count);
        }

        // read is only allowed in the input file, for any other fd, we return
        // error.
        if fd != 1337 {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        // read also returns error if the input_file is not open (i.e.
        // self.input_file == None).
        if let Some(input_file) = &mut self.input_file {
            let cursor = input_file.cursor;
            let remaining = input_file.contents.len() - cursor;

            // count cannot be bigger than the number of remaining bytes.
            let count = cmp::min(count, remaining);

            // checked_add is not needed here because the cursor is already
            // sanitized by the lseek handler.
            self.emu
                .mmu_mut()
                .write(buf, &input_file.contents[cursor..cursor + count])?;

            input_file.cursor += count as usize;
            self.emu.set_reg(RegAlias::A0, count as u64)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// write syscall handle.
    fn syscall_write(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.reg(RegAlias::A0)?;
        let buf = self.emu.reg(RegAlias::A1)?;
        let count = self.emu.reg(RegAlias::A2)?;

        if DEBUG {
            eprintln!("write: fd={} count={}", fd, count);
        }

        // We only support writing to stdout and stderr.
        if fd != 1 && fd != 2 {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        if DEBUG_OUTPUT {
            let mut bytes = vec![0; count as usize];
            self.emu.mmu().read(VirtAddr(buf as usize), &mut bytes)?;
            let buf = String::from_utf8(bytes)
                .unwrap_or_else(|_| String::from("(invalid string)"));
            eprint!("{}", buf);
        }

        self.emu.set_reg(RegAlias::A0, count)?;

        Ok(())
    }

    /// fstat syscall handle.
    fn syscall_fstat(&mut self) -> Result<(), FuzzExit> {
        // This is not a real implementation, but it's enough to bypass the
        // objdump checks. We just return with error.
        let fd = self.emu.reg(RegAlias::A0)?;

        if DEBUG {
            eprintln!("fstat: fd={}", fd);
        }

        self.emu.set_reg(RegAlias::A0, !0)?;

        Ok(())
    }

    /// exit syscall handle.
    fn syscall_exit(&mut self) -> Result<(), FuzzExit> {
        let code = self.emu.reg(RegAlias::A0)?;

        if DEBUG {
            eprintln!("exit: code={}", code);
        }

        Err(FuzzExit::ProgramExit(code))
    }

    /// brk syscall handle.
    fn syscall_brk(&mut self) -> Result<(), FuzzExit> {
        let addr = self.emu.reg(RegAlias::A0)? as usize;

        if DEBUG {
            eprintln!(
                "brk: addr={:#010x} (previous addr={})",
                addr, self.brk_addr
            );
        }

        // If address is zero, brk returns the current brk address.
        if addr == 0 {
            self.emu.set_reg(RegAlias::A0, *self.brk_addr as u64)?;
            return Ok(());
        }

        // We don't case about the returned values. On success, brk returns the
        // requested address, which is already in A0.
        if addr >= *self.brk_addr {
            // Allocate
            let size = addr - *self.brk_addr;
            if DEBUG {
                eprintln!("brk: allocate({:#x})", size);
            }
            self.allocate(size)?;
        } else {
            // Free
            let size = *self.brk_addr - addr;
            if DEBUG {
                eprintln!("brk: free({:#x})", size);
            }
            self.free(size)?;
        }

        Ok(())
    }

    /// open syscall handle.
    fn syscall_open(&mut self) -> Result<(), FuzzExit> {
        // We only care about the path name, so we don't even consider the
        // other arguments.
        let path_name = self.emu.reg(RegAlias::A0)?;

        let path_name = self.get_cstr(VirtAddr(path_name as usize))?;
        let path_name = String::from_utf8_lossy(&path_name);

        if DEBUG {
            eprintln!("open: pathname={}", path_name);
        }

        // We only allow to open the input file, which fd is always 1337. If
        // the path does not correspond to the expected input file, or the file
        // is already open, then return error.
        if path_name != "input_file" || self.input_file.is_some() {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        let input_file = InputFile {
            cursor: 0,
            contents: include_bytes!("../../test-targets/xauth"),
        };

        self.input_file = Some(input_file);
        self.emu.set_reg(RegAlias::A0, 1337)?;

        Ok(())
    }

    /// stat syscall handle.
    fn syscall_stat(&mut self) -> Result<(), FuzzExit> {
        // This is not a real implementation, but it's enough to bypass the
        // objdump checks.
        let path_name = self.emu.reg(RegAlias::A0)?;
        let statbuf = self.emu.reg(RegAlias::A1)?;

        let path_name = self.get_cstr(VirtAddr(path_name as usize))?;
        let path_name = String::from_utf8_lossy(&path_name);

        if DEBUG {
            eprintln!("stat: path_name={}", path_name);
        }

        // bucom.c:621, objdump.c:572
        // stat() >= 0
        // statbuf.st_mode == _IFREG
        // statbuf.st_size > 0
        //
        // Received struct:
        //   https://github.com/riscv/riscv-newlib/blob/f289cef6be67da67b2d97a47d6576fa7e6b4c858/libgloss/riscv/kernel_stat.h

        // Set st_mode
        let st_mode_addr =
            statbuf.checked_add(16).ok_or(FuzzExit::SyscallError)?;
        let st_mode_addr = VirtAddr(st_mode_addr as usize);
        self.emu.mmu_mut().write_int::<u32>(st_mode_addr, 0x8000)?;

        // Set st_size
        let st_size_addr =
            statbuf.checked_add(48).ok_or(FuzzExit::SyscallError)?;
        let st_size_addr = VirtAddr(st_size_addr as usize);
        self.emu.mmu_mut().write_int::<u64>(st_size_addr, 0x1337)?;

        self.emu.set_reg(RegAlias::A0, 0)?;

        Ok(())
    }

    /// Allocates new memory. Allocation is done by incrementing the program's
    /// data space by `size` bytes. This function returns the previous brk
    /// address.
    ///
    /// If CHECK_RAW is true, the new memory has RAW|WRITE permissions, so
    /// unitialized memory accesses are detected. On the other hand, if
    /// CHECK_RAW is false, the new memory has READ|WRITE permissions.
    fn allocate(&mut self, size: usize) -> Result<VirtAddr, FuzzExit> {
        if size == 0 {
            return Ok(self.brk_addr);
        }

        let perms = if CHECK_RAW {
            Perm(PERM_RAW | PERM_WRITE)
        } else {
            Perm(PERM_READ | PERM_WRITE)
        };

        self.emu.mmu_mut().set_perms(self.brk_addr, size, perms)?;

        let prev_brk_addr = self.brk_addr;

        // checked_add() is not needed here because this has been already
        // checked in the previous call to set_perms().
        self.brk_addr = VirtAddr(*prev_brk_addr + size);

        Ok(prev_brk_addr)
    }

    /// Deallocate memory. This is done by decreasing the program's data space
    /// by `size` bytes and removing all the perissions of the freed region.
    fn free(&mut self, size: usize) -> Result<(), FuzzExit> {
        if size == 0 {
            return Ok(());
        }

        let new_brk_addr = self
            .brk_addr
            .checked_sub(size)
            .ok_or(FuzzExit::SyscallError)?;
        let new_brk_addr = VirtAddr(new_brk_addr);

        self.emu.mmu_mut().set_perms(new_brk_addr, size, Perm(0))?;

        self.brk_addr = new_brk_addr;

        Ok(())
    }

    /// Reads a C string (NULL terminated) from memory at `addr`.
    fn get_cstr(&self, addr: VirtAddr) -> Result<Vec<u8>, FuzzExit> {
        let mut result = Vec::new();
        let mut curaddr = addr;
        loop {
            let ch = self.emu.mmu().read_int::<u8>(curaddr)?;
            if ch == 0 {
                break Ok(result);
            }
            result.push(ch);
            *curaddr = curaddr.checked_add(1).ok_or(
                mmu::Error::AddressIntegerOverflow {
                    addr: curaddr,
                    size: 1,
                },
            )?;
        }
    }
}

/// Returns true if the VmExit variant corresponds to a crash.
pub fn vmexit_is_crash(vmexit: &VmExit) -> bool {
    false
}

/// Set up a stack with a size of `STACK_SIZE` bytes. It also configures
/// the command line argumets passed to the program.
///
/// # Panics
///
/// This function will panic if the memory size of the VM is not higher than
/// `STACK_SIZE`.
fn setup_stack(emu: &mut Emulator) -> Result<(), FuzzExit> {
    let mem_size = emu.mmu().size();

    assert!(mem_size > STACK_SIZE, "the VM memory is not big enough");

    let argv_base: usize = mem_size - 512;
    let stack_init: usize = mem_size - 1024;

    // Set the permissions of the stack to RW.
    emu.mmu_mut().set_perms(
        VirtAddr(mem_size - STACK_SIZE),
        STACK_SIZE,
        Perm(PERM_READ | PERM_WRITE),
    )?;

    // Store program args
    emu.mmu_mut().poke(VirtAddr(argv_base), b"objdump\x00")?;
    emu.mmu_mut().poke(VirtAddr(argv_base + 32), b"-x\x00")?;
    emu.mmu_mut()
        .poke(VirtAddr(argv_base + 64), b"input_file\x00")?;

    // Store argc
    emu.mmu_mut().poke_int::<u64>(VirtAddr(stack_init), 3)?;

    // Store pointers to program args
    emu.mmu_mut()
        .poke_int::<u64>(VirtAddr(stack_init + 8), argv_base as u64)?;
    emu.mmu_mut()
        .poke_int::<u64>(VirtAddr(stack_init + 16), argv_base as u64 + 32)?;
    emu.mmu_mut()
        .poke_int::<u64>(VirtAddr(stack_init + 24), argv_base as u64 + 64)?;

    // Set SP
    emu.set_reg(RegAlias::Sp, stack_init as u64)?;

    Ok(())
}

fn main() {
    let mmu = Mmu::new(VM_MEM_SIZE);
    let mut emu_init = Emulator::new(mmu);

    // Load the program file.
    let brk_addr = emu_init
        .load_program("test-targets/binutils/objdump-riscv")
        .unwrap_or_else(|err| {
            eprintln!("error: could not create emulator: {}", err);
            process::exit(1);
        });

    // Set up the stack.
    setup_stack(&mut emu_init).unwrap_or_else(|err| {
        eprintln!("error: could not set up the stack: {}", err);
        process::exit(1);
    });

    // In JIT mode, create a cache and pass it to the emulator.
    if USE_JIT {
        let jit_cache = JitCache::new(*brk_addr, JIT_CACHE_SIZE);
        emu_init = emu_init.with_jit(jit_cache);
    }

    // `emu_init` and `stats` will be shared among threads. So, they must be
    // wrapped inside `Arc`. In the case of `stats`, it will be modified by the
    // threads, so it also need a `Mutex`.
    let emu_init = Arc::new(emu_init);
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Get the current time to calculate statistics.
    let start = Instant::now();

    // Start one worker per thread.
    let num_threads = if DEBUG_ONE { 1 } else { NUM_THREADS };

    for _ in 0..num_threads {
        let emu_init = Arc::clone(&emu_init);
        let stats = Arc::clone(&stats);

        let fuzzer = Fuzzer::new(emu_init, brk_addr, stats);

        thread::spawn(move || fuzzer.go());
    }

    // Show statistics in the main thread.
    let mut last_fuzz_cases = 0;
    let mut last_total_inst = 0;
    loop {
        thread::sleep(Duration::from_millis(1000));

        let stats = stats.lock().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let last_fcps = stats.fuzz_cases - last_fuzz_cases;
        let fcps = stats.fuzz_cases as f64 / elapsed;
        let last_instps = stats.total_inst - last_total_inst;
        let instps = stats.total_inst as f64 / elapsed;
        let vm_time = stats.vm_cycles as f64 / stats.total_cycles as f64;
        let reset_time = stats.reset_cycles as f64 / stats.total_cycles as f64;
        let syscall_time =
            stats.syscall_cycles as f64 / stats.total_cycles as f64;

        println!(
            "[{:10.4}] cases {:10} | fcps (last) {:10.0} | fcps {:10.1} | \
            Minst/s (last) {:10.0}| Minst/s {:10.1} | crashes {:5} | \
            vm {:6.4} | reset {:6.4} | syscall {:6.4}",
            elapsed,
            stats.fuzz_cases,
            last_fcps,
            fcps,
            last_instps as f64 / 1e6,
            instps / 1e6,
            stats.crashes,
            vm_time,
            reset_time,
            syscall_time,
        );

        last_fuzz_cases = stats.fuzz_cases;
        last_total_inst = stats.total_inst;
    }
}
