//! Fuzzer based on a RISC-V emulator.

use std::cmp;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, Trace, VmExit};
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
const NUM_THREADS: usize = 8;

/// Memory size of the VM.
const VM_MEM_SIZE: usize = 128 * 1024 * 1024;

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

/// Inputs directory.
const INPUTS_PATH: &str = "test-targets/inputs";

/// Crashes directory.
const CRASHES_PATH: &str = "test-targets/crashes";

/// Log filename.
const LOG_FILENAME: &str = "test-targets/fuzzer-objdump.log";

/// Fuzzer's exit reason.
#[derive(Debug)]
enum FuzzExit {
    ProgramExit(u64),
    VmExit(VmExit),
}

impl fmt::Display for FuzzExit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuzzExit::ProgramExit(code) => write!(f, "program exit: {}", code),
            FuzzExit::VmExit(vmexit) => write!(f, "VM exit: {}", vmexit),
        }
    }
}

impl From<mmu::Error> for FuzzExit {
    fn from(error: mmu::Error) -> FuzzExit {
        FuzzExit::VmExit(VmExit::MmuError(error))
    }
}

impl From<VmExit> for FuzzExit {
    fn from(vmexit: VmExit) -> FuzzExit {
        FuzzExit::VmExit(vmexit)
    }
}

/// A unique crash is defined by the following characteristics:
/// - PC when the crash occurred.
/// - Fault type.
/// - Memory address type.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct UniqueCrash(VirtAddr, FaultType, AddressType);

impl fmt::Display for UniqueCrash {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pc={} fault={} address={}", self.0, self.1, self.2)
    }
}

/// Fault classification.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum FaultType {
    /// Execution fault, due to non-executable memory, misaligned address or
    /// invalid instruction.
    Exec,

    /// Memory read fault.
    Read,

    /// Memory write fault.
    Write,

    /// Uninitialized memory fault.
    Uninit,

    /// Memory address out of the program memory.
    Bounds,
}

impl fmt::Display for FaultType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FaultType::Exec => write!(f, "exec"),
            FaultType::Read => write!(f, "read"),
            FaultType::Write => write!(f, "write"),
            FaultType::Uninit => write!(f, "uninit"),
            FaultType::Bounds => write!(f, "bounds"),
        }
    }
}

/// Address range classification.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum AddressType {
    /// Address within the range [0, 32KiB).
    Null,

    /// Address within the range [-32KiB, 0).
    Negative,

    /// Other addresses.
    Normal,
}

impl fmt::Display for AddressType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AddressType::Null => write!(f, "null"),
            AddressType::Negative => write!(f, "negative"),
            AddressType::Normal => write!(f, "normal"),
        }
    }
}

impl UniqueCrash {
    fn new(vmexit: VmExit, pc: VirtAddr) -> Option<UniqueCrash> {
        match vmexit {
            VmExit::AddressMisaligned => {
                Some(UniqueCrash(pc, FaultType::Exec, AddressType::from(pc)))
            }
            VmExit::InvalidInstruction => {
                Some(UniqueCrash(pc, FaultType::Exec, AddressType::from(pc)))
            }
            VmExit::MmuError(mmu::Error::ExecFault { addr, .. }) => {
                Some(UniqueCrash(pc, FaultType::Exec, AddressType::from(addr)))
            }
            VmExit::MmuError(mmu::Error::ReadFault { addr, .. }) => {
                Some(UniqueCrash(pc, FaultType::Read, AddressType::from(addr)))
            }
            VmExit::MmuError(mmu::Error::WriteFault { addr, .. }) => Some(
                UniqueCrash(pc, FaultType::Write, AddressType::from(addr)),
            ),
            VmExit::MmuError(mmu::Error::UninitFault { addr, .. }) => Some(
                UniqueCrash(pc, FaultType::Uninit, AddressType::from(addr)),
            ),
            VmExit::MmuError(mmu::Error::InvalidAddress { addr, .. }) => Some(
                UniqueCrash(pc, FaultType::Bounds, AddressType::from(addr)),
            ),
            _ => None,
        }
    }

    fn filename(&self) -> String {
        format!("{}_{}_{}", self.0, self.1, self.2)
    }
}

impl From<VirtAddr> for AddressType {
    fn from(addr: VirtAddr) -> AddressType {
        match *addr as isize {
            0..=32767 => AddressType::Null,
            -32768..=-1 => AddressType::Negative,
            _ => AddressType::Normal,
        }
    }
}

/// Xorshift pseudorandom number generator.
struct Rng(u64);

impl Rng {
    /// Returns a new Xorshift PRNG.
    fn new() -> Rng {
        Rng(0x5273e95b7c721b5a ^ rdtsc())
    }

    /// Returns the next random number.
    fn rand(&mut self) -> usize {
        let val = self.0;

        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;

        val as usize
    }
}

/// Statistics recorded during the fuzzing session.
#[derive(Default)]
struct Stats {
    /// Total number of fuzz cases.
    fuzz_cases: u64,

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

    /// Total number of CPU cycles spent mutating the input.
    mutation_cycles: u64,

    /// Total number of crashes.
    crashes: u64,
}

/// Returns the current value of the Timestamp Counter.
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// InputFile represents the file opened by objdump.
struct InputFile {
    /// Contents of the file.
    contents: Vec<u8>,

    /// Current read/write position in the file.
    cursor: usize,
}

/// Profiling information for one fuzz case.
struct Profile {
    /// Emulator trace.
    emu_trace: Trace,

    /// Number of cycles expent in the VM.
    vm_cycles: u64,

    /// Number of cycles expent in syscall handling.
    syscall_cycles: u64,
}

/// A Fuzzer represents a instance of our fuzzer. It links everything together,
/// for instance, memory allocation, file handling, statistics, etc.
struct Fuzzer {
    /// Initial state of the emulator.
    emu_init: Arc<Emulator>,

    /// Current state of the emulator.
    emu: Emulator,

    /// Global statistics.
    stats: Arc<Mutex<Stats>>,

    /// Global coverage.
    coverage: Arc<Mutex<HashSet<VirtAddr>>>,

    /// Global corpus.
    corpus: Arc<Mutex<HashSet<Vec<u8>>>>,

    /// Global set of unique crashes.
    unique_crashes: Arc<Mutex<HashSet<UniqueCrash>>>,

    /// The initial program break.
    brk_addr_init: VirtAddr,

    /// The current program break.
    brk_addr: VirtAddr,

    /// If `true`, the input file is open.
    input_file_is_open: bool,

    /// Input file. The fuzzer only supports one open file, which is enough for
    /// our use case.
    input_file: InputFile,

    /// Random number generator.
    rng: Rng,
}

impl Fuzzer {
    /// Returns a new fuzzer instance.
    fn new(
        emu_init: Arc<Emulator>,
        brk_addr_init: VirtAddr,
        coverage: Arc<Mutex<HashSet<VirtAddr>>>,
        corpus: Arc<Mutex<HashSet<Vec<u8>>>>,
        unique_crashes: Arc<Mutex<HashSet<UniqueCrash>>>,
        stats: Arc<Mutex<Stats>>,
    ) -> Fuzzer {
        let emu = emu_init.fork();

        Fuzzer {
            emu_init,
            emu,
            coverage,
            corpus,
            unique_crashes,
            stats,
            brk_addr_init,
            brk_addr: brk_addr_init,
            input_file_is_open: false,
            input_file: InputFile {
                contents: Vec::new(),
                cursor: 0,
            },
            rng: Rng::new(),
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

                let mutation_start = rdtsc();
                self.mutate_input();
                let mutation_cycles = rdtsc() - mutation_start;

                // Run the target and handle the result.
                let mut profile = Profile {
                    emu_trace: Trace {
                        inst_execed: 0,
                        coverage: HashSet::new(),
                    },
                    vm_cycles: 0,
                    syscall_cycles: 0,
                };
                let fcexit = self.run_fc(&mut profile);

                self.handle_fcexit(fcexit, &mut local_stats);

                if DEBUG_ONE {
                    panic!("DEBUG_ONE is enabled");
                }

                // Update local stats.
                local_stats.fuzz_cases += 1;
                local_stats.total_inst += profile.emu_trace.inst_execed;
                local_stats.vm_cycles += profile.vm_cycles;
                local_stats.syscall_cycles += profile.syscall_cycles;
                local_stats.mutation_cycles += mutation_cycles;

                // Update coverage.
                let mut coverage = self.coverage.lock().unwrap();
                let new_coverage = profile
                    .emu_trace
                    .coverage
                    .iter()
                    .fold(false, |acc, addr| acc | coverage.insert(*addr));

                // If the coverage is bigger, add the fuzz case to the corpus.
                if new_coverage {
                    let mut corpus = self.corpus.lock().unwrap();
                    corpus.insert(self.input_file.contents.clone());
                }
            }

            // Update global stats.
            let mut stats = self.stats.lock().unwrap();

            stats.fuzz_cases += local_stats.fuzz_cases;
            stats.total_inst += local_stats.total_inst;
            stats.reset_cycles += local_stats.reset_cycles;
            stats.vm_cycles += local_stats.vm_cycles;
            stats.syscall_cycles += local_stats.syscall_cycles;
            stats.mutation_cycles += local_stats.mutation_cycles;
            stats.crashes += local_stats.crashes;

            stats.total_cycles += rdtsc() - batch_start;
        }
    }

    /// Pick an input from the corpus and mutate it.
    fn mutate_input(&mut self) {
        // Get input from corpus.
        let mut contents = {
            let corpus = self.corpus.lock().unwrap();
            let idx = self.rng.rand() % corpus.len();
            corpus.iter().nth(idx).unwrap().clone()
        };

        // Mutate input.
        if !contents.is_empty() {
            for _ in 0..self.rng.rand() % 512 {
                let sel = self.rng.rand() % contents.len();
                contents[sel] = self.rng.rand() as u8;
            }
        }

        // Set input file contents.
        self.input_file.contents = contents;
    }

    /// Runs a single fuzz_case.
    fn run_fc(&mut self, profile: &mut Profile) -> FuzzExit {
        loop {
            let vm_start = rdtsc();
            let run_result = self.emu.run(&mut profile.emu_trace);
            profile.vm_cycles += rdtsc() - vm_start;

            match run_result {
                Err(VmExit::Ecall) => {
                    let syscall_start = rdtsc();
                    let syscall_result = self.syscall_dispatcher();
                    profile.syscall_cycles += rdtsc() - syscall_start;

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
        // Calling unwrap here is safe , we now that `.reg()` cannot fail
        // because `RegAlias::Pc` is a valid register.
        let pc = self.emu.reg(RegAlias::Pc).unwrap();

        let unique_crash = match fcexit {
            FuzzExit::ProgramExit(code) => {
                if DEBUG {
                    eprintln!("program exited with {}", code);
                }
                None
            }
            FuzzExit::VmExit(vmexit) => {
                UniqueCrash::new(vmexit, VirtAddr(pc as usize))
            }
        };

        if let Some(unique_crash) = unique_crash {
            stats.crashes += 1;

            let new_crash = {
                let mut unique_crashes = self.unique_crashes.lock().unwrap();
                unique_crashes.insert(unique_crash)
            };

            if new_crash {
                if DEBUG {
                    eprintln!("unique_crash={}", unique_crash);
                }
                let crash_path =
                    Path::new(CRASHES_PATH).join(unique_crash.filename());
                fs::write(crash_path, &self.input_file.contents)
                    .unwrap_or_else(|err| {
                        eprintln!(
                            "error: could not create crash file: {}",
                            err
                        );
                        process::exit(1);
                    });

                let mut corpus = self.corpus.lock().unwrap();
                corpus.insert(self.input_file.contents.clone());
            }
        }
    }

    /// Restore the fuzzer to its initial state.
    fn reset(&mut self) {
        self.emu.reset(&self.emu_init);
        self.brk_addr = self.brk_addr_init;
        self.input_file_is_open = false;
        self.input_file.contents.clear();
        self.input_file.cursor = 0;
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
            self.input_file_is_open = false;
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

        // If the input_file is not open, return error.
        if !self.input_file_is_open {
            self.emu.set_reg(RegAlias::Pc, !0)?;
            return Ok(());
        }

        // Emulate lseek.
        let contents_len = self.input_file.contents.len();

        let new_cursor = match whence {
            SEEK_SET => offset as usize,
            SEEK_CUR => self.input_file.cursor + offset as usize,
            SEEK_END => (contents_len as i64 + offset as i64) as usize,
            _ => {
                // Return error if the whence value is invalid.
                self.emu.set_reg(RegAlias::A0, !0)?;
                return Ok(());
            }
        };

        // Although it should be possible, we do not support seeking beyond
        // the end of the file, so we return with error in that case.
        if new_cursor >= contents_len {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        self.input_file.cursor = new_cursor;
        self.emu
            .set_reg(RegAlias::A0, self.input_file.cursor as u64)?;

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

        // If the input file is not open, return error.
        if !self.input_file_is_open {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        // Emulate read.
        let cursor = self.input_file.cursor;
        let remaining = self.input_file.contents.len() - cursor;

        // count cannot be bigger than the number of remaining bytes.
        let count = cmp::min(count, remaining);

        // checked_add is not needed here because the cursor is already
        // sanitized by the lseek handler.
        self.emu
            .mmu_mut()
            .write(buf, &self.input_file.contents[cursor..cursor + count])?;

        self.input_file.cursor += count as usize;
        self.emu.set_reg(RegAlias::A0, count as u64)?;

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

        let mut bytes = vec![0; count as usize];
        self.emu.mmu().read(VirtAddr(buf as usize), &mut bytes)?;
        let buf = String::from_utf8(bytes)
            .unwrap_or_else(|_| String::from("(invalid string)"));

        if DEBUG_OUTPUT {
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
        if path_name != "input_file" || self.input_file_is_open {
            self.emu.set_reg(RegAlias::A0, !0)?;
            return Ok(());
        }

        self.input_file_is_open = true;
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
        let st_mode_addr = statbuf.checked_add(16).ok_or(
            mmu::Error::AddressIntegerOverflow {
                addr: VirtAddr(statbuf as usize),
                size: 16,
            },
        )?;
        let st_mode_addr = VirtAddr(st_mode_addr as usize);
        self.emu.mmu_mut().write_int::<u32>(st_mode_addr, 0x8000)?;

        // Set st_size
        let st_size_addr = statbuf.checked_add(48).ok_or(
            mmu::Error::AddressIntegerOverflow {
                addr: VirtAddr(statbuf as usize),
                size: 48,
            },
        )?;
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

        let new_brk_addr = self.brk_addr.checked_sub(size).ok_or(
            mmu::Error::AddressIntegerOverflow {
                addr: self.brk_addr,
                size,
            },
        )?;
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
    emu.mmu_mut().poke(VirtAddr(argv_base + 32), b"-g\x00")?;
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

/// Read the contents of a directory and return a set with a deduplicated
/// corpus.
fn populate_corpus<P: AsRef<Path>>(
    path: P,
) -> Result<HashSet<Vec<u8>>, io::Error> {
    let mut corpus = HashSet::new();

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_contents = fs::read(entry.path())?;
        corpus.insert(file_contents);
    }

    Ok(corpus)
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

    // Populate the initial corpus
    let corpus = populate_corpus(INPUTS_PATH).unwrap_or_else(|err| {
        eprintln!("error: could not generate intial corpus: {}", err);
        process::exit(1);
    });

    // The following elements are shared among all the spawned threads, so they
    // are wrapped within an `Arc`. The ones that are mutable are also
    // protected with a `Mutex`.
    let emu_init = Arc::new(emu_init);
    let coverage = Arc::new(Mutex::new(HashSet::new()));
    let corpus = Arc::new(Mutex::new(corpus));
    let unique_crashes = Arc::new(Mutex::new(HashSet::new()));
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Get the current time to calculate statistics.
    let start = Instant::now();

    // Start one worker per thread.
    let num_threads = if DEBUG_ONE { 1 } else { NUM_THREADS };

    for _ in 0..num_threads {
        let emu_init = Arc::clone(&emu_init);
        let coverage = Arc::clone(&coverage);
        let corpus = Arc::clone(&corpus);
        let unique_crashes = Arc::clone(&unique_crashes);
        let stats = Arc::clone(&stats);

        let fuzzer = Fuzzer::new(
            emu_init,
            brk_addr,
            coverage,
            corpus,
            unique_crashes,
            stats,
        );

        thread::spawn(move || fuzzer.go());
    }

    // Show statistics in the main thread.
    let mut last_fuzz_cases = 0;
    let mut last_total_inst = 0;
    let mut last_stats_time = Instant::now();

    let mut logfile = fs::File::create(LOG_FILENAME).unwrap_or_else(|err| {
        eprintln!("error: could not create log file: {}", err);
        process::exit(1);
    });

    loop {
        thread::sleep(Duration::from_millis(10));

        let elapsed = start.elapsed().as_secs_f64();
        let stats = stats.lock().unwrap();
        let unique_crashes = unique_crashes.lock().unwrap();
        let coverage = coverage.lock().unwrap();

        writeln!(
            logfile,
            "{:.4} {} {} {}",
            elapsed,
            stats.fuzz_cases,
            unique_crashes.len(),
            coverage.len()
        )
        .unwrap_or_else(|err| {
            eprintln!("error: could not write to log file: {}", err);
            process::exit(1);
        });

        let now = Instant::now();
        if now.duration_since(last_stats_time) < Duration::from_millis(1000) {
            continue;
        }

        let corpus = corpus.lock().unwrap();

        let last_fcps = stats.fuzz_cases - last_fuzz_cases;
        let fcps = stats.fuzz_cases as f64 / elapsed;
        let last_instps = stats.total_inst - last_total_inst;
        let instps = stats.total_inst as f64 / elapsed;
        let vm_time = stats.vm_cycles as f64 / stats.total_cycles as f64;
        let reset_time = stats.reset_cycles as f64 / stats.total_cycles as f64;
        let syscall_time =
            stats.syscall_cycles as f64 / stats.total_cycles as f64;
        let mutation_time =
            stats.mutation_cycles as f64 / stats.total_cycles as f64;

        println!(
            "[{elapsed:10.4}] cases {fuzz_cases:10} | \
            unique crashes {unique_crashes:5} | crashes {crashes:5} | \
            fcps (last) {last_fcps:10.0} | fcps {fcps:10.1} | \
            Minst/s (last) {last_instps:10.0}| Minst/s {instps:10.1} | \
            coverage {coverage:10} | corpus {corpus:10} | \
            vm {vm_time:6.4} | reset {reset_time:6.4} | \
            syscall {syscall_time:6.4} | mutation {mutation_time:6.4}",
            elapsed = elapsed,
            fuzz_cases = stats.fuzz_cases,
            unique_crashes = unique_crashes.len(),
            crashes = stats.crashes,
            last_fcps = last_fcps,
            fcps = fcps,
            last_instps = last_instps as f64 / 1e6,
            instps = instps / 1e6,
            coverage = coverage.len(),
            corpus = corpus.len(),
            vm_time = vm_time,
            reset_time = reset_time,
            syscall_time = syscall_time,
            mutation_time = mutation_time
        );

        last_fuzz_cases = stats.fuzz_cases;
        last_total_inst = stats.total_inst;
        last_stats_time = now;
    }
}
