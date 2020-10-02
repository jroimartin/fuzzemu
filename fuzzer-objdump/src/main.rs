//! Fuzzer based on a RISC-V emulator.

use std::cmp;
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use elf::defs::{SegmentType, PF_R, PF_W, PF_X};
use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::jit::JitCache;
use riscv_emu::mmu::{
    self, Mmu, Perm, VirtAddr, PERM_EXEC, PERM_READ, PERM_WRITE,
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

/// If `true`, mutate inputs.
const MUTATE: bool = true;

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
    InvalidMemorySegment,

    ProgramExit(u64),
    VmExit(VmExit),

    ElfError(elf::Error),
    IoError(io::Error),
}

impl fmt::Display for FuzzExit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuzzExit::InvalidMemorySegment => {
                write!(f, "invalid memory segment")
            }
            FuzzExit::ProgramExit(code) => write!(f, "program exit: {}", code),
            FuzzExit::VmExit(vmexit) => write!(f, "VM exit: {}", vmexit),
            FuzzExit::ElfError(err) => write!(f, "ELF error: {}", err),
            FuzzExit::IoError(err) => write!(f, "IO error: {}", err),
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

impl From<elf::Error> for FuzzExit {
    fn from(err: elf::Error) -> FuzzExit {
        FuzzExit::ElfError(err)
    }
}

impl From<io::Error> for FuzzExit {
    fn from(err: io::Error) -> FuzzExit {
        FuzzExit::IoError(err)
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

    /// Invalid free, due to double free or corrupted heap.
    Free,
}

impl fmt::Display for FaultType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FaultType::Exec => write!(f, "exec"),
            FaultType::Read => write!(f, "read"),
            FaultType::Write => write!(f, "write"),
            FaultType::Uninit => write!(f, "uninit"),
            FaultType::Bounds => write!(f, "bounds"),
            FaultType::Free => write!(f, "free"),
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

impl From<VirtAddr> for AddressType {
    fn from(addr: VirtAddr) -> AddressType {
        match *addr as isize {
            0..=32767 => AddressType::Null,
            -32768..=-1 => AddressType::Negative,
            _ => AddressType::Normal,
        }
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

impl UniqueCrash {
    fn filename(&self) -> String {
        format!("{}_{}_{}", self.0, self.1, self.2)
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

    /// Total number of timeouts.
    timeouts: u64,
}

/// Returns the current value of the Timestamp Counter.
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// InputFile represents the file opened by objdump.
#[derive(Clone)]
struct InputFile {
    /// Contents of the file.
    contents: Vec<u8>,

    /// Current read/write position in the file.
    cursor: usize,
}

/// Profiling information for one fuzz case.
#[derive(Default)]
struct Profile {
    /// Number of cycles expent in the VM.
    vm_cycles: u64,

    /// Number of cycles expent in syscall handling.
    syscall_cycles: u64,
}

/// A Fuzzer represents a instance of our fuzzer. It links everything together,
/// for instance, memory allocation, file handling, statistics, etc.
struct Fuzzer {
    /// Initial state of the emulator.
    emu_init: Emulator,

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

    /// If `true`, the input file is open.
    input_file_is_open: bool,

    /// Input file. The fuzzer only supports one input file, which is enough
    /// for our use case.
    input_file: InputFile,

    /// Random number generator.
    rng: xorshift::Rng,
}

impl Fuzzer {
    /// Returns a new fuzzer instance.
    fn new(
        emu_init: Emulator,
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
            input_file_is_open: false,
            input_file: InputFile {
                contents: Vec::new(),
                cursor: 0,
            },
            rng: xorshift::Rng::new(0x5273e95b7c721b5a),
        }
    }

    /// Restore the fuzzer to its initial state.
    fn reset(&mut self) {
        self.emu.reset(&self.emu_init);
        self.input_file_is_open = false;
        self.input_file.contents.clear();
        self.input_file.cursor = 0;
    }

    /// Returns a copy of the fuzzer instance.
    fn fork(&self) -> Fuzzer {
        Fuzzer {
            emu_init: self.emu.fork(),
            emu: self.emu.fork(),

            coverage: Arc::clone(&self.coverage),
            corpus: Arc::clone(&self.corpus),
            unique_crashes: Arc::clone(&self.unique_crashes),
            stats: Arc::clone(&self.stats),
            input_file_is_open: self.input_file_is_open,
            input_file: self.input_file.clone(),
            rng: xorshift::Rng::new(0x5273e95b7c721b5a ^ rdtsc()),
        }
    }

    /// Take a snapshot at the specified address.
    fn run_until(&mut self, addr: VirtAddr) -> Result<(), FuzzExit> {
        // Input file size must be bigger than 0 to pass objdump checks.
        self.input_file.contents = vec![0; 16];

        loop {
            let run_result = self.emu.run_emu_until(addr);
            match run_result {
                Err(VmExit::UserBreakpoint) => return Ok(()),
                Err(VmExit::Ecall) => {
                    if let Err(err) = self.syscall_dispatcher() {
                        break Err(err);
                    } else {
                        continue;
                    }
                }
                Err(err) => break Err(err.into()),
                _ => unreachable!(),
            }
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
                self.set_and_mutate_input();
                let mutation_cycles = rdtsc() - mutation_start;

                // Run the target and handle the result.
                let mut profile = Profile::default();
                let fcexit = self.run_fc(&mut profile);

                self.handle_fcexit(fcexit, &mut local_stats);

                if DEBUG_ONE {
                    panic!("DEBUG_ONE is enabled");
                }

                let emu_coverage = self.emu.coverage();

                // Update local stats.
                local_stats.fuzz_cases += 1;
                local_stats.total_inst += emu_coverage.inst_execed;
                local_stats.vm_cycles += profile.vm_cycles;
                local_stats.syscall_cycles += profile.syscall_cycles;
                local_stats.mutation_cycles += mutation_cycles;

                // Update coverage.
                let mut coverage = self.coverage.lock().unwrap();
                let new_coverage = emu_coverage
                    .pcs
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
            stats.timeouts += local_stats.timeouts;

            stats.total_cycles += rdtsc() - batch_start;
        }
    }

    /// Pick an input from the corpus and mutate it.
    fn set_and_mutate_input(&mut self) {
        // Get input from corpus.
        let mut contents = {
            let corpus = self.corpus.lock().unwrap();
            let idx = self.rng.rand() % corpus.len();
            corpus.iter().nth(idx).unwrap().clone()
        };

        // Mutate input.
        if MUTATE && !contents.is_empty() {
            for _ in 0..self.rng.rand() % 128 {
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
            let run_result = self.emu.run();
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
        let pc = self.emu.reg(RegAlias::Pc).unwrap();
        let pc = VirtAddr(pc as usize);

        if DEBUG {
            eprintln!("Fuzz exit: {}", fcexit);
        }

        let unique_crash = match fcexit {
            FuzzExit::ProgramExit(_) => {
                return;
            }
            FuzzExit::VmExit(vmexit) => match vmexit {
                VmExit::Timeout => {
                    stats.timeouts += 1;
                    return;
                }
                VmExit::AddressMisaligned => {
                    UniqueCrash(pc, FaultType::Exec, AddressType::from(pc))
                }
                VmExit::InvalidInstruction => {
                    UniqueCrash(pc, FaultType::Exec, AddressType::from(pc))
                }
                VmExit::MmuError(mmu::Error::ExecFault { addr, .. }) => {
                    UniqueCrash(pc, FaultType::Exec, AddressType::from(addr))
                }
                VmExit::MmuError(mmu::Error::ReadFault { addr, .. }) => {
                    UniqueCrash(pc, FaultType::Read, AddressType::from(addr))
                }
                VmExit::MmuError(mmu::Error::WriteFault { addr, .. }) => {
                    UniqueCrash(pc, FaultType::Write, AddressType::from(addr))
                }
                VmExit::MmuError(mmu::Error::UninitFault { addr, .. }) => {
                    UniqueCrash(pc, FaultType::Uninit, AddressType::from(addr))
                }
                VmExit::MmuError(mmu::Error::InvalidAddress {
                    addr, ..
                }) => {
                    UniqueCrash(pc, FaultType::Bounds, AddressType::from(addr))
                }
                VmExit::MmuError(mmu::Error::InvalidFree { addr }) => {
                    UniqueCrash(pc, FaultType::Free, AddressType::from(addr))
                }
                _ => panic!("Unexpected VmExit error: {}", vmexit),
            },
            _ => panic!("Unexpected FuzzExit error: {}", fcexit),
        };

        stats.crashes += 1;

        let new_crash = {
            let mut unique_crashes = self.unique_crashes.lock().unwrap();
            unique_crashes.insert(unique_crash)
        };

        if new_crash {
            if DEBUG {
                eprintln!("Unique crash: {}", unique_crash);
            }
            let crash_path =
                Path::new(CRASHES_PATH).join(unique_crash.filename());
            fs::write(crash_path, &self.input_file.contents)
                .expect("could not create crash file");

            let mut corpus = self.corpus.lock().unwrap();
            corpus.insert(self.input_file.contents.clone());
        }
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
        // objdump checks.
        let fd = self.emu.reg(RegAlias::A0)?;
        let statbuf = self.emu.reg(RegAlias::A1)?;

        if DEBUG {
            eprintln!("fstat: fd={}", fd);
        }

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
        self.emu.mmu_mut().write_int::<u64>(
            st_size_addr,
            self.input_file.contents.len() as u64,
        )?;

        self.emu.set_reg(RegAlias::A0, 0)?;

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
        // brk should not be called, the allocator funcions are hooked and
        // replaced by our own implementations. The only special case is
        // A0 == 0; in that case, 0 is returned.
        let addr = self.emu.reg(RegAlias::A0)?;
        if addr == 0 {
            return Ok(());
        }
        panic!("brk should not be called: addr={:#010x}", addr);
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
        self.emu.mmu_mut().write_int::<u64>(
            st_size_addr,
            self.input_file.contents.len() as u64,
        )?;

        self.emu.set_reg(RegAlias::A0, 0)?;

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

/// Converts from ELF segment flags to `Mmu` permissions.
fn segment_flags_to_perm(flags: u32) -> Perm {
    let mut perms = 0;

    if flags & PF_R != 0 {
        perms |= PERM_READ;
    }
    if flags & PF_W != 0 {
        perms |= PERM_WRITE;
    }
    if flags & PF_X != 0 {
        perms |= PERM_EXEC;
    }

    Perm(perms)
}

/// Loads an ELF program in the emulator. It also points the program
/// counter to the entrypoint of the program and sets the program break.
fn load_program<P: AsRef<Path>>(
    emu: &mut Emulator,
    program: P,
) -> Result<(), FuzzExit> {
    let contents = fs::read(program)?;
    let elf_headers = elf::parse(&contents)?;

    let mut max_addr = 0;

    for phdr in elf_headers.phdrs() {
        // Get header type and skip non PT_LOAD headers.
        match phdr.segment_type() {
            SegmentType::Load => {
                let phdr_offset = phdr.offset() as usize;
                let phdr_filesz = phdr.filesz() as usize;

                let file_end = phdr_offset
                    .checked_add(phdr_filesz)
                    .ok_or(FuzzExit::InvalidMemorySegment)?;

                let file_bytes = contents
                    .get(phdr_offset..file_end)
                    .ok_or(FuzzExit::InvalidMemorySegment)?;

                let phdr_vaddr = phdr.vaddr() as usize;

                let mem_start = VirtAddr(phdr_vaddr);
                let mem_size = phdr.memsz() as usize;

                emu.mmu_mut().poke(mem_start, file_bytes)?;

                let perms = segment_flags_to_perm(phdr.flags());
                emu.mmu_mut().set_perms(mem_start, mem_size, perms)?;

                // checked_add() is not needed here because integer overflows have
                // been already checked in the previous call to set_perms().
                let mem_end = *mem_start + mem_size;

                max_addr = cmp::max(max_addr, mem_end);
            }
            _ => {
                continue;
            }
        }
    }

    // Place the program counter in the entrypoint.
    let entrypoint = elf_headers.ehdr().entry();
    emu.set_reg(RegAlias::Pc, entrypoint)?;

    // Set the program break to point just after the end of the process's
    // memory. 16-byte aligned.
    let max_addr_aligned = max_addr
        .checked_add(0xf)
        .ok_or(FuzzExit::InvalidMemorySegment)?
        & !0xf;
    emu.mmu_mut().set_brk(VirtAddr(max_addr_aligned));

    Ok(())
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

/// Reads the contents of a directory recursively and returns a set with a
/// deduplicated corpus.
fn populate_corpus<P: AsRef<Path>>(
    path: P,
    corpus: &mut HashSet<Vec<u8>>,
) -> Result<(), io::Error> {
    for entry in fs::read_dir(path)? {
        let path = entry?.path();
        if path.is_dir() {
            populate_corpus(path, corpus)?;
        } else {
            let file_contents = fs::read(path)?;
            corpus.insert(file_contents);
        }
    }
    Ok(())
}

/// _malloc_r hook.
fn malloc_r_cb(emu: &mut Emulator) -> Result<(), VmExit> {
    let size = emu.reg(RegAlias::A1)? as usize;
    if DEBUG {
        println!("malloc: size={:#x}", size);
    }

    if size == 0 {
        emu.set_reg(RegAlias::A0, 0)?;
    } else {
        let addr = emu.mmu_mut().malloc(size, CHECK_RAW)?;
        if DEBUG {
            println!("malloc: ret={}", addr);
        }
        emu.set_reg(RegAlias::A0, *addr as u64)?;
    }

    emu.set_reg(RegAlias::Pc, emu.reg(RegAlias::Ra)?)?;
    Ok(())
}

/// _calloc_r hook.
fn calloc_r_cb(emu: &mut Emulator) -> Result<(), VmExit> {
    let nmemb = emu.reg(RegAlias::A1)? as usize;
    let size = emu.reg(RegAlias::A2)? as usize;
    if DEBUG {
        println!("calloc: nmemb={:#x} size={:#x}", nmemb, size);
    }

    if nmemb == 0 || size == 0 {
        emu.set_reg(RegAlias::A0, 0)?;
    } else if let Some(total_size) = nmemb.checked_mul(size) {
        let addr = emu.mmu_mut().malloc(total_size, CHECK_RAW)?;

        // Set memory to zero.
        let zeros = vec![0u8; total_size];
        emu.mmu_mut().write(addr, &zeros)?;

        if DEBUG {
            println!("calloc: ret={}", addr);
        }
        emu.set_reg(RegAlias::A0, *addr as u64)?;
    } else {
        // If the multiplication of nmemb and size would result in integer
        // overflow, then calloc() returns an error.
        emu.set_reg(RegAlias::A0, 0)?;
    }

    emu.set_reg(RegAlias::Pc, emu.reg(RegAlias::Ra)?)?;
    Ok(())
}

/// _realloc_r hook.
fn realloc_r_cb(emu: &mut Emulator) -> Result<(), VmExit> {
    let ptr = emu.reg(RegAlias::A1)?;
    let ptr = VirtAddr(ptr as usize);
    let size = emu.reg(RegAlias::A2)? as usize;

    if DEBUG {
        println!("realloc: ptr={} size={:#x}", ptr, size);
    }

    if *ptr == 0 {
        // Equivalent to malloc.
        if size == 0 {
            emu.set_reg(RegAlias::A0, 0)?;
        } else {
            let addr = emu.mmu_mut().malloc(size as usize, CHECK_RAW)?;
            if DEBUG {
                println!("realloc: ret={}", addr);
            }
            emu.set_reg(RegAlias::A0, *addr as u64)?;
        }
    } else if size == 0 {
        // Equivalent to free.
        if *ptr != 0 {
            emu.mmu_mut().free(ptr)?;
        }
    } else {
        // Get the size of the realloced memory.
        let old_size = emu
            .mmu()
            .alloc_size(ptr)
            .ok_or(mmu::Error::InvalidFree { addr: ptr })?;

        // Calculate the amount of data to copy.
        let copy_size = cmp::min(old_size, size);

        // Allocate new memory and copy old data.
        let addr = emu.mmu_mut().malloc(size, false)?;
        let mut old_data = vec![0u8; copy_size];
        emu.mmu().peek(ptr, &mut old_data)?;
        emu.mmu_mut().poke(addr, &old_data)?;

        // Copy old permissions.
        let old_perms = emu.mmu().perms(ptr, copy_size)?.to_vec();
        for (offset, perms) in old_perms.iter().enumerate() {
            emu.mmu_mut()
                .set_perms(VirtAddr(*addr + offset), 1, *perms)?;
        }

        // Free old memory.
        emu.mmu_mut().free(ptr)?;

        // Return new address.
        if DEBUG {
            println!("realloc: ret={}", addr);
        }
        emu.set_reg(RegAlias::A0, *addr as u64)?;
    }

    emu.set_reg(RegAlias::Pc, emu.reg(RegAlias::Ra)?)?;
    Ok(())
}

/// _free_r hook.
fn free_r_cb(emu: &mut Emulator) -> Result<(), VmExit> {
    let addr = emu.reg(RegAlias::A1)?;
    if DEBUG {
        println!("free: addr={:#x}", addr);
    }

    if addr != 0 {
        emu.mmu_mut().free(VirtAddr(addr as usize))?;
    }

    emu.set_reg(RegAlias::Pc, emu.reg(RegAlias::Ra)?)?;
    Ok(())
}

fn main() {
    let mmu = Mmu::new(VM_MEM_SIZE);
    let mut emu_init = Emulator::new(mmu);

    // Load the program file.
    load_program(&mut emu_init, "test-targets/binutils/objdump-2.35-riscv")
        .expect("could not load target program");

    // Set up the stack.
    setup_stack(&mut emu_init).expect("could not set up the stack");

    // In JIT mode, create a cache and pass it to the emulator.
    let emu_brk = emu_init.mmu().brk();
    if USE_JIT {
        let jit_cache = JitCache::new(*emu_brk, JIT_CACHE_SIZE);
        emu_init = emu_init.with_jit(jit_cache);
    }

    // Set hooks in memory allocation functions.
    emu_init.hook(VirtAddr(0x10e2d0), malloc_r_cb);
    emu_init.hook(VirtAddr(0x10b3e0), calloc_r_cb);
    emu_init.hook(VirtAddr(0x110a30), realloc_r_cb);
    emu_init.hook(VirtAddr(0x10c7a8), free_r_cb);

    // Populate the initial corpus
    let mut corpus = HashSet::new();
    populate_corpus(INPUTS_PATH, &mut corpus)
        .expect("could not generate intial corpus");

    // The following elements are shared among all the spawned threads, so they
    // are wrapped within an `Arc`. The ones that are mutable are also
    // protected with a `Mutex`.
    let coverage = Arc::new(Mutex::new(HashSet::new()));
    let corpus = Arc::new(Mutex::new(corpus));
    let unique_crashes = Arc::new(Mutex::new(HashSet::new()));
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Create a base fuzzer instance and take a snapshot.
    let mut fuzzer = Fuzzer::new(
        emu_init,
        Arc::clone(&coverage),
        Arc::clone(&corpus),
        Arc::clone(&unique_crashes),
        Arc::clone(&stats),
    );

    //000000000012af04 <_open>:
    // ...
    // 12af1c:	40000893          	li	a7,1024
    // 12af20:	00000073          	ecall
    fuzzer
        .run_until(VirtAddr(0x12af20))
        .expect("could not take snapshot");

    // Get the current time to calculate statistics.
    let start = Instant::now();

    // Start one worker per thread.
    let num_threads = if DEBUG_ONE { 1 } else { NUM_THREADS };

    for _ in 0..num_threads {
        let fuzzer = fuzzer.fork();
        thread::spawn(move || fuzzer.go());
    }

    // Show statistics in the main thread.
    let mut last_fuzz_cases = 0;
    let mut last_total_inst = 0;
    let mut last_stats_time = Instant::now();

    let mut logfile =
        fs::File::create(LOG_FILENAME).expect("could not create log file");

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
        .unwrap();

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
            timeouts {timeouts:5} | \
            fcps (last) {last_fcps:10.0} | fcps {fcps:10.1} | \
            Minst/s (last) {last_instps:10.0} | Minst/s {instps:10.1} | \
            coverage {coverage:10} | corpus {corpus:10} | \
            vm {vm_time:6.4} | reset {reset_time:6.4} | \
            syscall {syscall_time:6.4} | mutation {mutation_time:6.4}",
            elapsed = elapsed,
            fuzz_cases = stats.fuzz_cases,
            unique_crashes = unique_crashes.len(),
            crashes = stats.crashes,
            timeouts = stats.timeouts,
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
