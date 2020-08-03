//! Fuzzer based on a RISC-V emulator.

use std::cmp;
use std::fmt;
use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::mmu::{self, Perm, VirtAddr, PERM_RAW, PERM_READ, PERM_WRITE};

/// Number of cores to use.
const NCORES: usize = 1;

/// Print debug messages.
const DEBUG: bool = true;

/// Print stdout/stderr output.
const DEBUG_OUTPUT: bool = false;

/// Memory size of the VM.
const VM_MEM_SIZE: usize = 32 * 1024 * 1024;

/// Amount of memory reserved for the stack at the end of the VM memory. Note
/// that 1024 bytes will be reserved to store the program arguments, so this
/// value must be bigger than 1024.
const STACK_SIZE: usize = 1024 * 1024;

/// Fuzzer's exit reason.
enum FuzzExit {
    Error(String),
    ProgramExit(u64),
    EcallError(u64),

    MmuError(mmu::Error),
    VmExit(VmExit),
}

impl fmt::Display for FuzzExit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FuzzExit::ProgramExit(code) => write!(f, "program exit: {}", code),
            FuzzExit::Error(err) => write!(f, "fuzzer error: {}", err),
            FuzzExit::EcallError(num) => {
                write!(f, "error while executing ecall {}", num)
            }
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
    fn go(&mut self) {
        loop {
            let batch_start = rdtsc();

            let mut local_stats = Stats::default();

            // Update global stats every 500M cycles.
            let it = rdtsc();
            while rdtsc() - it < 500_000_000 {
                // Reset memory to the initial state before starting fuzzing.
                let reset_start = rdtsc();
                self.reset();
                local_stats.reset_cycles += rdtsc() - reset_start;

                // Run the target.
                let fuzz_case_result = loop {
                    let mut total_inst = 0;

                    let vm_start = rdtsc();
                    let run_result = self.emu.run(&mut total_inst);
                    local_stats.vm_cycles += rdtsc() - vm_start;

                    local_stats.total_inst += total_inst;

                    match run_result {
                        Err(VmExit::Ecall) => {
                            let syscall_start = rdtsc();
                            let syscall_result = self.syscall_dispatcher();
                            local_stats.syscall_cycles +=
                                rdtsc() - syscall_start;

                            if let Err(err) = syscall_result {
                                break err;
                            } else {
                                continue;
                            }
                        }
                        Err(err) => break err.into(),
                        _ => unreachable!(),
                    }
                };

                // TODO(rm): Handle crashes as well as unexpected errors.
                match fuzz_case_result {
                    FuzzExit::VmExit(vmexit) => {
                        if DEBUG {
                            let pc = self.emu.get_reg(RegAlias::Pc).unwrap();
                            eprintln!(
                                "[{:#010x}] {}\n{}",
                                pc, vmexit, self.emu
                            );
                        }

                        if vmexit.is_crash() {
                            local_stats.crashes += 1;
                        }
                    }
                    err => todo!("unhandled error: {}", err),
                }

                todo!("stop here for now");

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

    /// Restore the fuzzer to its initial state.
    fn reset(&mut self) {
        self.emu.reset(&self.emu_init);
        self.brk_addr = self.brk_addr_init;
        self.input_file = None;
    }

    /// Dispatches syscalls when the emulator exits with `VmExit::Ecall`,
    /// redirecting the execution to the relevant syscall handler.
    fn syscall_dispatcher(&mut self) -> Result<(), FuzzExit> {
        let syscall_number = self.emu.get_reg(RegAlias::A7)?;
        let pc = self.emu.get_reg(RegAlias::Pc)?;

        let syscall_result = match syscall_number {
            57 => self.syscall_close(),
            62 => self.syscall_lseek(),
            63 => self.syscall_read(),
            64 => self.syscall_write(),
            80 => self.syscall_fstat(),
            93 => self.syscall_exit(),
            214 => self.syscall_brk(),
            1024 => self.syscall_open(),
            1038 => self.syscall_stat(),
            _ => todo!("[{:#010x}] unknown syscall", pc),
        };

        if let Err(err) = syscall_result {
            if DEBUG {
                eprintln!("Syscall {} error: {}", syscall_number, err);
            }
            return Err(FuzzExit::EcallError(syscall_number));
        }

        self.emu.set_reg(RegAlias::Pc, pc.wrapping_add(4))?;

        Ok(())
    }

    /// close's syscall handle.
    fn syscall_close(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.get_reg(RegAlias::A0)?;

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

    /// lseek's syscall handle.
    fn syscall_lseek(&mut self) -> Result<(), FuzzExit> {
        const SEEK_SET: u64 = 0;
        const SEEK_CUR: u64 = 1;
        const SEEK_END: u64 = 2;

        let fd = self.emu.get_reg(RegAlias::A0)?;
        let offset = self.emu.get_reg(RegAlias::A1)?;
        let whence = self.emu.get_reg(RegAlias::A2)?;

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
                _ => {
                    return Err(FuzzExit::Error(String::from(
                        "invalid whence",
                    )))
                }
            };

            // Although it should be possible, we do not support seeking beyond
            // the end of the file, so we return with error in that case.
            if new_cursor >= contents_len {
                self.emu.set_reg(RegAlias::A0, !0)?;
                return Ok(())
            }

            input_file.cursor = new_cursor;
            self.emu.set_reg(RegAlias::A0, input_file.cursor as u64)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// read's syscall handle.
    fn syscall_read(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.get_reg(RegAlias::A0)?;
        let buf = self.emu.get_reg(RegAlias::A1)?;
        let count = self.emu.get_reg(RegAlias::A2)? as usize;

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

            // checked_add is not needed here because is cursor is already
            // sanitized by the lseek handler.
            self.emu
                .mmu
                .write(buf, &input_file.contents[cursor..cursor + count])?;

            input_file.cursor += count as usize;
            self.emu.set_reg(RegAlias::A0, count as u64)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// write's syscall handle.
    fn syscall_write(&mut self) -> Result<(), FuzzExit> {
        let fd = self.emu.get_reg(RegAlias::A0)?;
        let buf = self.emu.get_reg(RegAlias::A1)?;
        let count = self.emu.get_reg(RegAlias::A2)?;

        if DEBUG {
            eprintln!("write: fd={} count={}", fd, count);
        }

        // We only support writing to stdout and stderr.
        if fd == 1 || fd == 2 {
            if DEBUG_OUTPUT {
                let mut bytes = vec![0; count as usize];
                self.emu.mmu.read(VirtAddr(buf as usize), &mut bytes)?;
                let buf = String::from_utf8(bytes)
                    .unwrap_or_else(|_| String::from("(invalid string)"));
                eprint!("{}", buf);
            }

            self.emu.set_reg(RegAlias::A0, count)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// fstat's syscall handle.
    fn syscall_fstat(&mut self) -> Result<(), FuzzExit> {
        // This is not a real implementation, but it's enough to bypass the
        // objdump checks. We just return with error.
        let fd = self.emu.get_reg(RegAlias::A0)?;

        if DEBUG {
            eprintln!("fstat: fd={}", fd);
        }

        self.emu.set_reg(RegAlias::A0, !0)?;

        Ok(())
    }

    /// exit's syscall handle.
    fn syscall_exit(&mut self) -> Result<(), FuzzExit> {
        let code = self.emu.get_reg(RegAlias::A0)?;

        if DEBUG {
            eprintln!("exit: {}", code);
        }

        Err(FuzzExit::ProgramExit(code))
    }

    /// brk's syscall handle.
    fn syscall_brk(&mut self) -> Result<(), FuzzExit> {
        let addr = self.emu.get_reg(RegAlias::A0)?;

        if DEBUG {
            println!(
                "brk: addr={:#010x} self.brk_addr={}",
                addr, self.brk_addr
            );
        }

        if addr == 0 {
            self.emu.set_reg(RegAlias::A0, *self.brk_addr as u64)?;
        } else {
            let increment = addr
                .checked_sub(*self.brk_addr as u64)
                .ok_or(FuzzExit::Error(String::from("invalid increment")))?;

            if DEBUG {
                println!("brk: increment={:#010x}", increment);
            }

            // After calling sbrk(), the returned value is discarded.
            // On success, brk returns the requested address, which is
            // already in A0.
            self.sbrk(increment as usize)?;
        }

        Ok(())
    }

    /// open's syscall handle.
    fn syscall_open(&mut self) -> Result<(), FuzzExit> {
        // We only care about the path name, so we don't even consider the
        // other arguments.
        let path_name = self.emu.get_reg(RegAlias::A0)?;

        let path_name = self.get_cstr(VirtAddr(path_name as usize))?;
        let path_name = String::from_utf8_lossy(&path_name);

        if DEBUG {
            eprintln!("open: pathname={}", path_name);
        }

        // We only allow to open the input file, which fd is always 1337. If
        // the path does not correspond to the expected input file, then return
        // error.
        if path_name == "input_file" && self.input_file.is_none() {
            let input_file = InputFile {
                cursor: 0,
                contents: include_bytes!(
                    "../../../testdata/binutils/objdump-riscv"
                ),
            };

            self.input_file = Some(input_file);
            self.emu.set_reg(RegAlias::A0, 1337)?;
        } else {
            self.emu.set_reg(RegAlias::A0, !0)?;
        }

        Ok(())
    }

    /// stat's syscall handle.
    fn syscall_stat(&mut self) -> Result<(), FuzzExit> {
        // This is not a real implementation, but it's enough to bypass the
        // objdump checks.
        let path_name = self.emu.get_reg(RegAlias::A0)?;
        let statbuf = self.emu.get_reg(RegAlias::A1)?;

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
        let st_mode_addr = statbuf
            .checked_add(16)
            .ok_or(FuzzExit::Error(String::from("invalid statbuf address")))?;
        let st_mode_addr = VirtAddr(st_mode_addr as usize);
        self.emu.mmu.write_int::<u32>(st_mode_addr, 0x8000)?;

        // Set st_size
        let st_size_addr = statbuf
            .checked_add(48)
            .ok_or(FuzzExit::Error(String::from("invalid statbuf address")))?;
        let st_size_addr = VirtAddr(st_size_addr as usize);
        self.emu.mmu.write_int::<u64>(st_size_addr, 0x1337)?;

        self.emu.set_reg(RegAlias::A0, 0)?;

        Ok(())
    }

    /// Increments the program's data space by `increment` bytes. Calling this
    /// function with an increment of 0 can be used to find the current
    /// location of the program break. The program break defines the end of the
    /// process's data segment
    fn sbrk(&mut self, increment: usize) -> Result<VirtAddr, FuzzExit> {
        if increment == 0 {
            return Ok(self.brk_addr);
        }

        // Initialize the new allocated memory as read-after-write, so it's
        // possible to detect accesses to unitialized memory.
        self.emu.mmu.set_perms(
            self.brk_addr,
            increment,
            Perm(PERM_RAW | PERM_WRITE),
        )?;

        let prev_brk_addr = self.brk_addr;

        // checked_add() is not needed here because this has been already
        // checked in the previous call to set_perms().
        self.brk_addr = VirtAddr(*prev_brk_addr + increment);

        Ok(prev_brk_addr)
    }

    /// Reads a C string (NULL terminated) from memory at `addr`.
    fn get_cstr(&self, addr: VirtAddr) -> Result<Vec<u8>, FuzzExit> {
        let mut result = Vec::new();
        let mut curaddr = addr;
        loop {
            let ch = self.emu.mmu.read_int::<u8>(curaddr)?;
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
    let mem_size = emu.mmu.size();

    assert!(mem_size > STACK_SIZE, "the VM memory is not big enough");

    let argv_base: usize = mem_size - 512;
    let stack_init: usize = mem_size - 1024;

    // Set the permissions of the stack to RW.
    emu.mmu.set_perms(
        VirtAddr(mem_size - STACK_SIZE),
        STACK_SIZE,
        Perm(PERM_READ | PERM_WRITE),
    )?;

    // Store program args
    emu.mmu.poke(VirtAddr(argv_base), b"objdump\x00")?;
    emu.mmu.poke(VirtAddr(argv_base + 32), b"-x\x00")?;
    emu.mmu.poke(VirtAddr(argv_base + 64), b"input_file\x00")?;

    // Store argc
    emu.mmu.poke_int::<u64>(VirtAddr(stack_init), 3)?;

    // Store pointers to program args
    emu.mmu
        .poke_int::<u64>(VirtAddr(stack_init + 8), argv_base as u64)?;
    emu.mmu
        .poke_int::<u64>(VirtAddr(stack_init + 16), argv_base as u64 + 32)?;
    emu.mmu
        .poke_int::<u64>(VirtAddr(stack_init + 24), argv_base as u64 + 64)?;

    // Set SP
    emu.set_reg(RegAlias::Sp, stack_init as u64)?;

    Ok(())
}

fn main() {
    let mut emu_init = Emulator::new(VM_MEM_SIZE);

    // Load the program file.
    let brk_addr = emu_init
        .load_program("testdata/binutils/objdump-riscv")
        .unwrap_or_else(|err| {
            eprintln!("error: could not create emulator: {}", err);
            process::exit(1);
        });

    // Set up the stack.
    setup_stack(&mut emu_init).unwrap_or_else(|err| {
        eprintln!("error: could not set up the stack: {}", err);
        process::exit(1);
    });

    // `emu_init` and `stats` will be shared among threads. So, they must be
    // wrapped inside `Arc`. In the case of `stats`, it will be modified by the
    // threads, so it also need a `Mutex`.
    let emu_init = Arc::new(emu_init);
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Get the current time to calculate statistics.
    let start = Instant::now();

    // Start one worker per thread.
    for _ in 0..NCORES {
        let mut fuzzer =
            Fuzzer::new(Arc::clone(&emu_init), brk_addr, Arc::clone(&stats));
        thread::spawn(move || fuzzer.go());
    }

    // Show statistics in the main thread.
    loop {
        thread::sleep(Duration::from_secs(1));

        let stats = stats.lock().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let fcps = stats.fuzz_cases as f64 / elapsed;
        let instps = stats.total_inst as f64 / elapsed;
        let vm_time = stats.vm_cycles as f64 / stats.total_cycles as f64;
        let reset_time = stats.reset_cycles as f64 / stats.total_cycles as f64;
        let syscall_time =
            stats.syscall_cycles as f64 / stats.total_cycles as f64;

        println!(
            "[{:10.4}] cases {:10} | {:10.1} fcps | {:10.1} inst/s | \
            crashes {:5} | vm {:6.4} | reset {:6.4} | syscall {:6.4}",
            elapsed,
            stats.fuzz_cases,
            fcps,
            instps,
            stats.crashes,
            vm_time,
            reset_time,
            syscall_time,
        );
    }
}
