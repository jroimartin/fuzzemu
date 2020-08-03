//! Fuzzer based on a RISC-V emulator.

use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::mmu::{self, Perm, VirtAddr, PERM_RAW, PERM_READ, PERM_WRITE};

/// Number of cores to use.
const NCORES: usize = 1;

/// Print debug information, like stdout output and debug messages.
const DEBUG: bool = true;

/// Memory size of the VM.
const VM_MEM_SIZE: usize = 32 * 1024 * 1024;

/// Amount of memory reserved for the stack at the end of the VM memory. Note
/// that 1024 bytes will be reserved to store the program arguments, so this
/// value must be bigger than 1024.
const STACK_SIZE: usize = 1024 * 1024;

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
}

/// Returns the current value of the Timestamp Counter.
fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

/// A Fuzzer represents a instance of our fuzzer. It links everything together,
/// for instance, memory allocation, file handling, statistics, etc.
struct Fuzzer {
    /// Initial state of the emulator. Shared among all fuzzer instances.
    emu_init: Arc<Emulator>,

    /// Global statistics. Shared among all fuzzer instances.
    stats: Arc<Mutex<Stats>>,

    /// The program break, which defines the end of the process's data segment.
    brk_addr: VirtAddr,
}

impl Fuzzer {
    /// Start a fuzzer worker. We usually spawn one worker per core.
    fn go(mut self) {
        // Fork the initial emulator state for this worker.
        let mut emu = self.emu_init.fork();

        loop {
            let batch_start = rdtsc();

            let mut local_stats = Stats::default();

            // Update global stats every 500M cycles.
            let it = rdtsc();
            while rdtsc() - it < 500_000_000 {
                // Reset memory to the initial state before starting fuzzing.
                let reset_start = rdtsc();
                emu.reset(&self.emu_init);
                local_stats.reset_cycles += rdtsc() - reset_start;

                // Run the target.
                let fuzz_case_result = loop {
                    let mut total_inst = 0;

                    let vm_start = rdtsc();
                    let run_result = emu.run(&mut total_inst);
                    local_stats.vm_cycles += rdtsc() - vm_start;

                    local_stats.total_inst += total_inst;

                    if let Err(VmExit::ECall) = run_result {
                        if let err @ Err(_) = self.handle_syscall(&mut emu) {
                            break err;
                        } else {
                            continue;
                        }
                    } else {
                        break run_result;
                    }
                };

                if let Err(vmexit) = fuzz_case_result {
                    if DEBUG {
                        let pc = emu.get_reg(RegAlias::Pc).unwrap();
                        eprintln!("[{:#010x}] {}", pc, vmexit);
                    }

                    // TODO(rm): Handle crashes as well as unexpected errors.
                    if vmexit.is_crash() {
                        local_stats.crashes += 1;
                    }

                    todo!("stop here for now");
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

            stats.total_cycles += rdtsc() - batch_start;
        }
    }

    /// Handle syscall when the emulator exits with `VmExit::ECall`.
    fn handle_syscall(&mut self, emu: &mut Emulator) -> Result<(), VmExit> {
        let syscall_number = emu.get_reg(RegAlias::A7)?;
        let pc = emu.get_reg(RegAlias::Pc)?;

        match syscall_number {
            // TODO(rm): Implement syscalls.
            57 => {
                // close
                todo!("[{:#010x}] close syscall is not yet implemented", pc);
            }
            64 => {
                // write
                let fd = emu.get_reg(RegAlias::A0)?;
                let buf = emu.get_reg(RegAlias::A1)?;
                let count = emu.get_reg(RegAlias::A2)?;

                if fd != 1 && fd != 2 {
                    todo!(
                        "[{:#010x}] write syscall does not support fd {}",
                        pc,
                        fd
                    );
                }

                if DEBUG {
                    let mut bytes = vec![0; count as usize];
                    emu.mmu.peek(VirtAddr(buf as usize), &mut bytes)?;
                    let buf = String::from_utf8(bytes)
                        .unwrap_or_else(|_| String::from("(invalid string)"));
                    eprint!("{}", buf);
                }

                emu.set_reg(RegAlias::A0, count)?;
            }
            80 => {
                // fstat
                todo!("[{:#010x}] fstat syscall is not yet implemented", pc);
            }
            93 => {
                // exit
                let code = emu.get_reg(RegAlias::A0)?;
                return Err(VmExit::ProgramExit(code));
            }
            214 => {
                // brk
                let addr = emu.get_reg(RegAlias::A0)?;

                if DEBUG {
                    println!(
                        "brk: addr={:#010x} self.brk_addr={}",
                        addr, self.brk_addr
                    );
                }

                if addr == 0 {
                    emu.set_reg(RegAlias::A0, *self.brk_addr as u64)?;
                } else {
                    let increment = addr
                        .checked_sub(*self.brk_addr as u64)
                        .ok_or(VmExit::SyscallError(syscall_number))?;

                    if DEBUG {
                        println!("brk: increment={:#010x}", increment);
                    }

                    // We call sbrk(), and discard the returned value. On
                    // success, brk returns the requested address, which is
                    // already in A0.
                    self.sbrk(emu, increment as usize)?;
                }
            }
            1024 => {
                // open
                todo!("[{:#010x}] open syscall is not yet implemented", pc);
            }
            1038 => {
                // stat
                //
                // XXX: This is not a real implementation, we only returned
                // what is needed to bypass the objdump checks.
                let path_name = emu.get_reg(RegAlias::A0)?;
                let statbuf = emu.get_reg(RegAlias::A1)?;

                let path_name = get_cstr(emu, VirtAddr(path_name as usize))?;
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
                    .ok_or(VmExit::SyscallError(syscall_number))?;
                let st_mode_addr = VirtAddr(st_mode_addr as usize);
                emu.mmu.write_int::<u32>(st_mode_addr, 0x8000)?;

                // Set st_size
                let st_size_addr = statbuf
                    .checked_add(48)
                    .ok_or(VmExit::SyscallError(syscall_number))?;
                let st_size_addr = VirtAddr(st_size_addr as usize);
                emu.mmu.write_int::<u64>(st_size_addr, 0x1337)?;

                emu.set_reg(RegAlias::A0, 0)?;
            }
            _ => todo!("[{:#010x}] unknown syscall", pc),
        }

        emu.set_reg(RegAlias::Pc, pc.wrapping_add(4))?;

        Ok(())
    }

    /// Increments the program's data space by `increment` bytes. Calling this
    /// function with an increment of 0 can be used to find the current
    /// location of the program break.
    fn sbrk(
        &mut self,
        emu: &mut Emulator,
        increment: usize,
    ) -> Result<VirtAddr, VmExit> {
        if increment == 0 {
            return Ok(self.brk_addr);
        }

        // Initialize the new allocated memory as read-after-write, so we can
        // detect accesses to unitialized memory.
        emu.mmu.set_perms(
            self.brk_addr,
            increment,
            Perm(PERM_RAW | PERM_WRITE),
        )?;

        let prev_brk_addr = self.brk_addr;

        // We don't need to use checked_add() here because this has been
        // already checked in the previous call to set_perms().
        self.brk_addr = VirtAddr(*prev_brk_addr + increment);

        Ok(prev_brk_addr)
    }
}

/// Reads a C string (NULL terminated) from memory at `addr`.
fn get_cstr(emu: &mut Emulator, addr: VirtAddr) -> Result<Vec<u8>, VmExit> {
    let mut result = Vec::new();
    let mut curaddr = addr;
    loop {
        let ch = emu.mmu.read_int::<u8>(curaddr)?;
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

/// Set up a stack with a size of `STACK_SIZE` bytes. It also configures
/// the command line argumets passed to the program.
///
/// # Panics
///
/// This function will panic if the memory size of the VM is not higher than
/// `STACK_SIZE`.
fn setup_stack(emu: &mut Emulator) -> Result<(), VmExit> {
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
    emu.mmu
        .poke(VirtAddr(argv_base), b"testdata/binutils/objdump-riscv\x00")?;
    emu.mmu.poke(VirtAddr(argv_base + 32), b"-x\x00")?;
    emu.mmu.poke(
        VirtAddr(argv_base + 64),
        b"testdata/binutils/objdump-riscv\x00",
    )?;

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

    // `emu_init` and `stats` will be shared among threads. So, we have to wrap
    // them inside `Arc`. In the case of `stats`, it will be modified by the
    // threads, so we also need a `Mutex`.
    let emu_init = Arc::new(emu_init);
    let stats = Arc::new(Mutex::new(Stats::default()));

    // Get the current time to calculate statistics.
    let start = Instant::now();

    // Start one worker per thread.
    for _ in 0..NCORES {
        let fuzzer = Fuzzer {
            emu_init: Arc::clone(&emu_init),
            stats: Arc::clone(&stats),
            brk_addr,
        };

        thread::spawn(move || fuzzer.go());
    }

    // Show statistics in the main thread.
    loop {
        thread::sleep(Duration::from_secs(1));

        let stats = stats.lock().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let fcps = stats.fuzz_cases as f64 / elapsed;
        let instps = stats.total_inst as f64 / elapsed;
        let reset_time = stats.reset_cycles as f64 / stats.total_cycles as f64;
        let vm_time = stats.vm_cycles as f64 / stats.total_cycles as f64;

        println!(
            "[{:10.4}] cases {:10} | {:10.1} fcps | {:10.1} inst/s | \
            crashes {:5} | reset {:6.4} | vm {:6.4}",
            elapsed,
            stats.fuzz_cases,
            fcps,
            instps,
            stats.crashes,
            reset_time,
            vm_time
        );
    }
}
