//! Fuzzer based on a RISC-V emulator.

use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::mmu::{Perm, VirtAddr, PERM_READ, PERM_WRITE};

/// Number of cores to use.
const NCORES: usize = 1;

/// Print debug information, like stdout output and debug messages.
const DEBUG: bool = true;

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

/// Handle syscall when the emulator exits with `VmExit::ECall`.
fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    let syscall_number = emu.get_reg(RegAlias::A7)?;
    let pc = emu.get_reg(RegAlias::Pc)?;

    match syscall_number {
        // TODO(rm): Implement syscalls.
        57 => {
            // close
            emu.set_reg(RegAlias::A0, !0)?;
        }
        64 => {
            // write
            let fd = emu.get_reg(RegAlias::A0)?;
            let buf_ptr = emu.get_reg(RegAlias::A1)?;
            let count = emu.get_reg(RegAlias::A2)?;

            if DEBUG {
                let mut bytes = vec![0; count as usize];
                emu.mmu.peek(VirtAddr(buf_ptr as usize), &mut bytes)?;
                let buf = String::from_utf8(bytes)
                    .unwrap_or_else(|_| String::from("(invalid string)"));
                println!("fd={} count={}\n{}", fd, count, buf);
            }

            emu.set_reg(RegAlias::A0, count)?;
        }
        80 => {
            // fstat
            emu.set_reg(RegAlias::A0, !0)?;
        }
        93 => {
            // exit
            let code = emu.get_reg(RegAlias::A0)?;
            return Err(VmExit::ProgramExit(code));
        }
        214 => {
            // brk
            emu.set_reg(RegAlias::A0, !0)?;
        }
        _ => todo!("syscall"),
    }

    emu.set_reg(RegAlias::Pc, pc.wrapping_add(4))?;

    Ok(())
}

/// This function implements a fuzzer worker. It takes care of executing fuzz
/// cases. We spawn one worker per core.
fn worker(emu_init: Arc<Emulator>, stats: Arc<Mutex<Stats>>) {
    // Fork the initial emulator state for this worker.
    let mut emu = emu_init.fork();

    loop {
        let batch_start = rdtsc();

        let mut local_stats = Stats::default();

        // Update global stats every 500M cycles.
        let it = rdtsc();
        while rdtsc() - it < 500_000_000 {
            // Reset memory to the initial state before starting fuzzing.
            let reset_start = rdtsc();
            emu.reset(&emu_init);
            local_stats.reset_cycles += rdtsc() - reset_start;

            // Run the target.
            let fuzz_case_result = loop {
                let mut total_inst = 0;

                let vm_start = rdtsc();
                let run_result = emu.run(&mut total_inst);
                local_stats.vm_cycles += rdtsc() - vm_start;

                local_stats.total_inst += total_inst;

                if let Err(VmExit::ECall) = run_result {
                    if let err @ Err(_) = handle_syscall(&mut emu) {
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
                    println!("{}", vmexit);
                }

                // TODO(rm): Handle crashes as well as unexpected errors.
                if vmexit.is_crash() {
                    local_stats.crashes += 1;
                }
            }

            // Update local stats.
            local_stats.fuzz_cases += 1;
        }

        // Update global stats.
        let mut stats = stats.lock().unwrap();

        stats.fuzz_cases += local_stats.fuzz_cases;
        stats.crashes += local_stats.crashes;
        stats.total_inst += local_stats.total_inst;
        stats.reset_cycles += local_stats.reset_cycles;
        stats.vm_cycles += local_stats.vm_cycles;

        stats.total_cycles += rdtsc() - batch_start;
    }
}

/// Set up a stack with a size of `STACK_SIZE` bytes.
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

    // Create the stack and set the stack pointer.
    emu.mmu.set_perms(
        VirtAddr(mem_size - STACK_SIZE),
        STACK_SIZE,
        Perm(PERM_READ | PERM_WRITE),
    )?;

    // Store program args
    emu.mmu.poke(VirtAddr(argv_base), b"argv0").unwrap();
    emu.mmu.poke(VirtAddr(argv_base + 32), b"argv1")?;

    // Store argc
    emu.mmu.poke_int::<u64>(VirtAddr(stack_init), 2)?;

    // Store pointers to program args
    emu.mmu
        .poke_int::<u64>(VirtAddr(stack_init + 8), argv_base as u64)?;
    emu.mmu
        .poke_int::<u64>(VirtAddr(stack_init + 16), argv_base as u64 + 32)?;

    // Set SP
    emu.set_reg(RegAlias::Sp, stack_init as u64)?;

    Ok(())
}

fn main() {
    const VM_MEM_SIZE: usize = 32 * 1024 * 1024;

    let mut emu_init = Emulator::new(VM_MEM_SIZE);

    // Load the program file.
    emu_init
        .load_program("testdata/hello")
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
        let emu_init = Arc::clone(&emu_init);
        let stats = Arc::clone(&stats);

        thread::spawn(move || worker(emu_init, stats));
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
