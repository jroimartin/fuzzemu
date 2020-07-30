use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::{Emulator, RegAlias, VmExit};
use riscv_emu::mmu::{Perm, VirtAddr, PERM_READ, PERM_WRITE};

const NCORES: usize = 1;

#[derive(Default)]
struct Stats {
    fuzz_cases: u64,
}

fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn handle_syscall(emu: &mut Emulator) -> Result<(), VmExit> {
    todo!()
}

fn worker(emu_init: Arc<Emulator>, stats: Arc<Mutex<Stats>>) {
    // Fork the initial emulator state for this worker.
    let mut emu = emu_init.fork();

    loop {
        let mut local_stats = Stats::default();

        // Update global stats every 500M cycles.
        let it = rdtsc();
        while rdtsc() - it < 500_000_000 {
            // Reset memory to the initial state before starting fuzzing.
            emu.reset(&emu_init);

            // Run the target.
            loop {
                let vmexit = emu.run();

                match vmexit {
                    Err(VmExit::ECall) => handle_syscall(&mut emu).unwrap(),
                    Err(err) => panic!("Unexpected VM error: {}", err),
                    _ => unreachable!(),
                }
            }

            // Update local stats.
            local_stats.fuzz_cases += 1;
        }

        // Update global stats.
        let mut stats = stats.lock().unwrap();
        stats.fuzz_cases += local_stats.fuzz_cases;
    }
}

fn main() {
    const VM_MEM_SIZE: usize = 32 * 1024 * 1024;
    const STACK_SIZE: usize = 1024 * 1024;

    let mut emu_init = Emulator::new(VM_MEM_SIZE);

    // Load the program file.
    emu_init
        .load_program("testdata/hello")
        .unwrap_or_else(|err| {
            eprintln!("error: could not create emulator: {}", err);
            process::exit(1);
        });

    // Configure the stack.
    emu_init
        .mmu
        .set_perms(
            VirtAddr(VM_MEM_SIZE - STACK_SIZE),
            STACK_SIZE,
            Perm(PERM_READ | PERM_WRITE),
        )
        .unwrap_or_else(|err| {
            eprintln!("error: could not create stack: {}", err);
            process::exit(1);
        });
    emu_init
        .set_reg(RegAlias::Sp, VM_MEM_SIZE as u64 - 128)
        .unwrap_or_else(|err| {
            eprintln!("error: could not set the stack pointer: {}", err);
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

        println!("[{:10.4}] {:10.1} fcps", elapsed, fcps);
    }
}
