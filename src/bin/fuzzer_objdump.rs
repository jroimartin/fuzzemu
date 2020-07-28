use std::process;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use riscv_emu::emulator::Emulator;

const NCORES: usize = 1;

#[derive(Default)]
struct Stats {
    fuzz_cases: u64,
}

fn rdtsc() -> u64 {
    unsafe { core::arch::x86_64::_rdtsc() }
}

fn worker(emu_init: Arc<Emulator>, stats: Arc<Mutex<Stats>>) {
    let mut emu = emu_init.fork();

    loop {
        let mut local_stats = Stats::default();

        let it = rdtsc();
        while rdtsc() - it < 500_000_000 {
            emu.reset(&emu_init);
            if let Err(err) = emu.run() {
                panic!("VM error: {}", err);
            }
            local_stats.fuzz_cases += 1;
        }

        let mut stats = stats.lock().unwrap();
        stats.fuzz_cases += local_stats.fuzz_cases;
    }
}

fn main() {
    let mut emu_init = Emulator::new(32 * 1024 * 1024);
    emu_init
        .load_program("testdata/hello")
        .unwrap_or_else(|err| {
            eprintln!("error: could not create emulator: {}", err);
            process::exit(1);
        });
    let emu_init = Arc::new(emu_init);

    let stats = Arc::new(Mutex::new(Stats::default()));

    let start = Instant::now();

    for _ in 0..NCORES {
        let emu_init = Arc::clone(&emu_init);
        let stats = Arc::clone(&stats);

        thread::spawn(move || worker(emu_init, stats));
    }

    loop {
        thread::sleep(Duration::from_secs(1));

        let stats = stats.lock().unwrap();

        let elapsed = start.elapsed().as_secs_f64();
        let fcps = stats.fuzz_cases as f64 / elapsed;

        println!("[{:10.4}] {:10.1} fcps", elapsed, fcps);
    }
}
