//! riscv_emu provides a RISC-V emulator, as well as the building blocks used
//! to implement its internals.

#![feature(asm)]

pub mod elf;
pub mod emulator;
pub mod jit;
pub mod mmu;
