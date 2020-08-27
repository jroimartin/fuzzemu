//! RISC-V emulator. This implementation only supports the RV64i Base Integer
//! Instruction Set and assumes little-endian.

use std::cmp;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs;
use std::io;
use std::ops::Deref;
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::elf::{self, Elf};
use crate::jit::{self, JitCache};
use crate::mmu::{
    self, Mmu, Perm, VirtAddr, DIRTY_BLOCK_SIZE, PERM_EXEC, PERM_RAW,
    PERM_READ, PERM_WRITE,
};

/// Print debug messages.
const DEBUG: bool = false;

/// Maximum number of instructions to execute before returning a timeout.
const TIMEOUT: u64 = 100_000_000;

/// Emulator's exit reason.
#[derive(Debug)]
pub enum VmExit {
    Ebreak,
    Ecall,

    AddressMisaligned,
    InvalidInstruction,
    InvalidRegister,
    InvalidMemorySegment,
    UnimplementedInstruction,
    Timeout,

    IoError(io::Error),
    ElfError(elf::Error),
    MmuError(mmu::Error),
    NasmError(nasm::Error),
    JitError(jit::Error),
}

impl fmt::Display for VmExit {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            VmExit::Ebreak => write!(f, "EBREAK"),
            VmExit::Ecall => write!(f, "ECALL"),
            VmExit::AddressMisaligned => {
                write!(f, "address-missaligned exception")
            }
            VmExit::InvalidInstruction => write!(f, "invalid instruction"),
            VmExit::InvalidRegister => write!(f, "invalid register"),
            VmExit::InvalidMemorySegment => {
                write!(f, "invalid memory segment")
            }
            VmExit::UnimplementedInstruction => {
                write!(f, "unimplemented instruction")
            }
            VmExit::Timeout => write!(f, "timeout"),
            VmExit::IoError(err) => write!(f, "IO error: {}", err),
            VmExit::ElfError(err) => write!(f, "ELF error: {}", err),
            VmExit::MmuError(err) => write!(f, "MMU error: {}", err),
            VmExit::NasmError(err) => write!(f, "Nasm error: {}", err),
            VmExit::JitError(err) => write!(f, "JIT error: {}", err),
        }
    }
}

impl From<io::Error> for VmExit {
    fn from(error: io::Error) -> VmExit {
        VmExit::IoError(error)
    }
}

impl From<elf::Error> for VmExit {
    fn from(error: elf::Error) -> VmExit {
        VmExit::ElfError(error)
    }
}

impl From<mmu::Error> for VmExit {
    fn from(error: mmu::Error) -> VmExit {
        VmExit::MmuError(error)
    }
}

impl From<nasm::Error> for VmExit {
    fn from(error: nasm::Error) -> VmExit {
        VmExit::NasmError(error)
    }
}

impl From<jit::Error> for VmExit {
    fn from(error: jit::Error) -> VmExit {
        VmExit::JitError(error)
    }
}

/// A CPU Register.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Reg(pub u32);

impl Deref for Reg {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Alternative name for CPU registers.
///
/// Note that `Reg` implements the trait `From<RegAlias>`, which simplifies
/// referencing Registers by alias.
///
/// # Examples
///
/// All the following calls to `reg` are equivalent:
///
/// ```
/// use riscv_emu::emulator::{Emulator, Reg, RegAlias};
/// use riscv_emu::mmu::Mmu;
///
/// let mmu = Mmu::new(1024);
/// let mut emulator = Emulator::new(mmu);
///
/// emulator.reg(RegAlias::Zero);
/// emulator.reg(Reg::from(RegAlias::Zero));
/// emulator.reg(Reg(RegAlias::Zero as u32));
/// emulator.reg(Reg(0));
/// ```
///
/// For obvious reasons, the first option is recommended.
pub enum RegAlias {
    Zero = 0,
    Ra,
    Sp,
    Gp,
    Tp,
    T0,
    T1,
    T2,
    S0,
    S1,
    A0,
    A1,
    A2,
    A3,
    A4,
    A5,
    A6,
    A7,
    S2,
    S3,
    S4,
    S5,
    S6,
    S7,
    S8,
    S9,
    S10,
    S11,
    T3,
    T4,
    T5,
    T6,
    Pc,
}

impl From<RegAlias> for Reg {
    fn from(alias: RegAlias) -> Reg {
        Reg(alias as u32)
    }
}

/// Rtype encoding variant.
struct Rtype {
    funct7: u32,
    rs2: Reg,
    rs1: Reg,
    funct3: u32,
    rd: Reg,
}

impl From<u32> for Rtype {
    fn from(inst: u32) -> Rtype {
        let funct7 = (inst >> 25) & 0b111_1111;
        let rs2 = (inst >> 20) & 0b1_1111;
        let rs1 = (inst >> 15) & 0b1_1111;
        let funct3 = (inst >> 12) & 0b111;
        let rd = (inst >> 7) & 0b1_1111;

        Rtype {
            funct7,
            rs2: Reg(rs2),
            rs1: Reg(rs1),
            funct3,
            rd: Reg(rd),
        }
    }
}

/// Itype encoding variant.
struct Itype {
    imm: i32,
    rs1: Reg,
    funct3: u32,
    rd: Reg,
}

impl From<u32> for Itype {
    fn from(inst: u32) -> Itype {
        let imm110 = (inst >> 20) & 0b1111_1111_1111;
        let rs1 = (inst >> 15) & 0b1_1111;
        let funct3 = (inst >> 12) & 0b111;
        let rd = (inst >> 7) & 0b1_1111;

        let imm = ((imm110 as i32) << 20) >> 20;

        Itype {
            imm,
            rs1: Reg(rs1),
            funct3,
            rd: Reg(rd),
        }
    }
}

/// Stype encoding variant.
struct Stype {
    imm: i32,
    rs2: Reg,
    rs1: Reg,
    funct3: u32,
}

impl From<u32> for Stype {
    fn from(inst: u32) -> Stype {
        let imm115 = (inst >> 25) & 0b111_1111;
        let rs2 = (inst >> 20) & 0b1_1111;
        let rs1 = (inst >> 15) & 0b1_1111;
        let funct3 = (inst >> 12) & 0b111;
        let imm40 = (inst >> 7) & 0b1_1111;

        let imm = ((((imm115 << 5) | imm40) as i32) << 20) >> 20;

        Stype {
            imm,
            rs2: Reg(rs2),
            rs1: Reg(rs1),
            funct3,
        }
    }
}

/// Btype encoding variant.
struct Btype {
    imm: i32,
    rs2: Reg,
    rs1: Reg,
    funct3: u32,
}

impl From<u32> for Btype {
    fn from(inst: u32) -> Btype {
        let imm12 = (inst >> 31) & 0b1;
        let imm105 = (inst >> 25) & 0b11_1111;
        let rs2 = (inst >> 20) & 0b1_1111;
        let rs1 = (inst >> 15) & 0b1_1111;
        let funct3 = (inst >> 12) & 0b111;
        let imm41 = (inst >> 8) & 0b1111;
        let imm11 = (inst >> 7) & 0b1;

        let imm = (imm12 << 12) | (imm105 << 5) | (imm41 << 1) | (imm11 << 11);
        let imm = ((imm as i32) << 19) >> 19;

        Btype {
            imm,
            rs2: Reg(rs2),
            rs1: Reg(rs1),
            funct3,
        }
    }
}

/// Utype encoding variant.
struct Utype {
    imm: i32,
    rd: Reg,
}

impl From<u32> for Utype {
    fn from(inst: u32) -> Utype {
        let imm3112 = (inst >> 12) & 0b1111_1111_1111_1111_1111;
        let rd = (inst >> 7) & 0b1_1111;

        let imm = (imm3112 as i32) << 12;

        Utype { imm, rd: Reg(rd) }
    }
}

/// Jtype encoding variant.
struct Jtype {
    imm: i32,
    rd: Reg,
}

impl From<u32> for Jtype {
    fn from(inst: u32) -> Jtype {
        let imm20 = (inst >> 31) & 0b1;
        let imm101 = (inst >> 21) & 0b11_1111_1111;
        let imm11 = (inst >> 20) & 0b1;
        let imm1912 = (inst >> 12) & 0b1111_1111;
        let rd = (inst >> 7) & 0b1_1111;

        let imm =
            (imm20 << 20) | (imm101 << 1) | (imm11 << 11) | (imm1912 << 12);
        let imm = ((imm as i32) << 11) >> 11;

        Jtype { imm, rd: Reg(rd) }
    }
}

/// An execution trace.
pub struct Trace {
    /// Number of executed instructions.
    pub inst_execed: u64,

    /// Number of executed PCs.
    pub coverage: HashSet<VirtAddr>,
}

/// A callback called by a hook.
type HookCallback = fn(&mut Emulator) -> Result<(), VmExit>;

/// RISC-V emulator.
pub struct Emulator {
    /// State of the registers.
    regs: [u64; 33],

    /// MMU used by the emulator for memory operations.
    mmu: Mmu,

    /// JIT cache. If `Some`, execute code using JIT compilation. Otherwise,
    /// use emulation.
    ///
    /// The cache is shared among all the emulator instances and must be
    /// thread-safe, that's why it's wrapped inside Arc<Mutex<>>.
    jit_cache: Option<Arc<Mutex<JitCache>>>,

    /// User defined hooks. The callback will be called just before the
    /// instruction at the specific virtual address is executed.
    hooks: HashMap<VirtAddr, HookCallback>,
}

impl fmt::Display for Emulator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        const REG_STR: [&str; 33] = [
            "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1",
            "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3", "s4",
            "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5",
            "t6", "pc",
        ];

        let mut disp = String::new();
        disp.push_str("Registers:\n");
        for (i, reg_val) in self.regs.iter().enumerate() {
            let line = format!("  {:>4}: {:#010x} ", REG_STR[i], reg_val);
            disp.push_str(&line);
            if (i + 1) % 4 == 0 {
                disp.push('\n');
            }
        }
        write!(f, "{}", disp)
    }
}

impl Emulator {
    /// Returns a new emulator, its memory is handled by the passed MMU.
    pub fn new(mmu: Mmu) -> Emulator {
        Emulator {
            regs: [0; 33],
            mmu,
            jit_cache: None,
            hooks: HashMap::new(),
        }
    }

    /// Returns a reference to the internal MMU of the emulator.
    pub fn mmu(&self) -> &Mmu {
        &self.mmu
    }

    /// Returns a mutable reference to the internal MMU of the emulator.
    pub fn mmu_mut(&mut self) -> &mut Mmu {
        &mut self.mmu
    }

    /// Returns a copy of the Emulator, including its internal state.
    pub fn fork(&self) -> Emulator {
        let jit_cache = if let Some(cache) = &self.jit_cache {
            Some(Arc::clone(cache))
        } else {
            None
        };

        Emulator {
            regs: self.regs,
            mmu: self.mmu.fork(),
            jit_cache,
            hooks: self.hooks.clone(),
        }
    }

    /// Resets the internal state of the emulator to the given state `other`.
    pub fn reset(&mut self, other: &Emulator) {
        self.regs = other.regs;
        self.mmu.reset(&other.mmu);
    }

    /// Enable JIT compilation. `cache` is the JIT cache used to store the
    /// compiled instructions.
    pub fn with_jit(mut self, cache: JitCache) -> Emulator {
        let cache = Arc::new(Mutex::new(cache));

        self.jit_cache = Some(cache);
        self
    }

    /// Loads an ELF program in the emulator. It also points the program
    /// counter to the entrypoint of the program and sets the program break.
    pub fn load_program<P: AsRef<Path>>(
        &mut self,
        program: P,
    ) -> Result<(), VmExit> {
        let contents = fs::read(program)?;
        let elf = Elf::parse(&contents)?;

        let mut max_addr = 0;

        for phdr in elf.phdrs() {
            let file_offset = phdr.offset();
            let file_end = file_offset
                .checked_add(phdr.file_size())
                .ok_or(VmExit::InvalidMemorySegment)?;

            let file_bytes = contents
                .get(file_offset..file_end)
                .ok_or(VmExit::InvalidMemorySegment)?;

            let mem_start = phdr.virt_addr();
            let mem_size = phdr.mem_size();

            self.mmu.poke(mem_start, file_bytes)?;
            self.mmu.set_perms(mem_start, mem_size, phdr.perms())?;

            // checked_add() is not needed here because integer overflows have
            // been already checked in the previous call to set_perms().
            let mem_end = *mem_start + mem_size;

            max_addr = cmp::max(max_addr, mem_end);
        }

        // Place the program counter in the entrypoint.
        self.set_reg(RegAlias::Pc, *elf.entry() as u64)?;

        // Set the program break to point just after the end of the process's
        // memory. 16-byte aligned.
        let max_addr_aligned = max_addr
            .checked_add(0xf)
            .ok_or(VmExit::InvalidMemorySegment)?
            & !0xf;
        self.mmu.set_brk(VirtAddr(max_addr_aligned));

        Ok(())
    }

    /// Sets the value of the register `reg` to `val`.
    pub fn set_reg<R: Into<Reg>>(
        &mut self,
        reg: R,
        val: u64,
    ) -> Result<(), VmExit> {
        let reg = *reg.into() as usize;

        if reg >= self.regs.len() {
            return Err(VmExit::InvalidRegister);
        }

        // The zero register is always 0.
        if reg != RegAlias::Zero as usize {
            self.regs[reg] = val;
        }
        Ok(())
    }

    /// Returns the value stored in the register `reg`.
    pub fn reg<R: Into<Reg>>(&self, reg: R) -> Result<u64, VmExit> {
        let reg = *reg.into() as usize;

        if reg >= self.regs.len() {
            return Err(VmExit::InvalidRegister);
        }

        // The zero register is always 0.
        if reg == RegAlias::Zero as usize {
            Ok(0)
        } else {
            Ok(self.regs[reg])
        }
    }

    /// Hooks the virtual address `addr`. `cb` is the callback called just
    /// before the instruction at `addr` is executed.
    pub fn hook(&mut self, addr: VirtAddr, cb: HookCallback) {
        self.hooks.insert(addr, cb);
    }

    /// Run until vm exit or error. It returns the execution trace in `trace`.
    pub fn run(&mut self, trace: &mut Trace) -> Result<(), VmExit> {
        if self.jit_cache.is_some() {
            self.run_jit(trace)
        } else {
            self.run_emu(trace)
        }
    }

    /// Run code using pure emulation.
    pub fn run_emu(&mut self, trace: &mut Trace) -> Result<(), VmExit> {
        loop {
            let pc = self.reg(RegAlias::Pc)?;

            if let Some(hook_callback) = self.hooks.get(&VirtAddr(pc as usize))
            {
                hook_callback(self)?;

                // If the hook has changed the PC, continue execution in that
                // position. Otherwise, just continue executing the hooked
                // instruction.
                if self.reg(RegAlias::Pc)? != pc {
                    continue;
                }
            }

            if pc & 3 != 0 {
                return Err(VmExit::AddressMisaligned);
            }

            let inst = self.mmu.read_int_with_perms::<u32>(
                VirtAddr(pc as usize),
                Perm(PERM_EXEC),
            )?;

            if trace.inst_execed > TIMEOUT {
                return Err(VmExit::Timeout);
            }

            self.emulate_instruction(pc, inst)?;

            // Update trace.
            trace.inst_execed += 1;
            trace.coverage.insert(VirtAddr(pc as usize));
        }
    }

    /// Emulates a single instruction, updating the internal state of the
    /// emulator.
    fn emulate_instruction(
        &mut self,
        pc: u64,
        inst: u32,
    ) -> Result<(), VmExit> {
        let opcode = inst & 0b111_1111;

        if DEBUG {
            eprintln!("---");
            eprintln!("{}", self);
            eprintln!("{:#010x}: {:08x} {:07b}", pc, inst, opcode);
        }

        match opcode {
            0b0110111 => {
                // LUI
                let dec = Utype::from(inst);

                self.set_reg(dec.rd, dec.imm as u64)?;
            }
            0b0010111 => {
                // AUIPIC
                let dec = Utype::from(inst);

                self.set_reg(dec.rd, pc.wrapping_add(dec.imm as u64))?;
            }
            0b1101111 => {
                // JAL
                let dec = Jtype::from(inst);

                let offset = dec.imm as u64;

                self.set_reg(dec.rd, pc.wrapping_add(4))?;
                self.set_reg(RegAlias::Pc, pc.wrapping_add(offset))?;
                return Ok(());
            }
            0b1100111 => {
                let dec = Itype::from(inst);

                let offset = dec.imm as u64;
                let rs1 = self.reg(dec.rs1)?;

                match dec.funct3 {
                    0b000 => {
                        // JALR
                        self.set_reg(dec.rd, pc.wrapping_add(4))?;
                        self.set_reg(
                            RegAlias::Pc,
                            rs1.wrapping_add(offset) >> 1 << 1,
                        )?;
                        return Ok(());
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b1100011 => {
                let dec = Btype::from(inst);

                let offset = dec.imm as u64;
                let rs1 = self.reg(dec.rs1)?;
                let rs2 = self.reg(dec.rs2)?;

                match dec.funct3 {
                    0b000 => {
                        // BEQ
                        if rs1 == rs2 {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b001 => {
                        // BNE
                        if rs1 != rs2 {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b100 => {
                        // BLT
                        if (rs1 as i64) < (rs2 as i64) {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b101 => {
                        // BGE
                        if (rs1 as i64) >= (rs2 as i64) {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b110 => {
                        // BLTU
                        if rs1 < rs2 {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b111 => {
                        // BGEU
                        if rs1 >= rs2 {
                            self.set_reg(
                                RegAlias::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0000011 => {
                let dec = Itype::from(inst);

                let rs1 = self.reg(dec.rs1)?;
                let offset = dec.imm as u64;
                let vaddr = rs1.wrapping_add(offset);

                let vaddr = VirtAddr(vaddr as usize);

                match dec.funct3 {
                    0b000 => {
                        // LB
                        let value = self.mmu.read_int::<i8>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b001 => {
                        // LH
                        let value = self.mmu.read_int::<i16>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b010 => {
                        // LW
                        let value = self.mmu.read_int::<i32>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b100 => {
                        // LBU
                        let value = self.mmu.read_int::<u8>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b101 => {
                        // LHU
                        let value = self.mmu.read_int::<u16>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b110 => {
                        // LWU
                        let value = self.mmu.read_int::<u32>(vaddr)?;
                        self.set_reg(dec.rd, value as u64)?;
                    }
                    0b011 => {
                        // LD
                        let value = self.mmu.read_int::<u64>(vaddr)?;
                        self.set_reg(dec.rd, value)?;
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0100011 => {
                let dec = Stype::from(inst);

                let rs1 = self.reg(dec.rs1)?;
                let rs2 = self.reg(dec.rs2)?;
                let offset = dec.imm as u64;
                let vaddr = rs1.wrapping_add(offset);

                let vaddr = VirtAddr(vaddr as usize);

                match dec.funct3 {
                    0b000 => {
                        // SB
                        self.mmu.write_int::<u8>(vaddr, rs2 as u8)?;
                    }
                    0b001 => {
                        // SH
                        self.mmu.write_int::<u16>(vaddr, rs2 as u16)?;
                    }
                    0b010 => {
                        // SW
                        self.mmu.write_int::<u32>(vaddr, rs2 as u32)?;
                    }
                    0b011 => {
                        // SD
                        self.mmu.write_int::<u64>(vaddr, rs2)?;
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0010011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u64;
                let rs1 = self.reg(dec.rs1)?;

                match dec.funct3 {
                    0b000 => {
                        // ADDI
                        self.set_reg(dec.rd, rs1.wrapping_add(imm))?;
                    }
                    0b010 => {
                        // SLTI
                        if (rs1 as i64) < (imm as i64) {
                            self.set_reg(dec.rd, 1)?;
                        } else {
                            self.set_reg(dec.rd, 0)?;
                        }
                    }
                    0b011 => {
                        // SLTIU
                        if rs1 < imm {
                            self.set_reg(dec.rd, 1)?;
                        } else {
                            self.set_reg(dec.rd, 0)?;
                        }
                    }
                    0b100 => {
                        // XORI
                        self.set_reg(dec.rd, rs1 ^ imm)?;
                    }
                    0b110 => {
                        // ORI
                        self.set_reg(dec.rd, rs1 | imm)?;
                    }
                    0b111 => {
                        // ANDI
                        self.set_reg(dec.rd, rs1 & imm)?;
                    }
                    0b001 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SLLI
                                let shamt = dec.imm & 0b11_1111;
                                self.set_reg(dec.rd, rs1 << shamt)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SRLI
                                let shamt = dec.imm & 0b11_1111;
                                self.set_reg(dec.rd, rs1 >> shamt)?;
                            }
                            0b010000 => {
                                // SRAI
                                let shamt = dec.imm & 0b1_11111;
                                let value = ((rs1 as i64) >> shamt) as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0110011 => {
                let dec = Rtype::from(inst);

                let rs1 = self.reg(dec.rs1)?;
                let rs2 = self.reg(dec.rs2)?;

                match dec.funct3 {
                    0b000 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // ADD
                                self.set_reg(dec.rd, rs1.wrapping_add(rs2))?;
                            }
                            0b0100000 => {
                                // SUB
                                self.set_reg(dec.rd, rs1.wrapping_sub(rs2))?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b001 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLL
                                let shamt = rs2 & 0b1_1111;
                                self.set_reg(dec.rd, rs1 << shamt)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b010 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLT
                                if (rs1 as i64) < (rs2 as i64) {
                                    self.set_reg(dec.rd, 1)?;
                                } else {
                                    self.set_reg(dec.rd, 0)?;
                                }
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b011 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLTU
                                if rs1 < rs2 {
                                    self.set_reg(dec.rd, 1)?;
                                } else {
                                    self.set_reg(dec.rd, 0)?;
                                }
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b100 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // XOR
                                self.set_reg(dec.rd, rs1 ^ rs2)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SRL
                                let shamt = rs2 & 0b1_1111;
                                self.set_reg(dec.rd, rs1 >> shamt)?;
                            }
                            0b0100000 => {
                                // SRA
                                let shamt = rs2 & 0b1_1111;
                                let value = ((rs1 as i64) >> shamt) as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b110 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // OR
                                self.set_reg(dec.rd, rs1 | rs2)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b111 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // AND
                                self.set_reg(dec.rd, rs1 & rs2)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0001111 => {
                // FENCE
                return Err(VmExit::UnimplementedInstruction);
            }
            0b1110011 => {
                let dec = Itype::from(inst);

                if *dec.rd == 0 && dec.funct3 == 0 && *dec.rs1 == 0 {
                    if dec.imm == 0 {
                        // ECALL
                        return Err(VmExit::Ecall);
                    } else if dec.imm == 1 {
                        // EBREAK
                        return Err(VmExit::Ebreak);
                    } else {
                        return Err(VmExit::InvalidInstruction);
                    }
                } else {
                    return Err(VmExit::InvalidInstruction);
                }
            }
            0b0011011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u32;
                let rs1 = self.reg(dec.rs1)? as u32;

                match dec.funct3 {
                    0b000 => {
                        // ADDIW
                        let value = rs1.wrapping_add(imm) as i32 as u64;
                        self.set_reg(dec.rd, value)?;
                    }
                    0b001 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SLLIW
                                let shamt = dec.imm & 0b11_1111;
                                let value = (rs1 << shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SRLIW
                                let shamt = dec.imm & 0b11_1111;
                                let value = (rs1 >> shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            0b010000 => {
                                // SRAIW
                                let shamt = dec.imm & 0b11_1111;
                                let value =
                                    ((rs1 as i32) >> shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0111011 => {
                let dec = Rtype::from(inst);

                let rs1 = self.reg(dec.rs1)? as u32;
                let rs2 = self.reg(dec.rs2)? as u32;

                match dec.funct3 {
                    0b000 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //ADDW
                                let value =
                                    rs1.wrapping_add(rs2) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            0b0100000 => {
                                //SUBW
                                let value =
                                    rs1.wrapping_sub(rs2) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b001 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //SLLW
                                let shamt = rs2 & 0b11_1111;
                                let value = (rs1 << shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //SRLW
                                let shamt = rs2 & 0b11_1111;
                                let value = (rs1 >> shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            0b0100000 => {
                                //SRAW
                                let shamt = rs2 & 0b11_1111;
                                let value = ((rs1 as i32) >> shamt) as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            _ => return Err(VmExit::InvalidInstruction),
        }

        self.set_reg(RegAlias::Pc, pc.wrapping_add(4))?;

        Ok(())
    }

    /// Run code using JIT compilation.
    ///
    /// # Panics
    ///
    /// This function will panic if the Emulator's JIT cache has not been
    /// initialized using `with_jit`.
    ///
    /// # Calling convention
    ///
    /// Input:
    /// - `r8`: Number of executed instructions.
    /// - `r9`: JIT cache lookup table.
    /// - `r10`: Emulator registers.
    /// - `r11`: MMU memory.
    /// - `r12`: MMU dirty blocks.
    /// - `r13`: MMU dirty bitmap.
    /// - `r14`: MMU dirty length.
    /// - `r15`: MMU memory permissions.
    ///
    /// Output:
    /// - `rax`: JIT exit reason.
    ///   - `rax=0`: JIT cache lookup error.
    ///   - `rax=1`: ECALL exception.
    ///   - `rax=2`: EBREAK exception.
    ///   - `rax=3`: Read fault, `rcx`: Memory address, `rdx`: Size.
    ///   - `rax=4`: Write fault, `rcx`: Memory address, `rdx`: Size.
    ///   - `rax=5`: Uninit fault, `rcx`: Memory address, `rdx`: Size.
    ///   - `rax=6`: Timeout.
    ///   - `rax=7`: Hook. `rcx`: reentry address.
    /// - `rbx`: Next PC. In the case of an exception (EBREAK, ECALL or
    ///   read/write fault) or a hook, it's the address of the instruction
    ///   causing the exception.
    /// - `rcx`: Extra information.
    /// - `rdx`: Extra information.
    /// - `r8`: Updated number of executed instructions.
    /// - `r14`: Updated Mmu dirty len.
    pub fn run_jit(&mut self, trace: &mut Trace) -> Result<(), VmExit> {
        let mut pc = self.reg(RegAlias::Pc)?;
        let mut hook_reentry = None;

        loop {
            let block_ptr = if let Some(ptr) = hook_reentry.take() {
                ptr
            } else {
                if pc & 3 != 0 {
                    return Err(VmExit::AddressMisaligned);
                }

                let (lookup_table_len, block_lookup) = {
                    let jit_cache =
                        self.jit_cache.as_ref().unwrap().lock().unwrap();

                    (
                        jit_cache.lookup_table_len(),
                        jit_cache.lookup(VirtAddr(pc as usize)),
                    )
                };

                if let Some(ptr) = block_lookup {
                    ptr
                } else {
                    let block =
                        self.lift_block(pc, lookup_table_len, trace)?;

                    let mut jit_cache =
                        self.jit_cache.as_ref().unwrap().lock().unwrap();
                    jit_cache.insert(VirtAddr(pc as usize), block)?
                }
            };

            let lookup_table_ptr = {
                let jit_cache =
                    self.jit_cache.as_ref().unwrap().lock().unwrap();
                jit_cache.lookup_table_ptr()
            };

            let regs_ptr = self.regs.as_ptr();
            let memory_ptr = self.mmu.memory_ptr();
            let dirty_ptr = self.mmu.dirty_ptr();
            let dirty_bitmap_ptr = self.mmu.dirty_bitmap_ptr();
            let mut dirty_len = self.mmu.dirty_len();
            let perms_ptr = self.mmu.perms_ptr();

            let mut inst_execed = trace.inst_execed;

            let jit_exit: u64;
            let next_pc: u64;
            let rcx: u64;
            let rdx: u64;

            unsafe {
                asm!("call {block_ptr}",
                     block_ptr = in(reg) block_ptr,
                     inout("r8") inst_execed,
                     in("r9") lookup_table_ptr,
                     in("r10") regs_ptr,
                     in("r11") memory_ptr,
                     in("r12") dirty_ptr,
                     in("r13") dirty_bitmap_ptr,
                     inout("r14") dirty_len,
                     in("r15") perms_ptr,
                     out("rax") jit_exit,
                     out("rbx") next_pc,
                     out("rcx") rcx,
                     out("rdx") rdx,
                );
            }

            if DEBUG {
                eprintln!(
                    "jit_exit={:#x} next_pc={:#x} inst_execed={} dirty_len={}",
                    jit_exit, next_pc, inst_execed, dirty_len
                );
            }

            // Update trace.
            trace.inst_execed = inst_execed;

            // Update the length of the list of dirty blocks with the new
            // value.
            unsafe {
                self.mmu.set_dirty_len(dirty_len);
            }

            // Update the PC register with the next PC. Needed by exception
            // handlers and debug messages.
            self.set_reg(RegAlias::Pc, next_pc)?;

            match jit_exit {
                0 => {
                    pc = next_pc;
                    continue;
                }
                1 => {
                    return Err(VmExit::Ecall);
                }
                2 => {
                    return Err(VmExit::Ebreak);
                }
                3 => {
                    return Err(VmExit::MmuError(mmu::Error::ReadFault {
                        addr: VirtAddr(rcx as usize),
                        size: rdx as usize,
                    }));
                }
                4 => {
                    return Err(VmExit::MmuError(mmu::Error::WriteFault {
                        addr: VirtAddr(rcx as usize),
                        size: rdx as usize,
                    }));
                }
                5 => {
                    return Err(VmExit::MmuError(mmu::Error::UninitFault {
                        addr: VirtAddr(rcx as usize),
                        size: rdx as usize,
                    }));
                }
                6 => return Err(VmExit::Timeout),
                7 => {
                    if let Some(hook_callback) =
                        self.hooks.get(&VirtAddr(next_pc as usize))
                    {
                        hook_callback(self)?;

                        // If the hook has changed the PC, continue execution
                        // in that position. Otherwise, just continue executing
                        // the hooked instruction.
                        let hook_pc = self.reg(RegAlias::Pc)?;
                        if hook_pc != next_pc {
                            pc = hook_pc;
                        } else {
                            hook_reentry = Some(rcx as *const u8);
                        }
                        continue;
                    } else {
                        panic!(
                            "could not find hook's callback: {:#x}",
                            next_pc
                        );
                    }
                }
                _ => unimplemented!("unknown jit_exit value"),
            }
        }
    }

    /// Lifts a basic block.
    fn lift_block(
        &mut self,
        pc: u64,
        lookup_table_len: usize,
        trace: &mut Trace,
    ) -> Result<Vec<u8>, VmExit> {
        let mut block_code = String::new();
        let mut cur_pc = pc;

        if DEBUG {
            eprintln!("lifting {:#010x}", pc);
        }

        loop {
            let inst = self.mmu.read_int_with_perms::<u32>(
                VirtAddr(cur_pc as usize),
                Perm(PERM_EXEC),
            )?;

            // Update trace.
            trace.coverage.insert(VirtAddr(cur_pc as usize));

            match self.lift_instruction(cur_pc, inst, lookup_table_len) {
                Ok((inst_code, end)) => {
                    block_code.push_str(&format!(
                        "
                            inst_{cur_pc:x}:
                        ",
                        cur_pc = cur_pc,
                    ));

                    if self.hooks.contains_key(&VirtAddr(cur_pc as usize)) {
                        block_code.push_str(&format!(
                            "
                                mov rax, 7
                                mov rbx, {cur_pc}
                                lea rcx, [rel .hook_reentry]
                                ret

                                .hook_reentry:
                            ",
                            cur_pc = cur_pc,
                        ));
                    }

                    block_code.push_str(&format!(
                        "
                            add r8, 1

                            {inst_code}
                        ",
                        inst_code = inst_code
                    ));

                    cur_pc = cur_pc.wrapping_add(4);

                    if end {
                        break;
                    }
                }
                Err(err) => return Err(err),
            }
        }

        let code = format!(
            "
                BITS 64

                ; Exit with timeout if the number of executed instructions is
                ; too high.
                mov rax, {timeout}
                cmp r8, rax
                jb .notimeout
                mov rax, 6
                mov rbx, {pc}
                ret

                .notimeout:
                {block_code}
            ",
            pc = pc,
            block_code = block_code,
            timeout = TIMEOUT
        );

        let block = match nasm::assemble(&code) {
            Ok(bytes) => bytes,
            Err(err) => {
                if DEBUG {
                    eprintln!("{}", &code);
                }
                return Err(err.into());
            }
        };

        Ok(block)
    }

    /// Lifts a single instruction. It returns a String containing the lifted
    /// assembly code and a boolean signaling if the lifted instruction is the
    /// end of the block.
    fn lift_instruction(
        &mut self,
        pc: u64,
        inst: u32,
        lookup_table_len: usize,
    ) -> Result<(String, bool), VmExit> {
        let opcode = inst & 0b111_1111;

        let mut code = String::new();

        // Returns a `String` containing the asm code to write into a RISC-V
        // register.
        //
        // We don't need to check for OOB here because instruction encoding
        // forces registers to be in the range [0, 31].
        macro_rules! write_reg {
            ($dst_riscv_reg:expr, $src:expr) => {
                if *$dst_riscv_reg == RegAlias::Zero as u32 {
                    String::from("\n")
                } else {
                    format!(
                        "mov qword [r10+8*{riscv_reg}], {src}\n",
                        riscv_reg = *$dst_riscv_reg,
                        src = $src
                    )
                }
            };
        }

        // Returns a `String` containing the asm code to read from a RISC-V
        // register.
        //
        // We don't need to check for OOB here because instruction encoding
        // forces registers to be in the range [0, 31].
        macro_rules! read_reg {
            ($src_riscv_reg:expr, $dst:expr) => {
                if *$src_riscv_reg == RegAlias::Zero as u32 {
                    format!("xor {dst}, {dst}\n", dst = $dst)
                } else {
                    format!(
                        "mov {dst}, qword [r10+8*{riscv_reg}]\n",
                        dst = $dst,
                        riscv_reg = *$src_riscv_reg
                    )
                }
            };
        }

        // Returns a `String` containing the asm code to perform a jit cache
        // lookup, jumping to the lifted block if found. Otherwise, it will
        // exit the JIt with rax=0 and rbx=target.
        //
        // It clobbers the registers `rax` and `rbx` and uses the local label
        // `.lookup_error`.
        macro_rules! cache_lookup {
            ($target:expr) => {
                format!(
                    "
                        mov rbx, {target}
                        mov rax, rbx
                        shr rax, 2
                        cmp rax, {lookup_table_len}
                        jae .lookup_error_{target}
                        mov rax, [r9+8*rax]
                        test rax, rax
                        jz .lookup_error_{target}
                        jmp rax
                        .lookup_error_{target}:
                        xor rax, rax
                        ret
                    ",
                    target = $target,
                    lookup_table_len = lookup_table_len
                )
            };
        }

        match opcode {
            0b0110111 => {
                // LUI
                let dec = Utype::from(inst);

                code.push_str(&write_reg!(dec.rd, dec.imm as u64));
            }
            0b0010111 => {
                // AUIPIC
                let dec = Utype::from(inst);

                code.push_str(&write_reg!(
                    dec.rd,
                    pc.wrapping_add(dec.imm as u64)
                ));
            }
            0b1101111 => {
                // JAL
                let dec = Jtype::from(inst);

                let offset = dec.imm as u64;

                code.push_str(&write_reg!(dec.rd, pc.wrapping_add(4)));
                code.push_str(&cache_lookup!(pc.wrapping_add(offset)));

                return Ok((code, true));
            }
            0b1100111 => {
                let dec = Itype::from(inst);

                let offset = dec.imm as u64;

                match dec.funct3 {
                    0b000 => {
                        // JALR
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&write_reg!(dec.rd, pc.wrapping_add(4)));
                        code.push_str(&format!(
                            "
                                add rax, {offset}
                                shr rax, 1
                                shl rax, 1
                                {cache_lookup}
                            ",
                            offset = offset as i32,
                            cache_lookup = cache_lookup!("rax")
                        ));

                        return Ok((code, true));
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b1100011 => {
                let dec = Btype::from(inst);

                let offset = dec.imm as u64;

                let cmp_inst = match dec.funct3 {
                    0b000 => "jne",  // BEQ
                    0b001 => "je",   // BNE
                    0b100 => "jnl",  // BLT
                    0b101 => "jnge", // BGE
                    0b110 => "jnb",  // BLTU
                    0b111 => "jnae", // BGEU
                    _ => return Err(VmExit::InvalidInstruction),
                };

                code.push_str(&read_reg!(dec.rs1, "rcx"));
                code.push_str(&read_reg!(dec.rs2, "rdx"));
                code.push_str(&format!(
                    "
                        cmp rcx, rdx
                        {cmp_inst} .out
                        {cache_lookup_true}
                        .out:
                        {cache_lookup_false}
                    ",
                    cmp_inst = cmp_inst,
                    cache_lookup_true = cache_lookup!(pc.wrapping_add(offset)),
                    cache_lookup_false = cache_lookup!(pc.wrapping_add(4))
                ));

                return Ok((code, true));
            }
            0b0000011 => {
                let dec = Itype::from(inst);

                let offset = dec.imm as u64;

                let (mov, movzx, size_mod, rax, movzx_rax, size) =
                    match dec.funct3 {
                        0b000 => ("movsx", "movzx", "byte", "rax", "rax", 1), // LB
                        0b001 => ("movsx", "movzx", "word", "rax", "rax", 2), // LH
                        0b010 => ("movsx", "mov", "dword", "rax", "eax", 4), // LW
                        0b100 => ("movzx", "movzx", "byte", "rax", "rax", 1), // LBU
                        0b101 => ("movzx", "movzx", "word", "rax", "rax", 2), // LHU
                        0b110 => ("mov", "mov", "dword", "eax", "eax", 4), // LWU
                        0b011 => ("mov", "mov", "qword", "rax", "rax", 8), // LD
                        _ => return Err(VmExit::InvalidInstruction),
                    };

                let mut read_mask = 0u64;
                let mut raw_mask = 0u64;
                for i in 0..size {
                    read_mask |= (PERM_READ as u64) << (i * 8);
                    raw_mask |= (PERM_RAW as u64) << (i * 8);
                }

                code.push_str(&read_reg!(dec.rs1, "rcx"));
                code.push_str(&format!(
                    "
                        add rcx, {offset}

                        ; Check memory boundaries.
                        cmp rcx, {memory_len} - {size}
                        ja .read_fault

                        ; Check uninit.
                        {movzx} {movzx_rax}, {size_mod} [r15+rcx]
                        mov rbx, {raw_mask}
                        and rax, rbx
                        jnz .uninit_fault

                        ; Check unreadable.
                        {movzx} {movzx_rax}, {size_mod} [r15+rcx]
                        mov rbx, {read_mask}
                        and rax, rbx
                        cmp rax, rbx
                        jne .read_fault

                        ; Read.
                        {mov} {rax}, {size_mod} [r11+rcx]
                        jmp .out

                        .uninit_fault:
                        mov rax, 5
                        mov rbx, {pc}
                        mov rdx, {size}
                        ret

                        .read_fault:
                        mov rax, 3
                        mov rbx, {pc}
                        mov rdx, {size}
                        ret

                        .out:
                    ",
                    mov = mov,
                    movzx = movzx,
                    size_mod = size_mod,
                    rax = rax,
                    movzx_rax = movzx_rax,
                    size = size,
                    offset = offset as i32,
                    memory_len = self.mmu.memory_len(),
                    read_mask = read_mask,
                    raw_mask = raw_mask,
                    pc = pc
                ));
                code.push_str(&write_reg!(dec.rd, "rax"));
            }
            0b0100011 => {
                let dec = Stype::from(inst);

                let offset = dec.imm as u64;

                let (movzx, size_mod, rax, movzx_rax, size) = match dec.funct3
                {
                    0b000 => ("movzx", "byte", "al", "rax", 1), // SB
                    0b001 => ("movzx", "word", "ax", "rax", 2), // SH
                    0b010 => ("mov", "dword", "eax", "eax", 4), // SW
                    0b011 => ("mov", "qword", "rax", "rax", 8), // SD
                    _ => return Err(VmExit::InvalidInstruction),
                };

                let mut write_mask = 0u64;
                let mut raw_mask = 0u64;
                for i in 0..size {
                    write_mask |= (PERM_WRITE as u64) << (i * 8);
                    raw_mask |= (PERM_RAW as u64) << (i * 8);
                }

                // Check DIRTY_BLOCK_SIZE fits the requirements.
                assert_eq!(
                    DIRTY_BLOCK_SIZE.count_ones(),
                    1,
                    "DIRTY_BLOCK_SIZE must be a power of two"
                );
                let dirty_bs_shift = DIRTY_BLOCK_SIZE.trailing_zeros();

                code.push_str(&read_reg!(dec.rs1, "rcx"));
                code.push_str(&read_reg!(dec.rs2, "rbx"));
                code.push_str(&format!(
                    "
                        add rcx, {offset}

                        ; Check memory boundaries.
                        cmp rcx, {memory_len} - {size}
                        ja .fault

                        ; Check write.
                        {movzx} {movzx_rax}, {size_mod} [r15+rcx]
                        mov rdx, {write_mask}
                        and rax, rdx
                        cmp rax, rdx
                        jne .fault

                        ; Remove PERM_RAW and add PERM_READ.
                        {movzx} {movzx_rax}, {size_mod} [r15+rcx]
                        mov rdx, {raw_mask}
                        and rdx, rax
                        xor rax, rdx
                        shr rdx, 1
                        or rax, rdx
                        mov {size_mod} [r15+rcx], {rax}

                        ; Write.
                        mov rax, rbx
                        mov {size_mod} [r11+rcx], {rax}

                        ; Be conservative and mark both the starting block and
                        ; the next one as dirty. Computing if the second block
                        ; is dirty is more expensive than resetting more
                        ; memory blocks.
                        shr rcx, {dirty_bs_shift}

                        bts qword [r13], rcx
                        jc .next_block

                        ; Mark starting block as dirty.
                        mov qword [r12+8*r14], rcx
                        add r14, 1

                        .next_block:

                        ; FIXME: Off-by-one.
                        ;; Mark following block as dirty.
                        ;add rcx, 1
                        ;bts qword [r13], rcx
                        ;jc .out
                        ;mov qword [r12+8*r14], rcx
                        ;add r14, 1
                        jmp .out

                        .fault:
                        mov rax, 4
                        mov rbx, {pc}
                        mov rdx, {size}
                        ret

                        .out:
                    ",
                    movzx = movzx,
                    size_mod = size_mod,
                    rax = rax,
                    movzx_rax = movzx_rax,
                    size = size,
                    offset = offset as i32,
                    memory_len = self.mmu.memory_len(),
                    dirty_bs_shift = dirty_bs_shift,
                    write_mask = write_mask,
                    raw_mask = raw_mask,
                    pc = pc
                ));
            }
            0b0010011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u64;

                match dec.funct3 {
                    0b000 => {
                        // ADDI
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                add rax, {imm}
                            ",
                            imm = imm as i32
                        ));
                        code.push_str(&write_reg!(dec.rd, "rax"));
                    }
                    0b010 => {
                        // SLTI
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                xor rcx, rcx
                                mov rbx, {imm}
                                cmp rax, rbx
                                jnl .out
                                add rcx, 1
                                .out:
                            ",
                            imm = imm
                        ));
                        code.push_str(&write_reg!(dec.rd, "rcx"));
                    }
                    0b011 => {
                        // SLTIU
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                xor rcx, rcx
                                mov rbx, {imm}
                                cmp rax, rbx
                                jnb .out
                                add rcx, 1
                                .out:
                            ",
                            imm = imm
                        ));
                        code.push_str(&write_reg!(dec.rd, "rcx"));
                    }
                    0b100 => {
                        // XORI
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                xor rax, {imm:#x}
                            ",
                            imm = imm
                        ));
                        code.push_str(&write_reg!(dec.rd, "rax"));
                    }
                    0b110 => {
                        // ORI
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                or rax, {imm:#x}
                            ",
                            imm = imm
                        ));
                        code.push_str(&write_reg!(dec.rd, "rax"));
                    }
                    0b111 => {
                        // ANDI
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                and rax, {imm:#x}
                            ",
                            imm = imm
                        ));
                        code.push_str(&write_reg!(dec.rd, "rax"));
                    }
                    0b001 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SLLI
                                let shamt = dec.imm & 0b11_1111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        shl rax, {shamt:#x}
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SRLI
                                let shamt = dec.imm & 0b11_1111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        shr rax, {shamt:#x}
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b010000 => {
                                // SRAI
                                let shamt = dec.imm & 0b1_11111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        sar rax, {shamt:#x}
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0110011 => {
                let dec = Rtype::from(inst);

                match dec.funct3 {
                    0b000 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // ADD
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        add rax, rbx
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b0100000 => {
                                // SUB
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        sub rax, rbx
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b001 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLL
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        shl rax, cl
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b010 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLT
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        xor rcx, rcx
                                        cmp rax, rbx
                                        jnl .out
                                        add rcx, 1
                                        .out:
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rcx"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b011 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLTU
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        xor rcx, rcx
                                        cmp rax, rbx
                                        jnb .out
                                        add rcx, 1
                                        .out:
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rcx"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b100 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // XOR
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        xor rax, rbx
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SRL
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        shr rax, cl
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b0100000 => {
                                // SRA
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        sar rax, cl
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b110 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // OR
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        or rax, rbx
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b111 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // AND
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        and rax, rbx
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0001111 => {
                // FENCE
                return Err(VmExit::UnimplementedInstruction);
            }
            0b1110011 => {
                let dec = Itype::from(inst);

                if *dec.rd == 0 && dec.funct3 == 0 && *dec.rs1 == 0 {
                    if dec.imm == 0 {
                        // ECALL
                        code.push_str(&format!(
                            "
                                mov rax, 1
                                mov rbx, {pc:#x}
                                ret
                            ",
                            pc = pc,
                        ));
                        return Ok((code, true));
                    } else if dec.imm == 1 {
                        // EBREAK
                        code.push_str(&format!(
                            "
                                mov rax, 2
                                mov rbx, {pc:#x}
                                ret
                            ",
                            pc = pc,
                        ));
                        return Ok((code, true));
                    } else {
                        return Err(VmExit::InvalidInstruction);
                    }
                } else {
                    return Err(VmExit::InvalidInstruction);
                }
            }
            0b0011011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u32;

                match dec.funct3 {
                    0b000 => {
                        // ADDIW
                        code.push_str(&read_reg!(dec.rs1, "rax"));
                        code.push_str(&format!(
                            "
                                add eax, {imm}
                                movsx rax, eax
                            ",
                            imm = imm as i32
                        ));
                        code.push_str(&write_reg!(dec.rd, "rax"));
                    }
                    0b001 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SLLIW
                                let shamt = dec.imm & 0b11_1111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        shl eax, {shamt:#x}
                                        movsx rax, eax
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 6 {
                            0b000000 => {
                                // SRLIW
                                let shamt = dec.imm & 0b11_1111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        shr eax, {shamt:#x}
                                        movsx rax, eax
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b010000 => {
                                // SRAIW
                                let shamt = dec.imm & 0b11_1111;

                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&format!(
                                    "
                                        sar eax, {shamt:#x}
                                        movsx rax, eax
                                    ",
                                    shamt = shamt
                                ));
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            0b0111011 => {
                let dec = Rtype::from(inst);

                match dec.funct3 {
                    0b000 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //ADDW
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        add eax, ebx
                                        movsx rax, eax
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b0100000 => {
                                //SUBW
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rbx"));
                                code.push_str(
                                    "
                                        sub eax, ebx
                                        movsx rax, eax
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b001 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //SLLW
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        shl eax, cl
                                        movsx rax, eax
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.funct7 {
                            0b0000000 => {
                                //SRLW
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        shr eax, cl
                                        movsx rax, eax
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            0b0100000 => {
                                //SRAW
                                code.push_str(&read_reg!(dec.rs1, "rax"));
                                code.push_str(&read_reg!(dec.rs2, "rcx"));
                                code.push_str(
                                    "
                                        and rcx, 0x1f
                                        sar eax, cl
                                        movsx rax, eax
                                    ",
                                );
                                code.push_str(&write_reg!(dec.rd, "rax"));
                            }
                            _ => return Err(VmExit::InvalidInstruction),
                        }
                    }
                    _ => return Err(VmExit::InvalidInstruction),
                }
            }
            _ => return Err(VmExit::InvalidInstruction),
        }

        Ok((code, false))
    }
}
