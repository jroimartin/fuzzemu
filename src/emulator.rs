//! RISC-V emulator (little-endian rv64i only).

use std::convert::TryInto;
use std::fmt;
use std::fs;
use std::io;
use std::ops::Deref;
use std::path::Path;

use crate::elf::{self, Elf};
use crate::mmu::{self, Mmu, Perm, VirtAddr, PERM_EXEC};

/// Emulator error.
#[derive(Debug)]
pub enum Error {
    AddressMisaligned,
    InvalidInstruction,
    InvalidRegister,
    UnimplementedInstruction,
    EBreak,
    ECall,

    ElfError(elf::Error),
    MmuError(mmu::Error),

    IoError(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::AddressMisaligned => {
                write!(f, "address-missaligned exception")
            }
            Error::InvalidInstruction => write!(f, "invalid instruction"),
            Error::InvalidRegister => write!(f, "invalid register"),
            Error::UnimplementedInstruction => {
                write!(f, "unimplemented instruction")
            }
            Error::EBreak => write!(f, "EBREAK"),
            Error::ECall => write!(f, "ECALL"),
            Error::ElfError(err) => write!(f, "ELF error: {}", err),
            Error::MmuError(err) => write!(f, "MMU error: {}", err),
            Error::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl From<mmu::Error> for Error {
    fn from(error: mmu::Error) -> Error {
        Error::MmuError(error)
    }
}

impl From<elf::Error> for Error {
    fn from(error: elf::Error) -> Error {
        Error::ElfError(error)
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::IoError(error)
    }
}

/// CPU Registers.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Reg(pub u32);

impl Deref for Reg {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// CPU register by name. Note that `Reg` implements the trait `From<RegName>`,
/// which simplifies referencing Registers by name.
///
/// # Examples
///
/// All the following calls to `reg_set` are equivalent:
///
/// ```
/// use riscv_emu::emulator::{Emulator, Reg, RegName};
///
/// let mut emulator = Emulator::new(1024);
///
/// emulator.get_reg(RegName::Zero);
/// emulator.get_reg(Reg::from(RegName::Zero));
/// emulator.get_reg(Reg(RegName::Zero as u32));
/// emulator.get_reg(Reg(0));
/// ```
///
/// For obvious reasons, the first options is recommended.
pub enum RegName {
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

impl From<RegName> for Reg {
    fn from(name: RegName) -> Reg {
        Reg(name as u32)
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

/// RISC-V emulator. This implementation only supports the RV64i Base Integer
/// Instruction Set and assumes little-endian.
pub struct Emulator {
    regs: [u64; 33],
    mmu: Mmu,
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
    /// Returns a new emulator with a memory of size `mem_size`.
    pub fn new(mem_size: usize) -> Emulator {
        Emulator {
            regs: [0; 33],
            mmu: Mmu::new(mem_size),
        }
    }

    /// Loads an ELF program in the emulator. It also points the program
    /// counter to the entrypoint of the program.
    pub fn load_program<P: AsRef<Path>>(
        &mut self,
        program: P,
    ) -> Result<(), Error> {
        let contents = fs::read(program)?;
        let elf = Elf::parse(&contents)?;

        for phdr in elf.phdrs() {
            let segment_bytes = contents
                .get(phdr.offset()..phdr.offset() + phdr.file_size())
                .ok_or(elf::Error::MalformedFile)?;

            self.mmu.poke(phdr.virt_addr(), segment_bytes)?;
            self.mmu.set_perms(
                phdr.virt_addr(),
                phdr.mem_size(),
                phdr.perms(),
            )?;
        }

        self.set_reg(RegName::Pc, *elf.entry() as u64)?;

        Ok(())
    }

    /// Returns a copy of the Emulator, including its internal state.
    pub fn fork(&self) -> Emulator {
        Emulator {
            regs: self.regs,
            mmu: self.mmu.fork(),
        }
    }

    /// Resets the internal state of the emulator to the given state `other`.
    pub fn reset(&mut self, other: &Emulator) {
        self.regs = other.regs;
        self.mmu.reset(&other.mmu);
    }

    /// Sets the value of the register `reg` to `val`.
    pub fn set_reg<T: Into<Reg>>(
        &mut self,
        reg: T,
        val: u64,
    ) -> Result<(), Error> {
        let reg = *reg.into() as usize;

        if reg >= self.regs.len() {
            return Err(Error::InvalidRegister);
        }

        // zero register is always 0.
        if reg != 0 {
            self.regs[reg] = val;
        }
        Ok(())
    }

    /// Returns the value stored in the register `reg`.
    pub fn get_reg<T: Into<Reg>>(&self, reg: T) -> Result<u64, Error> {
        let reg = *reg.into() as usize;

        if reg >= self.regs.len() {
            return Err(Error::InvalidRegister);
        }

        // zero register is always 0.
        if reg == 0 {
            Ok(0)
        } else {
            Ok(self.regs[reg])
        }
    }

    /// Emulates until vm exit or error.
    pub fn run(&mut self) -> Result<(), Error> {
        loop {
            let pc = self.get_reg(RegName::Pc)?;
            let bytes = self.mmu.read_with_perms(
                VirtAddr(pc as usize),
                4,
                Perm(PERM_EXEC),
            )?;
            let inst = u32::from_le_bytes(bytes.try_into().unwrap());

            self.run_instruction(inst)?;
        }
    }

    /// Emulates a single instruction, updating the internal state of the
    /// emulator.
    fn run_instruction(&mut self, inst: u32) -> Result<(), Error> {
        let opcode = inst & 0b111_1111;

        let pc = self.get_reg(RegName::Pc)?;

        if pc & 3 != 0 {
            return Err(Error::AddressMisaligned);
        }

        eprintln!("---");
        eprintln!("{}", self);
        eprintln!("{:#010x}: {:08x} {:07b}", pc, inst, opcode);

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
                self.set_reg(RegName::Pc, pc.wrapping_add(offset))?;
                return Ok(());
            }
            0b1100111 => {
                let dec = Itype::from(inst);

                let offset = dec.imm as u64;
                let rs1 = self.get_reg(dec.rs1)?;

                match dec.funct3 {
                    0b000 => {
                        // JALR
                        self.set_reg(dec.rd, pc.wrapping_add(4))?;
                        self.set_reg(
                            RegName::Pc,
                            rs1.wrapping_add(offset) >> 1 << 1,
                        )?;
                        return Ok(());
                    }
                    _ => return Err(Error::UnimplementedInstruction),
                }
            }
            0b1100011 => {
                let dec = Btype::from(inst);

                let offset = dec.imm as u64;
                let rs1 = self.get_reg(dec.rs1)?;
                let rs2 = self.get_reg(dec.rs2)?;

                match dec.funct3 {
                    0b000 => {
                        // BEQ
                        if rs1 == rs2 {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b001 => {
                        // BNE
                        if rs1 != rs2 {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b100 => {
                        // BLT
                        if (rs1 as i64) < (rs2 as i64) {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b101 => {
                        // BGE
                        if (rs1 as i64) >= (rs2 as i64) {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b110 => {
                        // BLTU
                        if rs1 < rs2 {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    0b111 => {
                        // BGEU
                        if rs1 >= rs2 {
                            self.set_reg(
                                RegName::Pc,
                                pc.wrapping_add(offset),
                            )?;
                            return Ok(());
                        }
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0000011 => {
                let dec = Itype::from(inst);

                let rs1 = self.get_reg(dec.rs1)?;
                let offset = dec.imm as u64;
                let vaddr = rs1.wrapping_add(offset);

                let vaddr = VirtAddr(vaddr as usize);
                let bytes = self.mmu.read(vaddr, 8)?;
                let value = u64::from_le_bytes(bytes.try_into().unwrap());

                match dec.funct3 {
                    0b000 => {
                        // LB
                        self.set_reg(dec.rd, value as i8 as u64)?;
                    }
                    0b001 => {
                        // LH
                        self.set_reg(dec.rd, value as i16 as u64)?;
                    }
                    0b010 => {
                        // LW
                        self.set_reg(dec.rd, value as i32 as u64)?;
                    }
                    0b100 => {
                        // LBU
                        self.set_reg(dec.rd, value as u8 as u64)?;
                    }
                    0b101 => {
                        // LHU
                        self.set_reg(dec.rd, value as u16 as u64)?;
                    }
                    0b110 => {
                        // LWU
                        self.set_reg(dec.rd, value as u32 as u64)?;
                    }
                    0b011 => {
                        // LD
                        self.set_reg(dec.rd, value)?;
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0100011 => {
                let dec = Stype::from(inst);

                let rs1 = self.get_reg(dec.rs1)?;
                let rs2 = self.get_reg(dec.rs2)?;
                let offset = dec.imm as u64;
                let vaddr = rs1.wrapping_add(offset);

                let vaddr = VirtAddr(vaddr as usize);

                match dec.funct3 {
                    0b000 => {
                        //SB
                        let value = (rs2 as u8).to_le_bytes();
                        self.mmu.write(vaddr, &value[..])?;
                    }
                    0b001 => {
                        //SH
                        let value = (rs2 as u16).to_le_bytes();
                        self.mmu.write(vaddr, &value[..])?;
                    }
                    0b010 => {
                        //SW
                        let value = (rs2 as u32).to_le_bytes();
                        self.mmu.write(vaddr, &value[..])?;
                    }
                    0b011 => {
                        //SD
                        let value = rs2.to_le_bytes();
                        self.mmu.write(vaddr, &value[..])?;
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0010011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u64;
                let rs1 = self.get_reg(dec.rs1)?;

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
                        match dec.imm as u32 >> 5 {
                            0b0000000 => {
                                // SLLI
                                let shamt = dec.imm & 0b1_1111;
                                self.set_reg(dec.rd, rs1 << shamt)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 5 {
                            0b0000000 => {
                                // SRLI
                                let shamt = dec.imm & 0b1_1111;
                                self.set_reg(dec.rd, rs1 >> shamt)?;
                            }
                            0b0100000 => {
                                // SRAI
                                let shamt = dec.imm & 0b1_1111;
                                let value = ((rs1 as i64) >> shamt) as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0110011 => {
                let dec = Rtype::from(inst);

                let rs1 = self.get_reg(dec.rs1)?;
                let rs2 = self.get_reg(dec.rs2)?;

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
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b001 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // SLL
                                let shamt = rs2 & 0b1_1111;
                                self.set_reg(dec.rd, rs1 << shamt)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
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
                            _ => return Err(Error::InvalidInstruction),
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
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b100 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // XOR
                                self.set_reg(dec.rd, rs1 ^ rs2)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
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
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b110 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // OR
                                self.set_reg(dec.rd, rs1 | rs2)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b111 => {
                        match dec.funct7 {
                            0b0000000 => {
                                // AND
                                self.set_reg(dec.rd, rs1 & rs2)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0001111 => {
                // FENCE
                return Err(Error::UnimplementedInstruction);
            }
            0b1110011 => {
                let dec = Itype::from(inst);

                if dec.rd == Reg(0) && dec.funct3 == 0 && dec.rs1 == Reg(0) {
                    if dec.imm == 0 {
                        // ECALL
                        return Err(Error::ECall);
                    } else if dec.imm == 1 {
                        // EBREAK
                        return Err(Error::EBreak);
                    } else {
                        return Err(Error::InvalidInstruction);
                    }
                } else {
                    return Err(Error::InvalidInstruction);
                }
            }
            0b0011011 => {
                let dec = Itype::from(inst);

                let imm = dec.imm as u32;
                let rs1 = self.get_reg(dec.rs1)? as u32;

                match dec.funct3 {
                    0b000 => {
                        // ADDIW
                        let value = rs1.wrapping_add(imm) as i32 as u64;
                        self.set_reg(dec.rd, value)?;
                    }
                    0b001 => {
                        match dec.imm as u32 >> 5 {
                            0b0000000 => {
                                // SLLIW
                                let shamt = dec.imm & 0b1_1111;
                                let value = (rs1 << shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    0b101 => {
                        match dec.imm as u32 >> 5 {
                            0b0000000 => {
                                // SRLIW
                                let shamt = dec.imm & 0b1_1111;
                                let value = (rs1 >> shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            0b0100000 => {
                                // SRAIW
                                let shamt = dec.imm & 0b1_1111;
                                let value =
                                    ((rs1 as i32) >> shamt) as i32 as u64;
                                self.set_reg(dec.rd, value)?;
                            }
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            0b0111011 => {
                let dec = Rtype::from(inst);

                let rs1 = self.get_reg(dec.rs1)? as u32;
                let rs2 = self.get_reg(dec.rs2)? as u32;

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
                            _ => return Err(Error::InvalidInstruction),
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
                            _ => return Err(Error::InvalidInstruction),
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
                            _ => return Err(Error::InvalidInstruction),
                        }
                    }
                    _ => return Err(Error::InvalidInstruction),
                }
            }
            _ => return Err(Error::InvalidInstruction),
        }

        self.set_reg(RegName::Pc, pc.wrapping_add(4))?;

        Ok(())
    }
}
