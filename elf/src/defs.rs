//! Standard ELF types.

use std::convert::TryFrom;
use std::fmt;

use crate::Error;

/// ELF class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// Invalid class.
    None,

    /// 32-bit architecture.
    Elf32,

    /// 64-bit architecture.
    Elf64,
}

impl TryFrom<u8> for Class {
    type Error = Error;

    fn try_from(value: u8) -> Result<Class, Self::Error> {
        match value {
            0 => Ok(Class::None),
            1 => Ok(Class::Elf32),
            2 => Ok(Class::Elf64),
            _ => Err(Error::ParseError),
        }
    }
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Class::None => write!(f, "None"),
            Class::Elf32 => write!(f, "32-bit"),
            Class::Elf64 => write!(f, "64-bit"),
        }
    }
}

/// ELF data enconding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataEncoding {
    /// Invalid data encoding.
    None,

    /// 2's complement, little endian.
    LittleEndian,

    /// 2's complement, big endian.
    BigEndian,
}

impl TryFrom<u8> for DataEncoding {
    type Error = Error;

    fn try_from(value: u8) -> Result<DataEncoding, Self::Error> {
        match value {
            0 => Ok(DataEncoding::None),
            1 => Ok(DataEncoding::LittleEndian),
            2 => Ok(DataEncoding::BigEndian),
            _ => Err(Error::ParseError),
        }
    }
}

impl fmt::Display for DataEncoding {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DataEncoding::None => write!(f, "None"),
            DataEncoding::LittleEndian => {
                write!(f, "2's complement, little endian")
            }
            DataEncoding::BigEndian => write!(f, "2's complement, big endian"),
        }
    }
}

/// ELF version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Version {
    /// Invalid ELF version.
    None,

    /// Current version.
    Current,
}

impl TryFrom<u8> for Version {
    type Error = Error;

    fn try_from(value: u8) -> Result<Version, Self::Error> {
        match value {
            0 => Ok(Version::None),
            1 => Ok(Version::Current),
            _ => Err(Error::ParseError),
        }
    }
}

impl TryFrom<u32> for Version {
    type Error = Error;

    fn try_from(value: u32) -> Result<Version, Self::Error> {
        let value = u8::try_from(value).or(Err(Error::ParseError))?;
        Version::try_from(value)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Version::None => write!(f, "None"),
            Version::Current => write!(f, "Current"),
        }
    }
}

/// ELF OS ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsAbi {
    /// UNIX System V ABI.
    SysV,

    /// HP-UX.
    HpUx,

    /// NetBSD.
    NetBsd,

    /// Linux (object uses GNU ELF extensions).
    Linux,

    /// Sun Solaris.
    Solaris,

    /// IBM AIX.
    Aix,

    /// SGI Irix.
    Irix,

    /// FreeBSD.
    FreeBsd,

    /// Compaq TRU64 UNIX.
    Tru64,

    /// Novell Modesto.
    Modesto,

    /// OpenBSD.
    OpenBsd,

    /// ARM EABI.
    ArmEabi,

    /// ARM.
    Arm,

    /// Standalone (embedded) application.
    Standalone,

    /// Unknown.
    Unknown(u8),
}

impl From<u8> for OsAbi {
    fn from(value: u8) -> OsAbi {
        match value {
            0 => OsAbi::SysV,
            1 => OsAbi::HpUx,
            2 => OsAbi::NetBsd,
            3 => OsAbi::Linux,
            6 => OsAbi::Solaris,
            7 => OsAbi::Aix,
            8 => OsAbi::Irix,
            9 => OsAbi::FreeBsd,
            10 => OsAbi::Tru64,
            11 => OsAbi::Modesto,
            12 => OsAbi::OpenBsd,
            64 => OsAbi::ArmEabi,
            97 => OsAbi::Arm,
            255 => OsAbi::Standalone,
            val => OsAbi::Unknown(val),
        }
    }
}

impl fmt::Display for OsAbi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OsAbi::SysV => write!(f, "UNIX System V"),
            OsAbi::HpUx => write!(f, "HP-UX"),
            OsAbi::NetBsd => write!(f, "NetBSD"),
            OsAbi::Linux => write!(f, "Linux"),
            OsAbi::Solaris => write!(f, "Solaris"),
            OsAbi::Aix => write!(f, "AIX"),
            OsAbi::Irix => write!(f, "IRIX"),
            OsAbi::FreeBsd => write!(f, "FreeBSD"),
            OsAbi::Tru64 => write!(f, "TRU64"),
            OsAbi::Modesto => write!(f, "Modesto"),
            OsAbi::OpenBsd => write!(f, "OpenBSD"),
            OsAbi::ArmEabi => write!(f, "ARM EABI"),
            OsAbi::Arm => write!(f, "ARM"),
            OsAbi::Standalone => write!(f, "Embedded"),
            OsAbi::Unknown(val) => write!(f, "Unknown ({})", val),
        }
    }
}

/// Object file type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileType {
    /// No file type.
    None,

    /// Relocatable file.
    Rel,

    /// Executable file.
    Exec,

    /// Shared object file.
    Dyn,

    /// Core file.
    Core,

    /// Unknown.
    Unknown(u16),
}

impl From<u16> for FileType {
    fn from(value: u16) -> FileType {
        match value {
            0 => FileType::None,
            1 => FileType::Rel,
            2 => FileType::Exec,
            3 => FileType::Dyn,
            4 => FileType::Core,
            val => FileType::Unknown(val),
        }
    }
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            FileType::None => write!(f, "No file type"),
            FileType::Rel => write!(f, "Relocatable file"),
            FileType::Exec => write!(f, "Executable file"),
            FileType::Dyn => write!(f, "Shared object file"),
            FileType::Core => write!(f, "Core file"),
            FileType::Unknown(val) => write!(f, "Unknown ({})", val),
        }
    }
}

/// Specifies the required architecture for an individual file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Machine {
    /// No machine.
    None,

    /// AT&T WE 32100.
    M32,

    /// SUN SPARC.
    Sparc,

    /// Intel 80386.
    I386,

    /// Motorola m68k family.
    M68k,

    /// Motorola m88k family.
    M88k,

    /// Intel MCU.
    IaMcu,

    /// Intel 80860.
    I860,

    /// MIPS R3000 big-endian.
    Mips,

    /// IBM System/370.
    S370,

    /// MIPS R3000 little-endian.
    MipsRs3Le,

    /// HPPA.
    PaRisc,

    /// Fujitsu VPP500.
    Vpp500,

    /// Sun's "v8plus".
    Sparc32Plus,

    /// Intel 80960.
    I960,

    /// PowerPC.
    Ppc,

    /// PowerPC 64-bit.
    Ppc64,

    /// IBM S390.
    S390,

    /// IBM SPU/SPC.
    Spu,

    /// NEC V800 series.
    V800,

    /// Fujitsu FR20.
    Fr20,

    /// TRW RH-32.
    Rh32,

    /// Motorola RCE.
    Rce,

    /// ARM.
    Arm,

    /// Digital Alpha.
    FakeAlpha,

    /// Hitachi SH.
    Sh,

    /// SPARC v9 64-bit.
    SparcV9,

    /// Siemens Tricore.
    Tricore,

    /// Argonaut RISC Core.
    Arc,

    /// Hitachi H8/300.
    H8300,

    /// Hitachi H8/300H.
    H8300h,

    /// Hitachi H8S.
    H8s,

    /// Hitachi H8/500.
    H8500,

    /// Intel Merced.
    Ia64,

    /// Stanford MIPS-X.
    MipsX,

    /// Motorola Coldfire.
    Coldfire,

    /// Motorola M68HC12.
    M68hc12,

    /// Fujitsu MMA Multimedia Accelerator.
    Mma,

    /// Siemens PCP.
    Pcp,

    /// Sony nCPU embeeded RISC.
    NCpu,

    /// Denso NDR1 microprocessor.
    Ndr1,

    /// Motorola Start*Core processor.
    StarCore,

    /// Toyota ME16 processor.
    Me16,

    /// STMicroelectronic ST100 processor.
    St100,

    /// Advanced Logic Corp. Tinyj emb.fam.
    Tinyj,

    /// AMD x86-64 architecture.
    Amd64,

    /// Sony DSP Processor.
    Pdsp,

    /// Digital PDP-10.
    Pdp10,

    /// Digital PDP-11.
    Pdp11,

    /// Siemens FX66 microcontroller.
    Fx66,

    /// STMicroelectronics ST9+ 8/16 mc.
    St9Plus,

    /// STmicroelectronics ST7 8 bit mc.
    St7,

    /// Motorola MC68HC16 microcontroller.
    M68hc16,

    /// Motorola MC68HC11 microcontroller.
    M68hc11,

    /// Motorola MC68HC08 microcontroller.
    M68hc08,

    /// Motorola MC68HC05 microcontroller.
    M68hc05,

    /// Silicon Graphics SVx.
    Svx,

    /// STMicroelectronics ST19 8 bit mc.
    St19,

    /// Digital VAX.
    Vax,

    /// Axis Communications 32-bit emb.proc.
    Cris,

    /// Infineon Technologies 32-bit emb.proc.
    Javelin,

    /// Element 14 64-bit DSP Processor.
    Firepath,

    /// LSI Logic 16-bit DSP Processor.
    Zsp,

    /// Donald Knuth's educational 64-bit proc.
    Mmix,

    /// Harvard University machine-independent object files.
    Huany,

    /// SiTera Prism.
    Prism,

    /// Atmel AVR 8-bit microcontroller.
    Avr,

    /// Fujitsu FR30.
    Fr30,

    /// Mitsubishi D10V.
    D10v,

    /// Mitsubishi D30V.
    D30v,

    /// NEC v850.
    V850,

    /// Mitsubishi M32R.
    M32r,

    /// Matsushita MN10300.
    Mn10300,

    /// Matsushita MN10200.
    Mn10200,

    /// picoJava.
    Pj,

    /// OpenRISC 32-bit embedded processor.
    OpenRisc,

    /// ARC International ARCompact.
    ArcCompact,

    /// Tensilica Xtensa Architecture.
    Xtensa,

    /// Alphamosaic VideoCore.
    VideoCore,

    /// Thompson Multimedia General Purpose Proc.
    TmmGpp,

    /// National Semi. 32000.
    Ns32k,

    /// Tenor Network TPC.
    Tpc,

    /// Trebia SNP 1000.
    Snp1k,

    /// STMicroelectronics ST200.
    St200,

    /// Ubicom IP2xxx.
    Ip2k,

    /// MAX processor.
    Max,

    /// National Semi. CompactRISC.
    Cr,

    /// Fujitsu F2MC16.
    F2mc16,

    /// Texas Instruments msp430.
    Msp430,

    /// Analog Devices Blackfin DSP.
    Blackfin,

    /// Seiko Epson S1C33 family.
    SeC33,

    /// Sharp embedded microprocessor.
    Sep,

    /// Arca RISC.
    Arca,

    /// PKU-Unity & MPRC Peking Uni. mc series.
    Unicore,

    /// eXcess configurable cpu.
    Excess,

    /// Icera Semi. Deep Execution Processor.
    Dxp,

    /// Altera Nios II.
    AlteraNios2,

    /// National Semi. CompactRISC CRX.
    Crx,

    /// Motorola XGATE.
    Xgate,

    /// Infineon C16x/XC16x.
    C166,

    /// Renesas M16C.
    M16c,

    /// Microchip Technology dsPIC30F.
    DsPic30f,

    /// Freescale Communication Engine RISC.
    Ce,

    /// Renesas M32C.
    M32c,

    /// Altium TSK3000.
    Tsk3000,

    /// Freescale RS08.
    Rs08,

    /// Analog Devices SHARC family.
    Sharc,

    /// Cyan Technology eCOG2.
    Ecog2,

    /// Sunplus S+core7 RISC.
    SCore7,

    /// New Japan Radio (NJR) 24-bit DSP.
    Dsp24,

    /// Broadcom VideoCore III.
    VideoCore3,

    /// RISC for Lattice FPGA.
    LatticeMico32,

    /// Seiko Epson C17.
    SeC17,

    /// Texas Instruments TMS320C6000 DSP.
    TiC6000,

    /// Texas Instruments TMS320C2000 DSP.
    TiC2000,

    /// Texas Instruments TMS320C55x DSP.
    TiC5500,

    /// Texas Instruments App. Specific RISC.
    TiARP32,

    /// Texas Instruments Prog. Realtime Unit.
    TiPru,

    /// STMicroelectronics 64bit VLIW DSP.
    MmdspPlus,

    /// Cypress M8C.
    CypressM8c,

    /// Renesas R32C.
    R32c,

    /// NXP Semi. TriMedia.
    TriMedia,

    /// QUALCOMM DSP6.
    QDsp6,

    /// Intel 8051 and variants.
    I8051,

    /// STMicroelectronics STxP7x.
    StxP7x,

    /// Andes Tech. compact code emb. RISC.
    Nds32,

    /// Cyan Technology eCOG1X.
    Ecog1x,

    /// Dallas Semi. MAXQ30 mc.
    Maxq30,

    /// New Japan Radio (NJR) 16-bit DSP.
    Ximo16,

    /// M2000 Reconfigurable RISC.
    Manik,

    /// Cray NV2 vector architecture.
    CrayNv2,

    /// Renesas RX.
    Rx,

    /// Imagination Tech. META.
    Metag,

    /// MCST Elbrus.
    McstElbrus,

    /// Cyan Technology eCOG16.
    Ecog16,

    /// National Semi. CompactRISC CR16.
    Cr16,

    /// Freescale Extended Time Processing Unit.
    Etpu,

    /// Infineon Tech. SLE9X.
    Sle9x,

    /// Intel L10M.
    L10m,

    /// Intel K10M.
    K10m,

    /// ARM AARCH64.
    AArch64,

    /// Amtel 32-bit microprocessor.
    Avr32,

    /// STMicroelectronics STM8.
    StM8,

    /// Tileta TILE64.
    Tile64,

    /// Tilera TILEPro.
    TilePro,

    /// Xilinx MicroBlaze.
    MicroBlaze,

    /// NVIDIA CUDA.
    Cuda,

    /// Tilera TILE-Gx.
    TileGx,

    /// CloudShield.
    CloudShield,

    /// KIPO-KAIST Core-A 1st gen.
    Corea1st,

    /// KIPO-KAIST Core-A 2nd gen.
    Corea2nd,

    /// Synopsys ARCompact V2.
    ArcCompact2,

    /// Open8 RISC.
    Open8,

    /// Renesas RL78.
    Rl78,

    /// Broadcom VideoCore V.
    VideoCore5,

    /// Renesas 78KOR.
    Renesas78kor,

    /// Freescale 56800EX DSC.
    Freescale56800ex,

    /// Beyond BA1.
    Ba1,

    /// Beyond BA2.
    Ba2,

    /// XMOS xCORE.
    XCore,

    /// Microchip 8-bit PIC(r).
    MchpPic,

    /// KM211 KM32.
    Km32,

    /// KM211 KMX32.
    Kmx32,

    /// KM211 KMX16.
    Emx16,

    /// KM211 KMX8.
    Emx8,

    /// KM211 KVARC.
    Kvarc,

    /// Paneve CDP.
    Cdp,

    /// Cognitive Smart Memory Processor.
    Coge,

    /// Bluechip CoolEngine.
    Cool,

    /// Nanoradio Optimized RISC.
    Norc,

    /// CSR Kalimba.
    CsrKalimba,

    /// Zilog Z80.
    Z80,

    /// Controls and Data Services VISIUMcore.
    Visium,

    /// FTDI Chip FT32.
    Ft32,

    /// Moxie processor.
    Moxie,

    /// AMD GPU.
    AmdGpu,

    /// RISC-V.
    RiscV,

    /// Linux BPF -- in-kernel virtual machine.
    Bpf,

    /// C-SKY.
    CSky,

    /// Unknown.
    Unknown(u16),
}

impl From<u16> for Machine {
    fn from(value: u16) -> Machine {
        match value {
            0 => Machine::None,
            1 => Machine::M32,
            2 => Machine::Sparc,
            3 => Machine::I386,
            4 => Machine::M68k,
            5 => Machine::M88k,
            6 => Machine::IaMcu,
            7 => Machine::I860,
            8 => Machine::Mips,
            9 => Machine::S370,
            10 => Machine::MipsRs3Le,
            15 => Machine::PaRisc,
            17 => Machine::Vpp500,
            18 => Machine::Sparc32Plus,
            19 => Machine::I960,
            20 => Machine::Ppc,
            21 => Machine::Ppc64,
            22 => Machine::S390,
            23 => Machine::Spu,
            36 => Machine::V800,
            37 => Machine::Fr20,
            38 => Machine::Rh32,
            39 => Machine::Rce,
            40 => Machine::Arm,
            41 => Machine::FakeAlpha,
            42 => Machine::Sh,
            43 => Machine::SparcV9,
            44 => Machine::Tricore,
            45 => Machine::Arc,
            46 => Machine::H8300,
            47 => Machine::H8300h,
            48 => Machine::H8s,
            49 => Machine::H8500,
            50 => Machine::Ia64,
            51 => Machine::MipsX,
            52 => Machine::Coldfire,
            53 => Machine::M68hc12,
            54 => Machine::Mma,
            55 => Machine::Pcp,
            56 => Machine::NCpu,
            57 => Machine::Ndr1,
            58 => Machine::StarCore,
            59 => Machine::Me16,
            60 => Machine::St100,
            61 => Machine::Tinyj,
            62 => Machine::Amd64,
            63 => Machine::Pdsp,
            64 => Machine::Pdp10,
            65 => Machine::Pdp11,
            66 => Machine::Fx66,
            67 => Machine::St9Plus,
            68 => Machine::St7,
            69 => Machine::M68hc16,
            70 => Machine::M68hc11,
            71 => Machine::M68hc08,
            72 => Machine::M68hc05,
            73 => Machine::Svx,
            74 => Machine::St19,
            75 => Machine::Vax,
            76 => Machine::Cris,
            77 => Machine::Javelin,
            78 => Machine::Firepath,
            79 => Machine::Zsp,
            80 => Machine::Mmix,
            81 => Machine::Huany,
            82 => Machine::Prism,
            83 => Machine::Avr,
            84 => Machine::Fr30,
            85 => Machine::D10v,
            86 => Machine::D30v,
            87 => Machine::V850,
            88 => Machine::M32r,
            89 => Machine::Mn10300,
            90 => Machine::Mn10200,
            91 => Machine::Pj,
            92 => Machine::OpenRisc,
            93 => Machine::ArcCompact,
            94 => Machine::Xtensa,
            95 => Machine::VideoCore,
            96 => Machine::TmmGpp,
            97 => Machine::Ns32k,
            98 => Machine::Tpc,
            99 => Machine::Snp1k,
            100 => Machine::St200,
            101 => Machine::Ip2k,
            102 => Machine::Max,
            103 => Machine::Cr,
            104 => Machine::F2mc16,
            105 => Machine::Msp430,
            106 => Machine::Blackfin,
            107 => Machine::SeC33,
            108 => Machine::Sep,
            109 => Machine::Arca,
            110 => Machine::Unicore,
            111 => Machine::Excess,
            112 => Machine::Dxp,
            113 => Machine::AlteraNios2,
            114 => Machine::Crx,
            115 => Machine::Xgate,
            116 => Machine::C166,
            117 => Machine::M16c,
            118 => Machine::DsPic30f,
            119 => Machine::Ce,
            120 => Machine::M32c,
            131 => Machine::Tsk3000,
            132 => Machine::Rs08,
            133 => Machine::Sharc,
            134 => Machine::Ecog2,
            135 => Machine::SCore7,
            136 => Machine::Dsp24,
            137 => Machine::VideoCore3,
            138 => Machine::LatticeMico32,
            139 => Machine::SeC17,
            140 => Machine::TiC6000,
            141 => Machine::TiC2000,
            142 => Machine::TiC5500,
            143 => Machine::TiARP32,
            144 => Machine::TiPru,
            160 => Machine::MmdspPlus,
            161 => Machine::CypressM8c,
            162 => Machine::R32c,
            163 => Machine::TriMedia,
            164 => Machine::QDsp6,
            165 => Machine::I8051,
            166 => Machine::StxP7x,
            167 => Machine::Nds32,
            168 => Machine::Ecog1x,
            169 => Machine::Maxq30,
            170 => Machine::Ximo16,
            171 => Machine::Manik,
            172 => Machine::CrayNv2,
            173 => Machine::Rx,
            174 => Machine::Metag,
            175 => Machine::McstElbrus,
            176 => Machine::Ecog16,
            177 => Machine::Cr16,
            178 => Machine::Etpu,
            179 => Machine::Sle9x,
            180 => Machine::L10m,
            181 => Machine::K10m,
            183 => Machine::AArch64,
            185 => Machine::Avr32,
            186 => Machine::StM8,
            187 => Machine::Tile64,
            188 => Machine::TilePro,
            189 => Machine::MicroBlaze,
            190 => Machine::Cuda,
            191 => Machine::TileGx,
            192 => Machine::CloudShield,
            193 => Machine::Corea1st,
            194 => Machine::Corea2nd,
            195 => Machine::ArcCompact2,
            196 => Machine::Open8,
            197 => Machine::Rl78,
            198 => Machine::VideoCore5,
            199 => Machine::Renesas78kor,
            200 => Machine::Freescale56800ex,
            201 => Machine::Ba1,
            202 => Machine::Ba2,
            203 => Machine::XCore,
            204 => Machine::MchpPic,
            210 => Machine::Km32,
            211 => Machine::Kmx32,
            212 => Machine::Emx16,
            213 => Machine::Emx8,
            214 => Machine::Kvarc,
            215 => Machine::Cdp,
            216 => Machine::Coge,
            217 => Machine::Cool,
            218 => Machine::Norc,
            219 => Machine::CsrKalimba,
            220 => Machine::Z80,
            221 => Machine::Visium,
            222 => Machine::Ft32,
            223 => Machine::Moxie,
            224 => Machine::AmdGpu,
            243 => Machine::RiscV,
            247 => Machine::Bpf,
            252 => Machine::CSky,
            val => Machine::Unknown(val),
        }
    }
}

impl fmt::Display for Machine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Machine::None => write!(f, "No machine"),
            Machine::M32 => write!(f, "AT&T WE 32100"),
            Machine::Sparc => write!(f, "SUN SPARC"),
            Machine::I386 => write!(f, "Intel 80386"),
            Machine::M68k => write!(f, "Motorola m68k family"),
            Machine::M88k => write!(f, "Motorola m88k family"),
            Machine::IaMcu => write!(f, "Intel MCU"),
            Machine::I860 => write!(f, "Intel 80860"),
            Machine::Mips => write!(f, "MIPS R3000 big-endian"),
            Machine::S370 => write!(f, "IBM System/370"),
            Machine::MipsRs3Le => write!(f, "MIPS R3000 little-endian"),
            Machine::PaRisc => write!(f, "HPPA"),
            Machine::Vpp500 => write!(f, "Fujitsu VPP500"),
            Machine::Sparc32Plus => write!(f, "Sun's v8plus"),
            Machine::I960 => write!(f, "Intel 80960"),
            Machine::Ppc => write!(f, "PowerPC"),
            Machine::Ppc64 => write!(f, "PowerPC 64-bit"),
            Machine::S390 => write!(f, "IBM S390"),
            Machine::Spu => write!(f, "IBM SPU/SPC"),
            Machine::V800 => write!(f, "NEC V800 series"),
            Machine::Fr20 => write!(f, "Fujitsu FR20"),
            Machine::Rh32 => write!(f, "TRW RH-32"),
            Machine::Rce => write!(f, "Motorola RCE"),
            Machine::Arm => write!(f, "ARM"),
            Machine::FakeAlpha => write!(f, "Digital Alpha"),
            Machine::Sh => write!(f, "Hitachi SH"),
            Machine::SparcV9 => write!(f, "SPARC v9 64-bit"),
            Machine::Tricore => write!(f, "Siemens Tricore"),
            Machine::Arc => write!(f, "Argonaut RISC Core"),
            Machine::H8300 => write!(f, "Hitachi H8/300"),
            Machine::H8300h => write!(f, "Hitachi H8/300H"),
            Machine::H8s => write!(f, "Hitachi H8S"),
            Machine::H8500 => write!(f, "Hitachi H8/500"),
            Machine::Ia64 => write!(f, "Intel Merced"),
            Machine::MipsX => write!(f, "Stanford MIPS-X"),
            Machine::Coldfire => write!(f, "Motorola Coldfire"),
            Machine::M68hc12 => write!(f, "Motorola M68HC12"),
            Machine::Mma => write!(f, "Fujitsu MMA Multimedia Accelerator"),
            Machine::Pcp => write!(f, "Siemens PCP"),
            Machine::NCpu => write!(f, "Sony nCPU embeeded RISC"),
            Machine::Ndr1 => write!(f, "Denso NDR1 microprocessor"),
            Machine::StarCore => write!(f, "Motorola Start*Core processor"),
            Machine::Me16 => write!(f, "Toyota ME16 processor"),
            Machine::St100 => write!(f, "STMicroelectronic ST100 processor"),
            Machine::Tinyj => write!(f, "Advanced Logic Corp. Tinyj emb.fam"),
            Machine::Amd64 => write!(f, "AMD x86-64 architecture"),
            Machine::Pdsp => write!(f, "Sony DSP Processor"),
            Machine::Pdp10 => write!(f, "Digital PDP-10"),
            Machine::Pdp11 => write!(f, "Digital PDP-11"),
            Machine::Fx66 => write!(f, "Siemens FX66 microcontroller"),
            Machine::St9Plus => write!(f, "STMicroelectronics ST9+ 8/16 mc"),
            Machine::St7 => write!(f, "STmicroelectronics ST7 8 bit mc"),
            Machine::M68hc16 => write!(f, "Motorola MC68HC16 microcontroller"),
            Machine::M68hc11 => write!(f, "Motorola MC68HC11 microcontroller"),
            Machine::M68hc08 => write!(f, "Motorola MC68HC08 microcontroller"),
            Machine::M68hc05 => write!(f, "Motorola MC68HC05 microcontroller"),
            Machine::Svx => write!(f, "Silicon Graphics SVx"),
            Machine::St19 => write!(f, "STMicroelectronics ST19 8 bit mc"),
            Machine::Vax => write!(f, "Digital VAX"),
            Machine::Cris => write!(f, "Axis Communications 32-bit emb.proc"),
            Machine::Javelin => {
                write!(f, "Infineon Technologies 32-bit emb.proc")
            }
            Machine::Firepath => write!(f, "Element 14 64-bit DSP Processor"),
            Machine::Zsp => write!(f, "LSI Logic 16-bit DSP Processor"),
            Machine::Mmix => {
                write!(f, "Donald Knuth's educational 64-bit proc")
            }
            Machine::Huany => write!(
                f,
                "Harvard University machine-independent object files"
            ),
            Machine::Prism => write!(f, "SiTera Prism"),
            Machine::Avr => write!(f, "Atmel AVR 8-bit microcontroller"),
            Machine::Fr30 => write!(f, "Fujitsu FR30"),
            Machine::D10v => write!(f, "Mitsubishi D10V"),
            Machine::D30v => write!(f, "Mitsubishi D30V"),
            Machine::V850 => write!(f, "NEC v850"),
            Machine::M32r => write!(f, "Mitsubishi M32R"),
            Machine::Mn10300 => write!(f, "Matsushita MN10300"),
            Machine::Mn10200 => write!(f, "Matsushita MN10200"),
            Machine::Pj => write!(f, "picoJava"),
            Machine::OpenRisc => {
                write!(f, "OpenRISC 32-bit embedded processor")
            }
            Machine::ArcCompact => write!(f, "ARC International ARCompact"),
            Machine::Xtensa => write!(f, "Tensilica Xtensa Architecture"),
            Machine::VideoCore => write!(f, "Alphamosaic VideoCore"),
            Machine::TmmGpp => {
                write!(f, "Thompson Multimedia General Purpose Proc")
            }
            Machine::Ns32k => write!(f, "National Semi. 32000"),
            Machine::Tpc => write!(f, "Tenor Network TPC"),
            Machine::Snp1k => write!(f, "Trebia SNP 1000"),
            Machine::St200 => write!(f, "STMicroelectronics ST200"),
            Machine::Ip2k => write!(f, "Ubicom IP2xxx"),
            Machine::Max => write!(f, "MAX processor"),
            Machine::Cr => write!(f, "National Semi. CompactRISC"),
            Machine::F2mc16 => write!(f, "Fujitsu F2MC16"),
            Machine::Msp430 => write!(f, "Texas Instruments msp430"),
            Machine::Blackfin => write!(f, "Analog Devices Blackfin DSP"),
            Machine::SeC33 => write!(f, "Seiko Epson S1C33 family"),
            Machine::Sep => write!(f, "Sharp embedded microprocessor"),
            Machine::Arca => write!(f, "Arca RISC"),
            Machine::Unicore => {
                write!(f, "PKU-Unity & MPRC Peking Uni. mc series")
            }
            Machine::Excess => write!(f, "eXcess configurable cpu"),
            Machine::Dxp => write!(f, "Icera Semi. Deep Execution Processor"),
            Machine::AlteraNios2 => write!(f, "Altera Nios II"),
            Machine::Crx => write!(f, "National Semi. CompactRISC CRX"),
            Machine::Xgate => write!(f, "Motorola XGATE"),
            Machine::C166 => write!(f, "Infineon C16x/XC16x"),
            Machine::M16c => write!(f, "Renesas M16C"),
            Machine::DsPic30f => write!(f, "Microchip Technology dsPIC30F"),
            Machine::Ce => write!(f, "Freescale Communication Engine RISC"),
            Machine::M32c => write!(f, "Renesas M32C"),
            Machine::Tsk3000 => write!(f, "Altium TSK3000"),
            Machine::Rs08 => write!(f, "Freescale RS08"),
            Machine::Sharc => write!(f, "Analog Devices SHARC family"),
            Machine::Ecog2 => write!(f, "Cyan Technology eCOG2"),
            Machine::SCore7 => write!(f, "Sunplus S+core7 RISC"),
            Machine::Dsp24 => write!(f, "New Japan Radio (NJR) 24-bit DSP"),
            Machine::VideoCore3 => write!(f, "Broadcom VideoCore III"),
            Machine::LatticeMico32 => write!(f, "RISC for Lattice FPGA"),
            Machine::SeC17 => write!(f, "Seiko Epson C17"),
            Machine::TiC6000 => write!(f, "Texas Instruments TMS320C6000 DSP"),
            Machine::TiC2000 => write!(f, "Texas Instruments TMS320C2000 DSP"),
            Machine::TiC5500 => write!(f, "Texas Instruments TMS320C55x DSP"),
            Machine::TiARP32 => {
                write!(f, "Texas Instruments App. Specific RISC")
            }
            Machine::TiPru => {
                write!(f, "Texas Instruments Prog. Realtime Unit")
            }
            Machine::MmdspPlus => {
                write!(f, "STMicroelectronics 64bit VLIW DSP")
            }
            Machine::CypressM8c => write!(f, "Cypress M8C"),
            Machine::R32c => write!(f, "Renesas R32C"),
            Machine::TriMedia => write!(f, "NXP Semi. TriMedia"),
            Machine::QDsp6 => write!(f, "QUALCOMM DSP6"),
            Machine::I8051 => write!(f, "Intel 8051 and variants"),
            Machine::StxP7x => write!(f, "STMicroelectronics STxP7x"),
            Machine::Nds32 => write!(f, "Andes Tech. compact code emb. RISC"),
            Machine::Ecog1x => write!(f, "Cyan Technology eCOG1X"),
            Machine::Maxq30 => write!(f, "Dallas Semi. MAXQ30 mc"),
            Machine::Ximo16 => write!(f, "New Japan Radio (NJR) 16-bit DSP"),
            Machine::Manik => write!(f, "M2000 Reconfigurable RISC"),
            Machine::CrayNv2 => write!(f, "Cray NV2 vector architecture"),
            Machine::Rx => write!(f, "Renesas RX"),
            Machine::Metag => write!(f, "Imagination Tech. META"),
            Machine::McstElbrus => write!(f, "MCST Elbrus"),
            Machine::Ecog16 => write!(f, "Cyan Technology eCOG16"),
            Machine::Cr16 => write!(f, "National Semi. CompactRISC CR16"),
            Machine::Etpu => {
                write!(f, "Freescale Extended Time Processing Unit")
            }
            Machine::Sle9x => write!(f, "Infineon Tech. SLE9X"),
            Machine::L10m => write!(f, "Intel L10M"),
            Machine::K10m => write!(f, "Intel K10M"),
            Machine::AArch64 => write!(f, "ARM AARCH64"),
            Machine::Avr32 => write!(f, "Amtel 32-bit microprocessor"),
            Machine::StM8 => write!(f, "STMicroelectronics STM8"),
            Machine::Tile64 => write!(f, "Tileta TILE64"),
            Machine::TilePro => write!(f, "Tilera TILEPro"),
            Machine::MicroBlaze => write!(f, "Xilinx MicroBlaze"),
            Machine::Cuda => write!(f, "NVIDIA CUDA"),
            Machine::TileGx => write!(f, "Tilera TILE-Gx"),
            Machine::CloudShield => write!(f, "CloudShield"),
            Machine::Corea1st => write!(f, "KIPO-KAIST Core-A 1st gen."),
            Machine::Corea2nd => write!(f, "KIPO-KAIST Core-A 2nd gen."),
            Machine::ArcCompact2 => write!(f, "Synopsys ARCompact V2"),
            Machine::Open8 => write!(f, "Open8 RISC"),
            Machine::Rl78 => write!(f, "Renesas RL78"),
            Machine::VideoCore5 => write!(f, "Broadcom VideoCore V"),
            Machine::Renesas78kor => write!(f, "Renesas 78KOR"),
            Machine::Freescale56800ex => write!(f, "Freescale 56800EX DSC"),
            Machine::Ba1 => write!(f, "Beyond BA1"),
            Machine::Ba2 => write!(f, "Beyond BA2"),
            Machine::XCore => write!(f, "XMOS xCORE"),
            Machine::MchpPic => write!(f, "Microchip 8-bit PIC(r)"),
            Machine::Km32 => write!(f, "KM211 KM32"),
            Machine::Kmx32 => write!(f, "KM211 KMX32"),
            Machine::Emx16 => write!(f, "KM211 KMX16"),
            Machine::Emx8 => write!(f, "KM211 KMX8"),
            Machine::Kvarc => write!(f, "KM211 KVARC"),
            Machine::Cdp => write!(f, "Paneve CDP"),
            Machine::Coge => write!(f, "Cognitive Smart Memory Processor"),
            Machine::Cool => write!(f, "Bluechip CoolEngine"),
            Machine::Norc => write!(f, "Nanoradio Optimized RISC"),
            Machine::CsrKalimba => write!(f, "CSR Kalimba"),
            Machine::Z80 => write!(f, "Zilog Z80"),
            Machine::Visium => {
                write!(f, "Controls and Data Services VISIUMcore")
            }
            Machine::Ft32 => write!(f, "FTDI Chip FT32"),
            Machine::Moxie => write!(f, "Moxie processor"),
            Machine::AmdGpu => write!(f, "AMD GPU"),
            Machine::RiscV => write!(f, "RISC-V"),
            Machine::Bpf => {
                write!(f, "Linux BPF -- in-kernel virtual machine")
            }
            Machine::CSky => write!(f, "C-SKY"),
            Machine::Unknown(val) => write!(f, "Unknown ({})", val),
        }
    }
}

/// Special value for `phnum`. This indicates that the real number of program
/// headers is too large to fit into `phnum`. Instead the real value is in the
/// field `sh_info` of section 0.
pub const PN_XNUM: u16 = 0xffff;

/// Start of reserved indices.
pub const SHN_LORESERVE: u16 = 0xff00;

/// Index is in extra table.
pub const SHN_XINDEX: u16 = 0xffff;

/// Undefined section.
pub const SHN_UNDEF: u16 = 0;

/// Segment type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentType {
    /// Program header table entry unused.
    Null,

    /// Loadable program segment.
    Load,

    /// Dynamic linking information.
    Dynamic,

    /// Program interpreter.
    Interp,

    /// Auxiliary information.
    Note,

    /// Reserved.
    Shlib,

    /// Entry for header table itself.
    Phdr,

    /// Thread-local storage segment.
    Tls,

    /// GCC .eh_frame_hdr segment.
    GnuEhFrame,

    /// Indicates stack executability.
    GnuStack,

    /// Read-only after relocation.
    GnuRelro,

    /// Sun Specific segment.
    SunwBss,

    /// Stack segment.
    SunwStack,

    /// Unknown.
    Unknown(u32),
}

impl From<u32> for SegmentType {
    fn from(value: u32) -> Self {
        match value {
            0 => SegmentType::Null,
            1 => SegmentType::Load,
            2 => SegmentType::Dynamic,
            3 => SegmentType::Interp,
            4 => SegmentType::Note,
            5 => SegmentType::Shlib,
            6 => SegmentType::Phdr,
            7 => SegmentType::Tls,
            0x6474e550 => SegmentType::GnuEhFrame,
            0x6474e551 => SegmentType::GnuStack,
            0x6474e552 => SegmentType::GnuRelro,
            0x6ffffffa => SegmentType::SunwBss,
            0x6ffffffb => SegmentType::SunwStack,
            val => SegmentType::Unknown(val),
        }
    }
}

impl fmt::Display for SegmentType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SegmentType::Null => {
                write!(f, "Program header table entry unused")
            }
            SegmentType::Load => write!(f, "Loadable program segment"),
            SegmentType::Dynamic => write!(f, "Dynamic linking information"),
            SegmentType::Interp => write!(f, "Program interpreter"),
            SegmentType::Note => write!(f, "Auxiliary information"),
            SegmentType::Shlib => write!(f, "Reserved"),
            SegmentType::Phdr => write!(f, "Entry for header table itself"),
            SegmentType::Tls => write!(f, "Thread-local storage segment"),
            SegmentType::GnuEhFrame => write!(f, "GCC .eh_frame_hdr segment"),
            SegmentType::GnuStack => {
                write!(f, "Indicates stack executability")
            }
            SegmentType::GnuRelro => write!(f, "Read-only after relocation"),
            SegmentType::SunwBss => write!(f, "Sun Specific segment"),
            SegmentType::SunwStack => write!(f, "Stack segment"),
            SegmentType::Unknown(val) => write!(f, "Unknown ({})", val),
        }
    }
}

/// Segment is executable.
pub const PF_X: u32 = 1 << 0;

/// Segment is writable.
pub const PF_W: u32 = 1 << 1;

/// Segment is readable.
pub const PF_R: u32 = 1 << 2;

/// Section type, which defines the section's contents and semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionType {
    /// Section header table entry unused.
    Null,

    /// Program data.
    ProgBits,

    /// Symbol table.
    SymTab,

    /// String table.
    StrTab,

    /// Relocation entries with addends.
    RelA,

    /// Symbol hash table.
    Hash,

    /// Dynamic linking information.
    Dynamic,

    /// Notes.
    Note,

    /// Program space with no data (bss).
    NoBits,

    /// Relocation entries, no addends.
    Rel,

    /// Reserved.
    ShLib,

    /// Dynamic linker symbol table.
    DynSym,

    /// Array of constructors.
    InitArray,

    /// Array of destructors.
    FiniArray,

    /// Array of pre-constructors.
    PreinitArray,

    /// Section group.
    Group,

    /// Extended section indices.
    SymTabShNdx,

    /// Object attributes.
    GnuAttributes,

    /// GNU-style hash table.
    GnuHash,

    /// Prelink library list.
    GnuLibList,

    /// Checksum for DSO content.
    Checksum,

    /// SUN Move.
    SunwMove,

    /// SUN COMDAT.
    SunwComDat,

    /// SUN SymInfo.
    SunwSymInfo,

    /// Version definition section.
    GnuVerDef,

    /// Version needs section.
    GnuVerNeed,

    /// Version symbol table.
    GnuVerSym,

    /// RISC-V attributes.
    RiscVAttributes,

    /// Unknown.
    Unknown(u32),
}

impl From<u32> for SectionType {
    fn from(value: u32) -> Self {
        match value {
            0 => SectionType::Null,
            1 => SectionType::ProgBits,
            2 => SectionType::SymTab,
            3 => SectionType::StrTab,
            4 => SectionType::RelA,
            5 => SectionType::Hash,
            6 => SectionType::Dynamic,
            7 => SectionType::Note,
            8 => SectionType::NoBits,
            9 => SectionType::Rel,
            10 => SectionType::ShLib,
            11 => SectionType::DynSym,
            14 => SectionType::InitArray,
            15 => SectionType::FiniArray,
            16 => SectionType::PreinitArray,
            17 => SectionType::Group,
            18 => SectionType::SymTabShNdx,
            0x6ffffff5 => SectionType::GnuAttributes,
            0x6ffffff6 => SectionType::GnuHash,
            0x6ffffff7 => SectionType::GnuLibList,
            0x6ffffff8 => SectionType::Checksum,
            0x6ffffffa => SectionType::SunwMove,
            0x6ffffffb => SectionType::SunwComDat,
            0x6ffffffc => SectionType::SunwSymInfo,
            0x6ffffffd => SectionType::GnuVerDef,
            0x6ffffffe => SectionType::GnuVerNeed,
            0x6fffffff => SectionType::GnuVerSym,
            0x70000003 => SectionType::RiscVAttributes,
            val => SectionType::Unknown(val),
        }
    }
}

impl fmt::Display for SectionType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SectionType::Null => {
                write!(f, "Section header table entry unused")
            }
            SectionType::ProgBits => write!(f, "Program data"),
            SectionType::SymTab => write!(f, "Symbol table"),
            SectionType::StrTab => write!(f, "String table"),
            SectionType::RelA => write!(f, "Relocation entries with addends"),
            SectionType::Hash => write!(f, "Symbol hash table"),
            SectionType::Dynamic => write!(f, "Dynamic linking information"),
            SectionType::Note => write!(f, "Notes"),
            SectionType::NoBits => {
                write!(f, "Program space with no data (bss)")
            }
            SectionType::Rel => write!(f, "Relocation entries, no addends"),
            SectionType::ShLib => write!(f, "Reserved"),
            SectionType::DynSym => write!(f, "Dynamic linker symbol table"),
            SectionType::InitArray => write!(f, "Array of constructors"),
            SectionType::FiniArray => write!(f, "Array of destructors"),
            SectionType::PreinitArray => {
                write!(f, "Array of pre-constructors")
            }
            SectionType::Group => write!(f, "Section group"),
            SectionType::SymTabShNdx => write!(f, "Extended section indices"),
            SectionType::GnuAttributes => write!(f, "Object attributes"),
            SectionType::GnuHash => write!(f, "GNU-style hash table"),
            SectionType::GnuLibList => write!(f, "Prelink library list"),
            SectionType::Checksum => write!(f, "Checksum for DSO content"),
            SectionType::SunwMove => write!(f, "SUN Move"),
            SectionType::SunwComDat => write!(f, "SUN COMDAT"),
            SectionType::SunwSymInfo => write!(f, "SUN SymInfo"),
            SectionType::GnuVerDef => write!(f, "Version definition section"),
            SectionType::GnuVerNeed => write!(f, "Version needs section"),
            SectionType::GnuVerSym => write!(f, "Version symbol table"),
            SectionType::RiscVAttributes => write!(f, "RISC-V attributes"),
            SectionType::Unknown(val) => write!(f, "Unknown ({})", val),
        }
    }
}

/// Section is writable.
pub const SHF_WRITE: u64 = 1 << 0;

/// Section occupies memory during execution.
pub const SHF_ALLOC: u64 = 1 << 1;

/// Section is executable.
pub const SHF_EXECINSTR: u64 = 1 << 2;

/// Section might be merged.
pub const SHF_MERGE: u64 = 1 << 4;

/// Section contains null-terminated strings.
pub const SHF_STRINGS: u64 = 1 << 5;

/// The field `info` contains SHT index.
pub const SHF_INFO_LINK: u64 = 1 << 6;

/// Preserve order after combining.
pub const SHF_LINK_ORDER: u64 = 1 << 7;

/// Non-standard OS specific handling required.
pub const SHF_OS_NONCONFORMING: u64 = 1 << 8;

/// Section is member of a group.
pub const SHF_GROUP: u64 = 1 << 9;

/// Section hold thread-local data.
pub const SHF_TLS: u64 = 1 << 10;

/// Section with compressed data.
pub const SHF_COMPRESSED: u64 = 1 << 11;

/// OS-specific.
pub const SHF_MASKOS: u64 = 0x0ff00000;

/// Processor-specific.
pub const SHF_MASKPROC: u64 = 0xf0000000;

/// Special ordering requirement (Solaris).
pub const SHF_ORDERED: u64 = 1 << 30;

/// Section is excluded unless referenced or allocated (Solaris).
pub const SHF_EXCLUDE: u64 = 1 << 31;
