//! Easy-to-use interface with NASM (The Netwide Assembler).

use std::fmt;
use std::fs;
use std::io;
use std::process::Command;

/// NASM assembler instance.
pub struct Nasm {
    /// command use to invoke NASM.
    cmd: String,
}

/// The error type for nasm operations.
#[derive(Debug)]
pub enum Error {
    CommandFailed,

    IoError(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::CommandFailed => write!(f, "command failed"),
            Error::IoError(err) => write!(f, "IO error: {}", err),
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IoError(err)
    }
}

impl Nasm {
    /// Creates a new NASM assembler. `cmd` is the command used to invoke NASM.
    pub fn new<S: Into<String>>(cmd: S) -> Nasm {
        Nasm { cmd: cmd.into() }
    }

    /// Assembles asm code. It returns the generated flat raw binary.
    pub fn assemble<C: AsRef<[u8]>>(&self, code: C) -> Result<Vec<u8>, Error> {
        let tmpdir = tempfile::tempdir()?;
        let input_file = tmpdir.path().join("input.asm");
        let output_file = tmpdir.path().join("output.bin");

        fs::write(&input_file, code)?;

        let status = Command::new(&self.cmd)
            .args(&[
                "-f",
                "bin",
                "-o",
                output_file.to_str().ok_or(Error::CommandFailed)?,
                input_file.to_str().ok_or(Error::CommandFailed)?,
            ])
            .status()?;

        if !status.success() {
            return Err(Error::CommandFailed);
        }

        let bytes = fs::read(&output_file)?;

        Ok(bytes)
    }
}

/// Assembles asm code using the default "nasm" command. It returns the
/// generated flat raw binary.
pub fn assemble<C: AsRef<[u8]>>(code: C) -> Result<Vec<u8>, Error> {
    Nasm::new("nasm").assemble(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_assemble() {
        let bytes = assemble("nop").unwrap();
        assert_eq!(bytes, &[0x90]);
    }

    #[test]
    fn nasm_assemble() {
        let nasm = Nasm::new("nasm");

        let bytes = nasm.assemble("nop").unwrap();
        assert_eq!(bytes, &[0x90]);
    }

    #[test]
    fn nasm_assemble_owned_string() {
        let nasm = Nasm::new(String::from("nasm"));

        let bytes = nasm.assemble(String::from("nop")).unwrap();
        assert_eq!(bytes, &[0x90]);
    }

    #[test]
    fn nasm_assemble_multiline() {
        let nasm = Nasm::new("nasm");

        let code = r#"
            BITS 32

            push ebp
            mov ebp, esp
            nop
            mov esp, ebp
            pop ebp
            ret
        "#;

        let bytes = nasm.assemble(code).unwrap();
        assert_eq!(bytes, b"\x55\x89\xe5\x90\x89\xec\x5d\xc3");
    }

    #[test]
    #[should_panic]
    fn nasm_assemble_invalid_cmd() {
        let nasm = Nasm::new("XXXnasmXXX");
        nasm.assemble("nop").unwrap();
    }

    #[test]
    #[should_panic]
    fn nasm_assemble_invalid_code() {
        let nasm = Nasm::new("nasm");
        nasm.assemble("push foobar").unwrap();
    }
}
