//! EndianRead is a type of reader that makes easy to read values from their
//! representation as a byte slice in little and big endian.

use std::convert::TryInto;
use std::io::{self, Cursor, Read};

/// Types implementing this trait can be created from their representation as a
/// byte slice in little and big endian.
pub trait FromBytes {
    type Target;

    /// Create a native endian value from its representation as a byte slice in
    /// little endian.
    ///
    /// # Panics
    ///
    /// This function will panic if the size of the slice does not match the
    /// size of the target type.
    fn from_le_bytes(bytes: &[u8]) -> Self::Target;

    /// Create a native endian value from its representation as a byte slice in
    /// big endian.
    ///
    /// # Panics
    ///
    /// This function will panic if the size of the slice does not match the
    /// size of the target type.
    fn from_be_bytes(bytes: &[u8]) -> Self::Target;
}

macro_rules! impl_from_bytes {
    ($Ty:ty) => {
        impl FromBytes for $Ty {
            type Target = $Ty;

            fn from_le_bytes(bytes: &[u8]) -> $Ty {
                <$Ty>::from_le_bytes(bytes.try_into().unwrap())
            }

            fn from_be_bytes(bytes: &[u8]) -> $Ty {
                <$Ty>::from_be_bytes(bytes.try_into().unwrap())
            }
        }
    };
}

// Implement `FromBytes` for unsigned integers.
impl_from_bytes!(u8);
impl_from_bytes!(u16);
impl_from_bytes!(u32);
impl_from_bytes!(u64);

// Implement `FromBytes` for signed integers.
impl_from_bytes!(i8);
impl_from_bytes!(i16);
impl_from_bytes!(i32);
impl_from_bytes!(i64);

/// An `EndianRead` is a type of `Read`er which allows to read values with a
/// specific endianness.
pub trait EndianRead: Read {
    /// Read a value as little endian.
    fn read_le<T: FromBytes>(&mut self) -> Result<T::Target, io::Error> {
        let size = std::mem::size_of::<T::Target>();

        let mut buf = vec![0u8; size];
        self.read_exact(&mut buf)?;

        Ok(T::from_le_bytes(&buf))
    }

    /// Read a value as big endian.
    fn read_be<T: FromBytes>(&mut self) -> Result<T::Target, io::Error> {
        let size = std::mem::size_of::<T::Target>();

        let mut buf = vec![0u8; size];
        self.read_exact(&mut buf)?;

        Ok(T::from_be_bytes(&buf))
    }
}

impl<T> EndianRead for Cursor<T> where T: AsRef<[u8]> {}

#[cfg(test)]
mod tests {
    use std::io::{Seek, SeekFrom};

    use super::*;

    #[test]
    fn read_le() {
        let mut buf = Cursor::new(vec![
            0xff, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xff,
        ]);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<u8>().unwrap();
        assert_eq!(value, 0x41u8);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<u16>().unwrap();
        assert_eq!(value, 0x4241u16);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<u32>().unwrap();
        assert_eq!(value, 0x44434241u32);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<u64>().unwrap();
        assert_eq!(value, 0x4847464544434241u64);

        // Signed integers.
        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<i8>().unwrap();
        assert_eq!(value, 0x41i8);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<i16>().unwrap();
        assert_eq!(value, 0x4241i16);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<i32>().unwrap();
        assert_eq!(value, 0x44434241i32);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<i64>().unwrap();
        assert_eq!(value, 0x4847464544434241i64);
    }

    #[test]
    fn read_be() {
        let mut buf = Cursor::new(vec![
            0xff, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0xff,
        ]);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<u8>().unwrap();
        assert_eq!(value, 0x41u8);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<u16>().unwrap();
        assert_eq!(value, 0x4142u16);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<u32>().unwrap();
        assert_eq!(value, 0x41424344u32);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<u64>().unwrap();
        assert_eq!(value, 0x4142434445464748u64);

        // Signed integers.
        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<i8>().unwrap();
        assert_eq!(value, 0x41i8);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<i16>().unwrap();
        assert_eq!(value, 0x4142i16);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<i32>().unwrap();
        assert_eq!(value, 0x41424344i32);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_be::<i64>().unwrap();
        assert_eq!(value, 0x4142434445464748i64);
    }

    #[test]
    fn read_oob() {
        let mut buf = Cursor::new(vec![0xff, 0x41]);

        buf.seek(SeekFrom::Start(0)).unwrap();
        assert!(buf.read_le::<u16>().is_ok());

        buf.seek(SeekFrom::Start(1)).unwrap();
        assert!(buf.read_le::<u16>().is_err());

        buf.seek(SeekFrom::Start(0)).unwrap();
        assert!(buf.read_le::<u32>().is_err());

        buf.seek(SeekFrom::Start(2)).unwrap();
        assert!(buf.read_le::<u8>().is_err());
    }

    #[test]
    fn read_multiple() {
        let mut buf = Cursor::new(vec![0xff, 0x41, 0x42, 0x43, 0x44, 0x45]);

        buf.seek(SeekFrom::Start(1)).unwrap();
        let value = buf.read_le::<u8>().unwrap();
        assert_eq!(value, 0x41u8);
        let value = buf.read_le::<u32>().unwrap();
        assert_eq!(value, 0x45444342u32);
    }
}
