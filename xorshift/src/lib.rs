//! Implementation of a Xorshift pseudorandom number generator. Xorshift is a
//! class of simple and extremely fast random number generators.
//!
//! References:
//! - [Original paper][paper]
//! - [Wikipedia entry][wikipedia]
//!
//! [paper]: https://www.jstatsoft.org/index.php/jss/article/view/v008i14/xorshift.pdf
//! [wikipedia]: https://en.wikipedia.org/wiki/Xorshift

/// Xorshift pseudorandom number generator.
pub struct Rng(u64);

impl Rng {
    /// Returns a new Xorshift PRNG.
    pub fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    /// Returns the next number in the sequence.
    pub fn rand(&mut self) -> usize {
        let val = self.0;

        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;

        val as usize
    }
}
