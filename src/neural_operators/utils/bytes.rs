//! [`LeReader`]: a cursor over little-endian binary framing, for the crate's hand-written
//! file formats. Every read checks the length first, so a short or overlong input is an
//! [`io::ErrorKind::InvalidData`] error, never a panic or an oversized allocation.
//!
//! The Gauss–Newton state (`optim::gauss_newton::GaussNewtonState`) reads through it.
//! `runs::checkpoint` still has its own copy of these helpers, to be ported when Phase 10
//! T3 nests the Gauss–Newton state inside `checkpoint.bin`.

use std::io;

/// An [`io::ErrorKind::InvalidData`] error with message `what`.
pub(crate) fn invalid(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

/// Reads little-endian values from the front of a byte slice, advancing it.
pub(crate) struct LeReader<'a> {
    rest: &'a [u8],
    /// What is being read, for error messages ("checkpoint", "Gauss-Newton state").
    what: &'static str,
}

impl<'a> LeReader<'a> {
    /// A reader over `bytes`; `what` names the format in error messages.
    pub(crate) fn new(bytes: &'a [u8], what: &'static str) -> Self {
        Self { rest: bytes, what }
    }

    /// The next `n` bytes.
    ///
    /// # Errors
    /// "`what` is truncated" if fewer than `n` remain.
    pub(crate) fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        if self.rest.len() < n {
            return Err(invalid(format!("{} is truncated", self.what)));
        }
        let (head, tail) = self.rest.split_at(n);
        self.rest = tail;
        Ok(head)
    }

    /// The next byte.
    pub(crate) fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    /// The next 8 bytes as a `u64`.
    pub(crate) fn u64(&mut self) -> io::Result<u64> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().expect("8 bytes")))
    }

    /// The next 8 bytes as an `f64`.
    pub(crate) fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// The next `n` f64 values (8n bytes).
    ///
    /// # Errors
    /// As [`take`](Self::take), or if 8n overflows.
    pub(crate) fn f64s(&mut self, n: usize) -> io::Result<Vec<f64>> {
        let len = n
            .checked_mul(8)
            .ok_or_else(|| invalid(format!("{}: length overflows", self.what)))?;
        let raw = self.take(len)?;
        Ok((0..n)
            .map(|k| f64::from_le_bytes(raw[8 * k..8 * k + 8].try_into().expect("8 bytes")))
            .collect())
    }

    /// Ends the read.
    ///
    /// # Errors
    /// "trailing bytes after the `what`" if any input is left.
    pub(crate) fn finish(self) -> io::Result<()> {
        if self.rest.is_empty() {
            Ok(())
        } else {
            Err(invalid(format!("trailing bytes after the {}", self.what)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_values_in_order_and_rejects_short_or_long_input() {
        let mut bytes = vec![7u8];
        bytes.extend_from_slice(&42u64.to_le_bytes());
        bytes.extend_from_slice(&1.5f64.to_le_bytes());
        bytes.extend_from_slice(&(-2.25f64).to_le_bytes());
        let mut r = LeReader::new(&bytes, "test blob");
        assert_eq!(r.u8().unwrap(), 7);
        assert_eq!(r.u64().unwrap(), 42);
        assert_eq!(r.f64s(2).unwrap(), [1.5, -2.25]);
        r.finish().unwrap();

        let mut short = LeReader::new(&bytes[..4], "test blob");
        short.u8().unwrap();
        let err = short.u64().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert_eq!(err.to_string(), "test blob is truncated");

        let long = LeReader::new(&bytes, "test blob");
        assert_eq!(
            long.finish().unwrap_err().to_string(),
            "trailing bytes after the test blob"
        );
        assert!(LeReader::new(&bytes, "x").f64s(usize::MAX).is_err());
    }
}
