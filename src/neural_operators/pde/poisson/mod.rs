//! −Δu = f on [−1, 1]² with u = 0 on the boundary (design §4).
//!
//! [`collocation`] is solver A, the source of every training label. [`sparse`] is solver C,
//! the validation oracle that checks A; it never produces labels.

pub mod collocation;
#[cfg(test)]
mod manufactured;
pub mod sparse;
