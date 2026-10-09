//! −Δu = f on [−1, 1]² with u = 0 on the boundary (design §4).
//!
//! [`collocation`] is solver A, the source of every training label. [`sparse`] is solver C,
//! the validation oracle that checks A; it never produces labels. [`grf`] samples the GRF
//! forcings of design §3.3, and [`dataset`] turns them into labelled, checked splits
//! (design §5.1).

pub mod collocation;
pub mod dataset;
pub mod grf;
#[cfg(test)]
mod manufactured;
pub mod sparse;
