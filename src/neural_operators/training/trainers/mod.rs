//! Per-PDE trainers
//!
//! Concrete training routines that specialise the generic loop for a given
//! dataset, wiring its batcher, model shape and normaliser.

pub mod burgers;
pub mod darcy;
