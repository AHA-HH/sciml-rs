//! Models
//!
//! Full operator networks assembled from the layer primitives (`fno`), the trait every
//! operator implements for the training stack (`operator`), and batched inference to the
//! host (`predict`).

pub mod fno;
pub mod operator;
pub mod predict;
