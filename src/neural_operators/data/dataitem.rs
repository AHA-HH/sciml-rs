//! Shared data-pipeline types used by both `OperatorDataset` and `OperatorBatcher`.

use ndarray::ArrayD;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Float type the preprocessed dataset is stored in on the host: `f32` or `f64`.
///
/// Loaders read and preprocess in `f64` (normalizer statistics included) and
/// cast to `T` once, when the dataset is built. The host dtype does not set
/// the training precision: batches are converted to the device's default
/// float dtype on upload. `f32` halves host memory; since casting the same
/// `f64` values once, earlier, rounds exactly as the per-batch conversion
/// does, an `f32` device sees bit-identical values either way. `f64` keeps
/// full precision for an `f64` device.
///
/// Sealed: implemented for `f32` and `f64` only.
pub trait HostFloat: burn::tensor::Element + sealed::Sealed {
    /// Rounds an `f64` to this type (round to nearest for `f32`).
    fn from_f64(v: f64) -> Self;
}

impl HostFloat for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl HostFloat for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
}

/// A single training example: one input/target pair seen by the model during training.
#[derive(Clone, Debug)]
pub struct DataItem<T: HostFloat> {
    pub input: ArrayD<T>,
    pub target: ArrayD<T>,
}
