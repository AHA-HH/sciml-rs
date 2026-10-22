//! The tail shared by the normalizing loaders (Darcy, Poisson): fit and apply the
//! normalizers, add the input channel axis and cast to the host dtype.
//!
//! Normalization (CONVENTIONS §10): both inputs are encoded with the `x` normalizer fit on
//! `x_train`; both target splits stay in physical units, and predictions are decoded back
//! to physical scale before the loss, not targets encoded forward. The returned
//! [`Normalizers::y`], fit on `y_train`, is what the caller needs to do that decode.

use crate::neural_operators::data::{
    dataset::OperatorDataset, host_float::HostFloat, io::errors::LoadError,
    transforms::normalizers::UnitGaussianNormalizer,
};
use ndarray::{ArrayD, Axis};

/// The two normalizers fitted while loading a dataset. Named fields because both have the
/// same type and swapping them fails silently.
pub struct Normalizers {
    /// Fitted on the train inputs `[N, *spatial]`; encodes inputs.
    pub x: UnitGaussianNormalizer,
    /// Fitted on the train targets `[N, *spatial]`; decodes predictions to physical scale.
    pub y: UnitGaussianNormalizer,
}

/// Checks that `n_train` can be normalized: the ddof = 1 std of a single sample is
/// 0/0 = NaN, which eps can't guard. Called before any file I/O.
///
/// # Errors
/// [`LoadError::Invalid`] if `n_train < 2`.
pub(crate) fn check_n_train(n_train: usize) -> Result<(), LoadError> {
    if n_train < 2 {
        return Err(LoadError::Invalid(format!(
            "n_train must be >= 2: a sample std needs at least 2 training samples, got {n_train}"
        )));
    }
    Ok(())
}

/// Normalizes a train/test pair and packages it: fits `x` on `x_train` and encodes both
/// inputs; fits `y` on `y_train` and encodes neither target split (both stay physical,
/// CONVENTIONS §10); adds a trailing channel axis to the inputs and casts to `T`.
///
/// All four fields are `[N, *spatial]` in `f64` with matching spatial shapes; the
/// datasets hold inputs `[N, *spatial, 1]` and targets `[N, *spatial]`.
///
/// # Panics
/// If `x_train` has fewer than 2 samples (callers check [`check_n_train`] first).
pub(crate) fn normalize_split_pair<T: HostFloat>(
    x_train: ArrayD<f64>,
    y_train: ArrayD<f64>,
    x_test: ArrayD<f64>,
    y_test: ArrayD<f64>,
) -> (OperatorDataset<T>, OperatorDataset<T>, Normalizers) {
    assert!(
        x_train.shape()[0] >= 2,
        "normalize_split_pair needs n_train >= 2"
    );
    let x = UnitGaussianNormalizer::fit(&x_train);
    let y = UnitGaussianNormalizer::fit(&y_train);

    (
        encode_split(&x, x_train, y_train),
        encode_split(&x, x_test, y_test),
        Normalizers { x, y },
    )
}

/// Packages a test split with normalizers fitted at training time (a run's saved ones):
/// encodes `x_test` with `normalizers.x`, keeps `y_test` physical, adds the channel axis
/// and casts to `T`, exactly as [`normalize_split_pair`] does with the refit ones.
///
/// `x_test` and `y_test` are `[N, *spatial]` in `f64`; the dataset holds inputs
/// `[N, *spatial, 1]` and targets `[N, *spatial]`.
///
/// # Errors
/// [`LoadError::Invalid`] if `normalizers.x` was not fitted on `*spatial`.
pub(crate) fn normalize_test_split<T: HostFloat>(
    x_test: ArrayD<f64>,
    y_test: ArrayD<f64>,
    normalizers: &Normalizers,
) -> Result<OperatorDataset<T>, LoadError> {
    let fitted = normalizers.x.mean().shape();
    if fitted != &x_test.shape()[1..] {
        return Err(LoadError::Invalid(format!(
            "the x normalizer was fitted on {fitted:?}, but the test samples are {:?}",
            &x_test.shape()[1..]
        )));
    }
    Ok(encode_split(&normalizers.x, x_test, y_test))
}

/// The encode step shared by both entry points: inputs encoded with `x` and given the
/// channel axis, targets left physical (CONVENTIONS §10), both cast to `T`.
fn encode_split<T: HostFloat>(
    x: &UnitGaussianNormalizer,
    inputs: ArrayD<f64>,
    targets: ArrayD<f64>,
) -> OperatorDataset<T> {
    OperatorDataset::from_f64(with_channel(x.encode(inputs)), targets)
}

/// `[N, *spatial]` → `[N, *spatial, 1]`: the single data channel the model expects.
pub(crate) fn with_channel(a: ArrayD<f64>) -> ArrayD<f64> {
    let last = a.ndim();
    a.insert_axis(Axis(last))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn with_channel_is_a_standard_layout_reshape() {
        let a = ArrayD::from_shape_fn(IxDyn(&[2, 3, 4]), |i| (i[0] * 12 + i[1] * 4 + i[2]) as f64);
        let b = with_channel(a.clone());
        assert_eq!(b.shape(), [2, 3, 4, 1]);
        assert!(b.is_standard_layout());
        assert_eq!(b, a.into_shape_with_order(IxDyn(&[2, 3, 4, 1])).unwrap());
    }
}
