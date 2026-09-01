//! Array reshaping utilities for building model input tensors.

use ndarray::ArrayD;

/// Subsamples `data` along axis `dim`, keeping every `rate`-th element.
/// Dimension-generic: works on any array rank.
pub fn subsample(data: ArrayD<f64>, dim: usize, rate: usize) -> ArrayD<f64> {
    assert!(rate > 0, "subsample rate must be > 0");
    assert!(
        dim < data.ndim(),
        "dim {} out of bounds for array with {} dims",
        dim,
        data.ndim()
    );

    // ndarray's slice_axis takes a dimension and a Slice
    data.slice_axis(
        ndarray::Axis(dim),
        ndarray::Slice::new(0, None, rate as isize),
    )
    .to_owned() // slice returns a view, to_owned() makes it an owned array
}
