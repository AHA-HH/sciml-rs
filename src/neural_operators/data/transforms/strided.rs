//! Strided row selection: subsampling and train/test splitting in one copy.

use ndarray::{Array, ArrayView, Dimension, Slice};
use std::ops::Range;

/// Copies rows `rows` of `field` (axis 0, the sample axis), keeping every `step`-th
/// point from index 0 on every other axis (CONVENTIONS §10). One allocation of just the
/// kept subset.
///
/// `field` has shape `[n, s_1, .., s_k]`; the result has shape
/// `[rows.len(), ceil(s_1 / step), .., ceil(s_k / step)]`.
///
/// # Panics
/// If `step == 0`, `rows.start > rows.end` or `rows.end > n`.
pub fn take_strided<A: Clone, D: Dimension>(
    field: ArrayView<A, D>,
    rows: Range<usize>,
    step: usize,
) -> Array<A, D> {
    assert!(step > 0, "step must be > 0");
    assert!(
        rows.start <= rows.end && rows.end <= field.shape()[0],
        "rows {rows:?} out of range for {} samples",
        field.shape()[0]
    );
    field
        .slice_each_axis(|ax| {
            if ax.axis.index() == 0 {
                Slice::new(rows.start as isize, Some(rows.end as isize), 1)
            } else {
                Slice::new(0, None, step as isize)
            }
        })
        .to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

    #[test]
    fn rank2_selects_rows_and_strides_the_axis() {
        let a = Array2::from_shape_fn((6, 8), |(i, j)| (i * 10 + j) as f64);
        let out = take_strided(a.view(), 2..5, 3);
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(out[[0, 0]], 20.0);
        assert_eq!(out[[1, 2]], 36.0);
        assert_eq!(out[[2, 1]], 43.0);
    }

    #[test]
    fn rank3_strides_every_spatial_axis() {
        // Distinct extents so an axis mix-up cannot pass.
        let a = Array3::from_shape_fn((4, 7, 9), |(i, j, k)| (i * 10_000 + j * 100 + k) as f64);
        let out = take_strided(a.view(), 1..3, 4);
        assert_eq!(out.shape(), &[2, 2, 3]);
        assert_eq!(out[[0, 0, 0]], 10_000.0);
        assert_eq!(out[[0, 1, 2]], 10_408.0);
        assert_eq!(out[[1, 1, 1]], 20_404.0);
    }

    #[test]
    fn step_one_full_range_is_identity() {
        let a = Array3::from_shape_fn((3, 4, 5), |(i, j, k)| (i + j * 3 + k * 7) as f64);
        assert_eq!(take_strided(a.view(), 0..3, 1), a);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn rows_past_the_end_panic() {
        take_strided(Array2::<f64>::zeros((3, 3)).view(), 1..4, 1);
    }

    #[test]
    #[should_panic(expected = "step must be > 0")]
    fn zero_step_panics() {
        take_strided(Array2::<f64>::zeros((3, 3)).view(), 0..3, 0);
    }
}
