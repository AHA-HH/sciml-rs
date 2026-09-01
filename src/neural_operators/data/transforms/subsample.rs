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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn keeps_every_nth_element() {
        let data = ArrayD::from_shape_fn(IxDyn(&[8]), |i| i[0] as f64);
        let out = subsample(data, 0, 2);
        assert_eq!(out.shape(), &[4]);
        assert_eq!(
            out.iter().copied().collect::<Vec<_>>(),
            vec![0.0, 2.0, 4.0, 6.0]
        );
    }

    #[test]
    fn subsamples_only_the_named_axis() {
        // Distinct dims so an axis mix-up can't pass.
        let data = ArrayD::from_shape_fn(IxDyn(&[2, 6, 3]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64
        });
        let out = subsample(data, 1, 3);
        assert_eq!(out.shape(), &[2, 2, 3]);
        assert_eq!(out[[0, 0, 0]], 0.0); // i=(0,0,0)
        assert_eq!(out[[0, 1, 0]], 30.0); // i=(0,3,0)
        assert_eq!(out[[1, 1, 2]], 132.0); // i=(1,3,2)
    }

    #[test]
    fn rate_one_is_identity() {
        let data = ArrayD::from_shape_fn(IxDyn(&[5, 4]), |i| (i[0] * 10 + i[1]) as f64);
        assert_eq!(subsample(data.clone(), 0, 1), data);
    }
}
