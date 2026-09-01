//! Data pipeline utilities for train/test splitting of operator-learning datasets.

use ndarray::{ArrayD, Axis, Slice};

/// Splits `data` along the batch axis (axis 0) into a training and test set.
///
/// Returns `(train, test)`: the first `n_train` examples and the last `n_test`
/// examples. If `n_train + n_test < n_total`, the examples strictly between
/// them are dropped from both returns, not included in either.
///
/// Panics if `n_train + n_test` exceeds the total number of examples.
pub fn train_test_split(
    data: ArrayD<f64>,
    n_train: usize,
    n_test: usize,
) -> (ArrayD<f64>, ArrayD<f64>) {
    let n_total = data.shape()[0];
    assert!(
        n_train + n_test <= n_total,
        "n_train ({}) + n_test ({}) = {} exceeds n_total ({})",
        n_train,
        n_test,
        n_train + n_test,
        n_total
    );

    // first n_train examples along the batch axis
    let train = data
        .slice_axis(Axis(0), Slice::new(0, Some(n_train as isize), 1))
        .to_owned();
    // last n_test examples along the batch axis
    let test = data
        .slice_axis(Axis(0), Slice::new(-(n_test as isize), None, 1))
        .to_owned();

    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;

    #[test]
    fn takes_first_n_train_and_last_n_test() {
        // 10 examples, value = index, so provenance is visible.
        let data = ArrayD::from_shape_fn(IxDyn(&[10, 2]), |i| (i[0] * 10 + i[1]) as f64);
        let (train, test) = train_test_split(data, 3, 2);

        assert_eq!(train.shape(), &[3, 2]);
        assert_eq!(test.shape(), &[2, 2]);

        assert_eq!(train[[0, 0]], 0.0); // example 0
        assert_eq!(train[[2, 0]], 20.0); // example 2
        assert_eq!(test[[0, 0]], 80.0); // example 8, NOT 3 — test comes from the end
        assert_eq!(test[[1, 0]], 90.0); // example 9
    }

    #[test]
    #[should_panic(expected = "exceeds n_total")]
    fn rejects_oversized_split() {
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 2]), |_| 0.0);
        train_test_split(data, 3, 3);
    }
}
