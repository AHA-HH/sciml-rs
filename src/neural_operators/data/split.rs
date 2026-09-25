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

/// Splits `data` along `axis` into an input window and the target window
/// that directly follows it.
///
/// Returns `(input, target)`: entries `0..n_in` and `n_in..n_in + n_out`
/// along `axis`. Entries past `n_in + n_out` are dropped. Used to turn a
/// trajectory's time axis into "first `n_in` snapshots in, next `n_out` out".
///
/// Panics if `n_in` or `n_out` is zero, `axis` is out of bounds, or
/// `n_in + n_out` exceeds the length of `axis`.
pub fn input_target_split(
    data: &ArrayD<f64>,
    axis: usize,
    n_in: usize,
    n_out: usize,
) -> (ArrayD<f64>, ArrayD<f64>) {
    assert!(n_in > 0 && n_out > 0, "n_in and n_out must be > 0");
    assert!(
        axis < data.ndim(),
        "axis {} out of bounds for array with {} dims",
        axis,
        data.ndim()
    );
    let len = data.shape()[axis];
    assert!(
        n_in + n_out <= len,
        "n_in ({}) + n_out ({}) = {} exceeds the {} entries along axis {}",
        n_in,
        n_out,
        n_in + n_out,
        len,
        axis
    );

    let input = data.slice_axis(Axis(axis), Slice::from(0..n_in)).to_owned();
    let target = data
        .slice_axis(Axis(axis), Slice::from(n_in..n_in + n_out))
        .to_owned();

    (input, target)
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

    #[test]
    fn input_target_split_takes_consecutive_windows() {
        // value = 10 * sample + time, so the source time index is visible.
        let data = ArrayD::from_shape_fn(IxDyn(&[2, 7]), |i| (i[0] * 10 + i[1]) as f64);
        let (input, target) = input_target_split(&data, 1, 2, 3);

        assert_eq!(input.shape(), &[2, 2]);
        assert_eq!(target.shape(), &[2, 3]);

        assert_eq!(input[[1, 0]], 10.0); // time 0
        assert_eq!(input[[1, 1]], 11.0); // time 1
        assert_eq!(target[[1, 0]], 12.0); // time 2, directly after the input
        assert_eq!(target[[1, 2]], 14.0); // time 4; times 5, 6 dropped
    }

    #[test]
    #[should_panic(expected = "exceeds the 4 entries along axis 1")]
    fn input_target_split_rejects_oversized_windows() {
        let data = ArrayD::zeros(IxDyn(&[2, 4]));
        input_target_split(&data, 1, 2, 3);
    }

    #[test]
    #[should_panic(expected = "must be > 0")]
    fn input_target_split_rejects_empty_window() {
        let data = ArrayD::zeros(IxDyn(&[2, 4]));
        input_target_split(&data, 1, 0, 3);
    }
}
