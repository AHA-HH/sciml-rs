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