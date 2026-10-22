//! Summary statistics of a per-sample error over a split.
//!
//! [`ErrorStats`] is shared by the Poisson generator (`pde::poisson::dataset`, feature
//! `datagen`), which writes it into each split's `.json` sidecar, and by the evaluation
//! in [`poisson`](super::poisson), which writes it into a run's `errors.json`. It is
//! built without `datagen`, so both can import it.

use burn::config::Config;

/// Mean and max of one error over the samples. Serialised with the keys `mean` and
/// `max`, as the sidecars and `errors.json` hold them.
#[derive(Config, Copy, Debug, PartialEq)]
pub struct ErrorStats {
    /// Mean over the samples.
    pub mean: f64,
    /// Max over the samples.
    pub max: f64,
}

impl ErrorStats {
    /// The mean and max of `values`. A NaN value makes both NaN, and so does an empty
    /// `values`: an empty split is reported, not fatal.
    pub fn of(values: &[f64]) -> Self {
        // 0 / 0 = NaN when empty
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let max = if values.is_empty() {
            f64::NAN
        } else {
            // unlike f64::max, keeps a NaN so it is not hidden in the report
            values.iter().fold(
                f64::NEG_INFINITY,
                |m, &v| if v.is_nan() || v > m { v } else { m },
            )
        };
        Self { mean, max }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_are_mean_and_max() {
        let st = ErrorStats::of(&[1.0, 3.0, 2.0]);
        assert_eq!((st.mean, st.max), (2.0, 3.0));
        let st = ErrorStats::of(&[1.0, f64::NAN]);
        assert!(st.mean.is_nan() && st.max.is_nan());
    }

    #[test]
    fn error_stats_propagate_nan() {
        let ok = ErrorStats::of(&[1.0, 3.0]);
        assert_eq!((ok.max, ok.mean), (3.0, 2.0));
        let bad = ErrorStats::of(&[1.0, f64::NAN, 3.0]);
        assert!(bad.max.is_nan() && bad.mean.is_nan());
        let empty = ErrorStats::of(&[]);
        assert!(empty.max.is_nan() && empty.mean.is_nan());
    }

    #[test]
    fn serialises_mean_and_max() {
        let st = ErrorStats::of(&[0.5, 0.25]);
        let text = st.to_string();
        assert!(
            text.contains("\"mean\"") && text.contains("\"max\""),
            "{text}"
        );
        assert_eq!(ErrorStats::load_binary(text.as_bytes()).unwrap(), st);
    }
}
