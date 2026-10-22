//! Darcy flow dataset: config and constructor.
//!
//! `DarcyConfig` holds Darcy-specific settings (subsample rate, derived
//! resolution) on top of the shared `DatasetConfig`. `load_darcy_uniform`
//! builds train/test `OperatorDataset`s from separate `.mat` files: reading,
//! truncating to configured sizes, subsampling both spatial axes, fitting and
//! applying normalization. Inputs carry the data channel only; the model
//! appends the 2D grid coordinates itself (see `FNO::forward`).
//!
//! normalization (CONVENTIONS §10): both inputs are encoded using the `x`
//! normalizer fit on `x_train`; both target splits stay in physical units -
//! predictions are decoded back to physical scale before the loss, not targets
//! encoded forward. The returned `normalizers.y`, fit on `y_train`, is what the
//! caller needs to do that decode.

use crate::neural_operators::data::{
    dataset::OperatorDataset,
    host_float::HostFloat,
    io::{errors::LoadError, readers::mat::MatFileReader},
    loaders::{
        base_dataset::DatasetConfig,
        common::{Normalizers, check_n_train, normalize_split_pair},
    },
};
use burn::config::Config;
use ndarray::{ArrayD, s};
use std::path::Path;

/// Darcy-specific config: subsample rate and the resolution it implies.
#[derive(Config, Debug)]
pub struct DarcyConfig {
    pub base: DatasetConfig,
    pub subsample_rate: usize,
}

impl DarcyConfig {
    /// Resolution after subsampling, assuming a raw resolution of 421.
    /// Checked against the actually-read resolution in `load_darcy_uniform`.
    pub fn s(&self) -> usize {
        (421 - 1) / self.subsample_rate + 1
    }
}

/// Builds the Darcy flow dataset: inputs `[n, s, s, 1]` (the normalized
/// coefficient; no grid channels, the model generates them), targets
/// `[n, s, s]`.
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is
/// what the caller needs to decode predictions back to physical scale;
/// `normalizers.x` is only needed for encoding inputs that didn't come from
/// this dataset.
///
/// # Errors
/// [`LoadError::Reader`] if a file can't be read or lacks `coeff`/`sol`;
/// [`LoadError::Invalid`] if `n_train < 2` (the normalizers' sample std is
/// undefined for one sample), the fields aren't `[samples, s, s]`, a file has
/// fewer samples than requested, or the subsample rate doesn't give
/// `config.s()` points per axis.
///
/// Reading and normalization are computed in `f64`, and the
/// normalizers keep `f64` statistics; the datasets are stored as `T`,
/// rounded once at the end (see [`HostFloat`]).
pub fn load_darcy_uniform<T: HostFloat>(
    train_path: impl AsRef<Path>,
    test_path: impl AsRef<Path>,
    config: &DarcyConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, Normalizers), LoadError> {
    if config.subsample_rate == 0 {
        return Err(LoadError::Invalid("subsample_rate must be > 0".into()));
    }
    check_n_train(config.base.n_train)?;

    // Read input ('coeff') and target ('sol') from each split's .mat file,
    // validate them, then truncate to the configured sample count and subsample
    // both spatial axes in one strided copy. Doing this per split means the
    // reader and full-size fields of one file are freed before the next is
    // parsed. Without the truncation, dataset size is whatever the file holds.
    let r = config.subsample_rate;
    let load_split = |path: &Path, split: &str, n: usize| -> Result<_, LoadError> {
        let (x, y) = {
            let reader = MatFileReader::new(path)?;
            (reader.read_field("coeff")?, reader.read_field("sol")?)
        };
        for (name, field) in [("coeff", &x), ("sol", &y)] {
            if field.ndim() != 3 {
                return Err(LoadError::Invalid(format!(
                    "{split} field '{name}' must be [samples, s, s], got shape {:?}",
                    field.shape()
                )));
            }
            if n > field.shape()[0] {
                return Err(LoadError::Invalid(format!(
                    "requested {n} {split} samples but '{name}' has {}",
                    field.shape()[0]
                )));
            }
        }
        Ok((take_subsampled(x, n, r), take_subsampled(y, n, r)))
    };
    let (x_train, y_train) = load_split(train_path.as_ref(), "train", config.base.n_train)?;
    let (x_test, y_test) = load_split(test_path.as_ref(), "test", config.base.n_test)?;
    darcy_from_fields(x_train, y_train, x_test, y_test, config)
}

/// The rest of [`load_darcy_uniform`] on truncated, subsampled fields:
/// check the resolution, then normalize, add the channel axis and cast. Separate from the
/// reader so the output can be tested without `.mat` files.
fn darcy_from_fields<T: HostFloat>(
    x_train: ArrayD<f64>,
    y_train: ArrayD<f64>,
    x_test: ArrayD<f64>,
    y_test: ArrayD<f64>,
    config: &DarcyConfig,
) -> Result<(OperatorDataset<T>, OperatorDataset<T>, Normalizers), LoadError> {
    let s = x_train.shape()[1];

    // cross-check the derived resolution against what was actually subsampled
    let spatial_ok = [&x_train, &y_train, &x_test, &y_test]
        .iter()
        .all(|a| a.shape()[1] == s && a.shape()[2] == s);
    if s != config.s() || !spatial_ok {
        return Err(LoadError::Invalid(format!(
            "subsampled grids must all be {0}x{0} (config.s()), got train {1:?} / test {2:?}",
            config.s(),
            &x_train.shape()[1..],
            &x_test.shape()[1..]
        )));
    }

    Ok(normalize_split_pair(x_train, y_train, x_test, y_test))
}

/// The first `n` samples of a `[samples, s, s]` field with every `r`-th
/// point kept on both spatial axes, as a standard-layout array.
///
/// One allocation of just the kept subset; when nothing is dropped the input
/// buffer is returned as is (the reader already produces standard layout).
fn take_subsampled(field: ArrayD<f64>, n: usize, r: usize) -> ArrayD<f64> {
    if n == field.shape()[0] && r == 1 {
        return field;
    }
    let step = r as isize;
    field.slice(s![..n, ..;step, ..;step]).to_owned().into_dyn()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::io::errors::ReaderError;
    use ndarray::IxDyn;

    fn cfg(subsample_rate: usize) -> DarcyConfig {
        DarcyConfig::new(
            DatasetConfig {
                n_train: 2,
                n_test: 1,
            },
            subsample_rate,
        )
    }

    // --- REVIEW1.md N1: one training sample gave an all-NaN normalizer ---

    #[test]
    fn single_train_sample_is_invalid_not_nan() {
        let mut config = cfg(5);
        config.base.n_train = 1;
        // nonexistent paths: the check must run before any file I/O
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &config)
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
        assert!(err.to_string().contains("at least 2"), "{err}");
    }

    #[test]
    fn missing_file_is_a_reader_error() {
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &cfg(5))
            .err()
            .expect("loading should fail");
        assert!(
            matches!(err, LoadError::Reader(ReaderError::FileNotFound(_))),
            "{err:?}"
        );
        assert!(err.to_string().contains("train.mat"), "{err}");
    }

    #[test]
    fn zero_subsample_rate_is_invalid_not_a_panic() {
        let err = load_darcy_uniform::<f32>("/nope/train.mat", "/nope/test.mat", &cfg(0))
            .err()
            .expect("loading should fail");
        assert!(matches!(err, LoadError::Invalid(_)), "{err:?}");
    }

    // --- REVIEW.md 4.7: one strided copy == old slice + 2x subsample ---

    fn old_path(field: ArrayD<f64>, n: usize, r: usize) -> ArrayD<f64> {
        use crate::neural_operators::data::transforms::subsample::subsample;
        let field = field.slice_axis(ndarray::Axis(0), (0..n).into()).to_owned();
        subsample(subsample(field, 1, r), 2, r)
    }

    #[test]
    fn take_subsampled_matches_old_slice_and_subsample() {
        // Non-square spatial extents and a rate that doesn't divide them.
        let field = ArrayD::from_shape_fn(IxDyn(&[5, 7, 9]), |i| {
            (i[0] * 10_000 + i[1] * 100 + i[2]) as f64
        });
        for (n, r) in [(5, 1), (3, 1), (5, 2), (2, 3), (1, 4)] {
            let new = take_subsampled(field.clone(), n, r);
            assert_eq!(new, old_path(field.clone(), n, r), "n = {n}, r = {r}");
            assert!(new.is_standard_layout(), "n = {n}, r = {r}");
            // the loader's row-major reshape must succeed
            let shape = [new.shape(), &[1]].concat();
            assert!(new.into_shape_with_order(IxDyn(&shape)).is_ok());
        }
    }

    #[test]
    fn take_subsampled_reuses_buffer_when_nothing_is_dropped() {
        let field = ArrayD::from_shape_fn(IxDyn(&[2, 3, 3]), |i| i[2] as f64);
        let ptr = field.as_ptr();
        assert_eq!(take_subsampled(field, 2, 1).as_ptr(), ptr);
    }

    // --- issue #13: no grid channels in the stored inputs ---

    #[test]
    fn inputs_hold_the_normalized_coefficient_only() {
        let config = cfg(210); // s = 3
        let s = config.s();
        let field = |n: usize, seed: f64| {
            ArrayD::from_shape_fn(IxDyn(&[n, s, s]), |i| {
                ((i[0] * s * s + i[1] * s + i[2]) as f64 * 0.7 + seed).sin()
            })
        };
        let (x_train, y_train, x_test, y_test) =
            (field(2, 0.0), field(2, 1.0), field(1, 2.0), field(1, 3.0));
        let (train, test, norms) = darcy_from_fields::<f64>(
            x_train.clone(),
            y_train.clone(),
            x_test.clone(),
            y_test.clone(),
            &config,
        )
        .expect("valid fields");

        assert_eq!(train.inputs().shape(), [2, s, s, 1]);
        assert_eq!(test.inputs().shape(), [1, s, s, 1]);
        let as_input = |a: ArrayD<f64>| {
            let n = a.shape()[0];
            a.into_shape_with_order(IxDyn(&[n, s, s, 1])).unwrap()
        };
        // x encoded with the train normalizer, both y splits raw.
        assert_eq!(*train.inputs(), as_input(norms.x.encode(x_train)));
        assert_eq!(*test.inputs(), as_input(norms.x.encode(x_test)));
        assert_eq!(*train.targets(), y_train);
        assert_eq!(*test.targets(), y_test);
    }

    /// Order-sensitive sum, compared bit for bit.
    fn checksum<T: Copy + Into<f64>>(a: &ArrayD<T>) -> u64 {
        let sum: f64 = a
            .iter()
            .enumerate()
            .map(|(i, &v)| v.into() * (i + 1) as f64)
            .sum();
        sum.to_bits()
    }

    // --- phase 4s T9: the loader tail's output is pinned across its refactor ---

    #[test]
    fn darcy_output_is_pinned() {
        let config = DarcyConfig::new(
            DatasetConfig {
                n_train: 3,
                n_test: 2,
            },
            105,
        ); // s = 5
        let s = config.s();
        let field = |n: usize, seed: f64| {
            ArrayD::from_shape_fn(IxDyn(&[n, s, s]), |i| {
                ((i[0] * s * s + i[1] * s + i[2]) as f64 * 0.37 + seed).sin() * (1.0 + seed)
            })
        };
        let (train, test, norms) = darcy_from_fields::<f32>(
            field(3, 0.0),
            field(3, 1.0),
            field(2, 2.0),
            field(2, 3.0),
            &config,
        )
        .expect("valid fields");

        assert_eq!(train.inputs().shape(), [3, s, s, 1]);
        assert_eq!(train.targets().shape(), [3, s, s]);
        assert_eq!(test.inputs().shape(), [2, s, s, 1]);
        assert_eq!(test.targets().shape(), [2, s, s]);
        let sums = [
            checksum(train.inputs()),
            checksum(train.targets()),
            checksum(test.inputs()),
            checksum(test.targets()),
            checksum(norms.x.mean()),
            checksum(norms.x.std()),
            checksum(norms.y.mean()),
            checksum(norms.y.std()),
        ];
        assert_eq!(
            sums,
            [
                13857006548521320448,
                4645398979642392576,
                13869920727614881792,
                4646307104645971968,
                4626803676421486758,
                4643276683912197713,
                4630239275482198924,
                4646823668636631701,
            ],
            "{sums:?}"
        );
    }
}
