//! 2D Navier-Stokes (vorticity form) dataset: config and constructor.
//!
//! Follows `fourier_3d.py` from Li et al. (zongyi-li/fourier_neural_operator):
//! the first `t_in` vorticity snapshots are the input and the next `t_out`
//! snapshots are the target, predicted in a single pass over (x, y, t).
//! `load_navier_stokes_uniform` builds train/test `NavierStokesDataset`s from
//! a `.mat` source file: reading, subsampling both spatial axes, splitting
//! samples and time, and fitting and applying normalization.
//!
//! # Source data
//!
//! A MATLAB v5 `.mat` holding one field `u` of shape `[N, s, s, T]`: sample,
//! x, y, time (e.g. `ns_V1e-3_N1200_T50.mat`, `[1200, 64, 64, 50]`, single
//! precision). v7.3 (HDF5) files can't be read by `MatFileReader` and must be
//! re-saved as v5 first.
//!
//! # Split
//!
//! Train takes the first `n_train` samples, test the last `n_test`, as Li does.
//!
//! # Layout
//!
//! Each item is built lazily in [`NavierStokesDataset::get`]:
//! - input `[s, s, t_out, 3 + t_in]`: channels are (x, y, t, u_0..u_{t_in-1}).
//!   The `t_in` input snapshots are repeated at every output time. Grid
//!   channels come first, matching Li's concatenation order and
//!   [`append_grid_3d`](crate::neural_operators::data::grids::append_grid_3d).
//! - target `[s, s, t_out]`.
//!
//! Space is on [0, 1] with both endpoints; time is on (0, 1], so
//! t_k = (k + 1) / t_out. See `uniform_grid_3d`.
//!
//! Materialising the repeated input up front would cost `t_out` times the
//! memory of the input snapshots (~17 GB in f64 for Li's setup), so only the
//! unrepeated snapshots are stored.
//!
//! # Normalization
//!
//! Same asymmetry as Darcy: the `x` normalizer is fitted on the training
//! inputs and applied to both splits; the `y` normalizer is fitted on the
//! training targets and applied to them only. Test targets stay in physical
//! units, and predictions are decoded with `y` before comparison. Both are
//! fitted pointwise (per x, y, t) before the time repeat.

use crate::neural_operators::data::{
    dataitem::DataItem,
    grids::uniform_grid_3d,
    io::{readers::mat::MatFileReader, traits::FieldReader},
    loaders::base_dataset::{BaseDatasetConfig, DatasetConfig, HasBaseConfig},
    split::{input_target_split, train_test_split},
    transforms::{
        normalizers::{Normalizer, UnitGaussianNormalizer},
        subsample::subsample,
    },
};
use burn::{
    config::Config,
    data::dataset::{Dataset, DatasetError},
};
use ndarray::{ArrayD, Axis, IxDyn, concatenate, stack};
use std::{
    io::{Error, ErrorKind},
    path::{Path, PathBuf},
};

/// Navier-Stokes config on top of the shared `DatasetConfig`.
#[derive(Config, Debug)]
pub struct NavierStokesConfig {
    pub base: DatasetConfig,
    /// Number of leading snapshots given as input.
    #[config(default = 10)]
    pub t_in: usize,
    /// Number of snapshots after the input to predict.
    #[config(default = 40)]
    pub t_out: usize,
    /// Keep every `subsample_rate`-th point on both spatial axes.
    #[config(default = 1)]
    pub subsample_rate: usize,
}

impl HasBaseConfig for NavierStokesConfig {
    fn base(&self) -> &DatasetConfig {
        &self.base
    }
}

/// The normalizers fitted during loading. `x` encodes inputs; `y` decodes
/// predictions back to physical units.
pub struct NavierStokesNormalizers {
    pub x: UnitGaussianNormalizer,
    pub y: UnitGaussianNormalizer,
}

/// Dataset that repeats the input snapshots along time and appends
/// the (x, y, t) grid per item, instead of storing the repeated input.
pub struct NavierStokesDataset {
    x: ArrayD<f64>,    // [n, s, s, t_in]
    y: ArrayD<f64>,    // [n, s, s, t_out]
    grid: ArrayD<f64>, // [s, s, t_out, 3], channels (x, y, t)
}

impl NavierStokesDataset {
    /// `x`: `[n, s, s, t_in]` input snapshots. `y`: `[n, s, s, t_out]` targets.
    ///
    /// # Panics
    ///
    /// If either array isn't rank 4, the sample counts differ, the spatial
    /// axes aren't square or don't match between `x` and `y`.
    pub fn new(x: ArrayD<f64>, y: ArrayD<f64>) -> Self {
        assert_eq!(
            x.ndim(),
            4,
            "x must be [n, s, s, t_in], got {:?}",
            x.shape()
        );
        assert_eq!(
            y.ndim(),
            4,
            "y must be [n, s, s, t_out], got {:?}",
            y.shape()
        );
        assert_eq!(
            x.shape()[0],
            y.shape()[0],
            "x and y must have the same number of samples"
        );
        assert_eq!(
            x.shape()[1],
            x.shape()[2],
            "spatial grid must be square, got {:?}",
            &x.shape()[1..3]
        );
        assert_eq!(
            x.shape()[1..3],
            y.shape()[1..3],
            "x and y spatial shapes differ"
        );

        let s = x.shape()[1];
        let t_out = y.shape()[3];
        let (xx, yy, tt) = uniform_grid_3d(0.0, 1.0, s, t_out);
        let grid = stack(Axis(3), &[xx.view(), yy.view(), tt.view()])
            .expect("grid components share a shape")
            .into_dyn();

        Self { x, y, grid }
    }
}

impl Dataset<DataItem> for NavierStokesDataset {
    fn get(&self, index: usize) -> Result<DataItem, DatasetError> {
        if index >= self.len() {
            return Err(DatasetError::new(Error::new(
                ErrorKind::InvalidInput,
                format!("index {index} out of bounds (len {})", self.len()),
            )));
        }

        let x = self.x.index_axis(Axis(0), index); // [s, s, t_in]
        let (s, t_in) = (x.shape()[0], x.shape()[2]);
        let t_out = self.y.shape()[3];

        // [s, s, t_in] -> [s, s, 1, t_in] -> [s, s, t_out, t_in]
        let with_time = x
            .into_shape_with_order(IxDyn(&[s, s, 1, t_in]))
            .expect("insert time axis");
        let repeated = with_time
            .broadcast(IxDyn(&[s, s, t_out, t_in]))
            .expect("broadcast along time");

        let input = concatenate(Axis(3), &[self.grid.view(), repeated])
            .expect("grid and data share spatial/time extents");
        let target = self.y.index_axis(Axis(0), index).to_owned();

        Ok(DataItem { input, target })
    }

    fn len(&self) -> usize {
        self.x.shape()[0]
    }
}

/// Loads the 2D Navier-Stokes dataset from a `.mat` file into train/test
/// `NavierStokesDataset`s.
///
/// Reads field `u` (`[N, s, s, T]` vorticity trajectories), subsamples both
/// spatial axes by `config.subsample_rate`, takes the first `n_train` and last
/// `n_test` samples, and splits time into the first `t_in` snapshots (input)
/// and the next `t_out` (target).
///
/// Returns `(train_dataset, test_dataset, normalizers)`. `normalizers.y` is
/// what the caller needs to decode predictions back to physical units.
///
/// # Panics
///
/// If the file or `u` can't be read, `u` isn't `[N, s, s, T]`,
/// `t_in + t_out > T`, or `n_train + n_test > N`.
pub fn load_navier_stokes_uniform(
    path: &PathBuf,
    config: &NavierStokesConfig,
) -> (
    NavierStokesDataset,
    NavierStokesDataset,
    NavierStokesNormalizers,
) {
    let reader =
        MatFileReader::new(Path::new(path)).expect("failed to open Navier-Stokes .mat file");
    let u = reader.read_field("u").expect("failed to read field 'u'");

    navier_stokes_datasets(u, config)
}

/// Builds the datasets from an already-read `u`; the reader-independent part
/// of [`load_navier_stokes_uniform`].
pub(crate) fn navier_stokes_datasets(
    u: ArrayD<f64>,
    config: &NavierStokesConfig,
) -> (
    NavierStokesDataset,
    NavierStokesDataset,
    NavierStokesNormalizers,
) {
    assert_eq!(u.ndim(), 4, "u must be [N, s, s, T], got {:?}", u.shape());
    let (n_train, n_test) = (config.n_train(), config.n_test());
    let (t_in, t_out) = (config.t_in, config.t_out);

    // downsample both spatial axes (1 and 2); time is kept at full resolution
    let u = subsample(
        subsample(u, 1, config.subsample_rate),
        2,
        config.subsample_rate,
    );

    // first n_train / last n_test samples, then first t_in / next t_out snapshots
    let (u_train, u_test) = train_test_split(u, n_train, n_test);
    let (x_train, y_train) = input_target_split(&u_train, 3, t_in, t_out);
    let (x_test, y_test) = input_target_split(&u_test, 3, t_in, t_out);

    let x_normalizer = UnitGaussianNormalizer::fit(&x_train);
    let x_train = x_normalizer.encode(x_train);
    let x_test = x_normalizer.encode(x_test);

    let y_normalizer = UnitGaussianNormalizer::fit(&y_train);
    let y_train = y_normalizer.encode(y_train);
    // y_test intentionally NOT encoded

    println!(
        "navier-stokes: {n_train} train / {n_test} test at s={}, t_in={t_in}, t_out={t_out}",
        x_train.shape()[1]
    );

    (
        NavierStokesDataset::new(x_train, y_train),
        NavierStokesDataset::new(x_test, y_test),
        NavierStokesNormalizers {
            x: x_normalizer,
            y: y_normalizer,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::data::grids::append_grid_3d;
    use approx::assert_relative_eq;
    use ndarray::{Array3, s};

    /// Synthetic `u` whose values encode their index:
    /// `u[n, i, j, t] = 1000 n + 100 i + 10 j + t`.
    fn synthetic_u(n: usize, s: usize, t: usize) -> ArrayD<f64> {
        ArrayD::from_shape_fn(IxDyn(&[n, s, s, t]), |i| {
            (1000 * i[0] + 100 * i[1] + 10 * i[2] + i[3]) as f64
        })
    }

    fn config(n_train: usize, n_test: usize, t_in: usize, t_out: usize) -> NavierStokesConfig {
        NavierStokesConfig::new(DatasetConfig::new(n_train, n_test))
            .with_t_in(t_in)
            .with_t_out(t_out)
    }

    #[test]
    fn item_layout_is_grid_then_repeated_snapshots() {
        // n=2, s=3, t_in=2, t_out=4
        let x = ArrayD::from_shape_fn(IxDyn(&[2, 3, 3, 2]), |i| {
            (1000 * i[0] + 100 * i[1] + 10 * i[2] + i[3]) as f64
        });
        let y = ArrayD::from_shape_fn(IxDyn(&[2, 3, 3, 4]), |i| {
            -((1000 * i[0] + 100 * i[1] + 10 * i[2] + i[3]) as f64)
        });
        let ds = NavierStokesDataset::new(x, y);
        let item = ds.get(1).unwrap();

        assert_eq!(item.input.shape(), &[3, 3, 4, 5]);
        assert_eq!(item.target.shape(), &[3, 3, 4]);

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..4 {
                    // grid: x on [0, 1] along axis 0, y along axis 1, t on (0, 1]
                    assert_relative_eq!(item.input[[i, j, k, 0]], i as f64 / 2.0);
                    assert_relative_eq!(item.input[[i, j, k, 1]], j as f64 / 2.0);
                    assert_relative_eq!(item.input[[i, j, k, 2]], (k + 1) as f64 / 4.0);
                    // data: sample 1's snapshots, identical at every output time k
                    for c in 0..2 {
                        let expected = (1000 + 100 * i + 10 * j + c) as f64;
                        assert_eq!(item.input[[i, j, k, 3 + c]], expected);
                    }
                    assert_eq!(
                        item.target[[i, j, k]],
                        -((1000 + 100 * i + 10 * j + k) as f64)
                    );
                }
            }
        }
    }

    #[test]
    fn item_matches_eager_grid_append() {
        let x = ArrayD::from_shape_fn(IxDyn(&[1, 4, 4, 3]), |i| {
            (100 * i[1] + 10 * i[2] + i[3]) as f64
        });
        let y = ArrayD::zeros(IxDyn(&[1, 4, 4, 5]));
        let item = NavierStokesDataset::new(x.clone(), y).get(0).unwrap();

        // Eager path: repeat along time, then the shared grid helper.
        let repeated = x
            .into_shape_with_order(IxDyn(&[1, 4, 4, 1, 3]))
            .unwrap()
            .broadcast(IxDyn(&[1, 4, 4, 5, 3]))
            .unwrap()
            .to_owned();
        let (xx, yy, tt): (Array3<f64>, _, _) = uniform_grid_3d(0.0, 1.0, 4, 5);
        let eager = append_grid_3d(repeated, xx, yy, tt);

        assert_eq!(item.input, eager.index_axis(Axis(0), 0));
    }

    #[test]
    fn get_rejects_out_of_range_index() {
        let ds = NavierStokesDataset::new(
            ArrayD::zeros(IxDyn(&[2, 2, 2, 1])),
            ArrayD::zeros(IxDyn(&[2, 2, 2, 1])),
        );
        assert_eq!(ds.len(), 2);
        assert!(ds.get(2).is_err());
    }

    #[test]
    fn loader_splits_samples_and_time() {
        let u = synthetic_u(6, 4, 7);
        let (train, test, norms) = navier_stokes_datasets(u.clone(), &config(3, 2, 2, 4));

        assert_eq!(train.len(), 3);
        assert_eq!(test.len(), 2);

        // Test targets stay raw: last two samples (4, 5), times t_in..t_in+t_out.
        for (idx, n) in [(0, 4), (1, 5)] {
            let target = test.get(idx).unwrap().target;
            assert_eq!(target.shape(), &[4, 4, 4]);
            for k in 0..4 {
                assert_eq!(target[[1, 2, k]], (1000 * n + 100 + 20 + 2 + k) as f64);
            }
        }

        // Train targets decode back to the first three samples.
        let target = norms.y.decode(train.get(2).unwrap().target);
        let raw = u.slice(s![2, .., .., 2..6]).to_owned().into_dyn();
        for (a, b) in target.iter().zip(raw.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }

        // Test inputs are encoded with the train-fitted x normalizer.
        let input = test.get(1).unwrap().input;
        let data = input.slice(s![.., .., 0, 3..]).to_owned().into_dyn();
        let decoded = norms.x.decode(data);
        let raw = u.slice(s![5, .., .., 0..2]).to_owned().into_dyn();
        for (a, b) in decoded.iter().zip(raw.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6);
        }
    }

    #[test]
    fn loader_subsamples_space_only() {
        let cfg = config(1, 1, 2, 3).with_subsample_rate(2);
        let (_, test, _) = navier_stokes_datasets(synthetic_u(3, 4, 5), &cfg);

        let item = test.get(0).unwrap();
        assert_eq!(item.input.shape(), &[2, 2, 3, 5]);
        // Spatial index 1 after subsampling is raw index 2.
        assert_eq!(item.target[[1, 1, 0]], (2000 + 200 + 20 + 2) as f64);
    }

    #[test]
    #[should_panic(expected = "exceeds the 5 entries along axis 3")]
    fn loader_rejects_too_many_snapshots() {
        let _ = navier_stokes_datasets(synthetic_u(3, 2, 5), &config(1, 1, 2, 4));
    }

    #[test]
    #[should_panic(expected = "exceeds n_total (3)")]
    fn loader_rejects_too_many_samples() {
        let _ = navier_stokes_datasets(synthetic_u(3, 2, 5), &config(2, 2, 2, 2));
    }

    #[test]
    #[should_panic(expected = "failed to open Navier-Stokes .mat file")]
    fn loader_rejects_missing_file() {
        let _ =
            load_navier_stokes_uniform(&PathBuf::from("does_not_exist.mat"), &config(1, 1, 1, 1));
    }
}
