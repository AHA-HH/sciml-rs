//! normalization strategies for operator-learning data: a shared `normalizer`
//! trait (fit / encode / decode) with three implementations differing in
//! what statistics they compute and over what scope.

use burn::config::Config;
use ndarray::{ArrayD, Axis, IxDyn};

// Default guard added to standard deviations to avoid division by zero.
const DEFAULT_EPS: f64 = 1e-5;
/// Shared interface for all normalizers: fit on training data, encode inputs
/// before training, decode predictions back to physical scale.
pub trait Normalizer {
    /// Computes and stores statistics from `data`.
    fn fit(data: &ArrayD<f64>) -> Self;

    /// normalizes `data` using stored statistics.
    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64>;

    /// Denormalizes `data` back to physical scale.
    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64>;
}

/// Serialisable form of a fitted [`UnitGaussianNormalizer`]. `mean` and `std`
/// are flattened; `shape` restores them.
#[derive(Config, Debug)]
pub struct NormalizerRecord {
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
    pub shape: Vec<usize>,
    pub eps: f64,
}

/// Pointwise normalization: per-spatial-point mean/std, computed across the
/// batch axis. `mean`/`std` have the shape of a single example, not a scalar.
#[derive(Clone)]
pub struct UnitGaussianNormalizer {
    mean: ArrayD<f64>,
    std: ArrayD<f64>,
    eps: f64,
}

impl UnitGaussianNormalizer {
    /// Fits with a caller-specified `eps` instead of `fit`'s default.
    pub fn with_eps(data: &ArrayD<f64>, eps: f64) -> Self {
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 0.0);
        Self { mean, std, eps }
    }

    // Exposes fitted statistics for bridging into Burn's autodiff Tensor
    // world inside the training loop. encode/decode above stay ndarray-based
    // and are used during data loading (load_darcy_uniform), where no
    // gradient tracking is needed.
    pub fn mean_ref(&self) -> &ArrayD<f64> {
        &self.mean
    }

    pub fn std_ref(&self) -> &ArrayD<f64> {
        &self.std
    }

    pub fn eps_val(&self) -> f64 {
        self.eps
    }

    pub fn to_record(&self) -> NormalizerRecord {
        NormalizerRecord {
            mean: self.mean.iter().copied().collect(),
            std: self.std.iter().copied().collect(),
            shape: self.mean.shape().to_vec(),
            eps: self.eps,
        }
    }

    pub fn from_record(r: &NormalizerRecord) -> Self {
        let shape = IxDyn(&r.shape);
        Self {
            mean: ArrayD::from_shape_vec(shape.clone(), r.mean.clone())
                .expect("mean shape mismatch"),
            std: ArrayD::from_shape_vec(shape, r.std.clone()).expect("std shape mismatch"),
            eps: r.eps,
        }
    }
}

impl Normalizer for UnitGaussianNormalizer {
    fn fit(data: &ArrayD<f64>) -> Self {
        Self::with_eps(data, DEFAULT_EPS)
    }

    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data - &self.mean) / (&self.std + self.eps)
    }

    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data * (&self.std + self.eps)) + &self.mean
    }
}

/// Global normalization: single scalar mean/std across all values, all
/// spatial points, all examples. No customization of `eps` — see gap noted
/// above `fit`.
#[derive(Clone)]
pub struct GaussianNormalizer {
    mean: f64,
    std: f64,
    eps: f64,
}

impl Normalizer for GaussianNormalizer {
    fn fit(data: &ArrayD<f64>) -> Self {
        let mean = data.mean().unwrap();
        let std = data.std(0.0);
        Self {
            mean,
            std,
            eps: DEFAULT_EPS,
        }
    }

    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data - self.mean) / (self.std + self.eps)
    }

    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data * (self.std + self.eps)) + self.mean
    }
}

/// Linear rescaling to `[low, high]` (default `[0, 1]` via `fit`), computed
/// per spatial point from batch min/max.
///
/// No `eps` guard: a constant channel (`max == min`) produces a division by
/// zero in `with_range`, propagating `inf`/`NaN` through `encode` silently.
#[derive(Clone)]
pub struct RangeNormalizer {
    a: ArrayD<f64>, // scale factor
    b: ArrayD<f64>, // offset
}

impl RangeNormalizer {
    pub fn with_range(data: &ArrayD<f64>, low: f64, high: f64) -> Self {
        let min = data.map_axis(Axis(0), |row| row.fold(f64::INFINITY, |a, &b| a.min(b)));
        let max = data.map_axis(Axis(0), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        // a = (high - low) / (max - min); b = -a * max + high
        let a = (&max - &min).mapv(|x| (high - low) / x);
        let b = -&a * &max + high;

        Self { a, b }
    }
}

impl Normalizer for RangeNormalizer {
    fn fit(data: &ArrayD<f64>) -> Self {
        Self::with_range(data, 0.0, 1.0)
    }

    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        &self.a * &data + &self.b
    }

    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (&data - &self.b) / &self.a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizer_survives_record_round_trip() {
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 3, 5]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64
        });
        let n = UnitGaussianNormalizer::fit(&data);
        let rebuilt = UnitGaussianNormalizer::from_record(&n.to_record());
        assert_eq!(n.encode(data.clone()), rebuilt.encode(data));
    }
}
