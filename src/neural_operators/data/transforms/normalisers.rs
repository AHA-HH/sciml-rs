//! Normalisation strategies for operator-learning data: a shared `Normaliser`
//! trait (fit / encode / decode) with three implementations differing in
//! what statistics they compute and over what scope.

use ndarray::{ArrayD, Axis};

/// Shared interface for all normalisers: fit on training data, encode inputs
/// before training, decode predictions back to physical scale.
pub trait Normaliser {
    /// Computes and stores statistics from `data`.
    fn fit(data: &ArrayD<f64>) -> Self;

    /// Normalises `data` using stored statistics.
    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64>;

    /// Denormalises `data` back to physical scale.
    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64>;
}

/// Pointwise normalisation: per-spatial-point mean/std, computed across the
/// batch axis. `mean`/`std` have the shape of a single example, not a scalar.
#[derive(Clone)]
pub struct UnitGaussianNormaliser {
    mean: ArrayD<f64>,
    std: ArrayD<f64>,
    eps: f64,
}

impl UnitGaussianNormaliser {
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
}

impl Normaliser for UnitGaussianNormaliser {
    fn fit(data: &ArrayD<f64>) -> Self {
        Self::with_eps(data, 0.00001)
    }

    fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data - &self.mean) / (&self.std + self.eps)
    }

    fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data * (&self.std + self.eps)) + &self.mean
    }
}

/// Global normalisation: single scalar mean/std across all values, all
/// spatial points, all examples. No customization of `eps` — see gap noted
/// above `fit`.
#[derive(Clone)]
pub struct GaussianNormaliser {
    mean: f64,
    std: f64,
    eps: f64,
}

impl Normaliser for GaussianNormaliser {
    fn fit(data: &ArrayD<f64>) -> Self {
        let mean = data.mean().unwrap();
        let std = data.std(0.0);
        Self {
            mean,
            std,
            eps: 0.00001,
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
pub struct RangeNormaliser {
    a: ArrayD<f64>, // scale factor
    b: ArrayD<f64>, // offset
}

impl RangeNormaliser {
    pub fn with_range(data: &ArrayD<f64>, low: f64, high: f64) -> Self {
        let min = data.map_axis(Axis(0), |row| row.fold(f64::INFINITY, |a, &b| a.min(b)));
        let max = data.map_axis(Axis(0), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));

        // a = (high - low) / (max - min); b = -a * max + high
        let a = (&max - &min).mapv(|x| (high - low) / x);
        let b = -&a * &max + high;

        Self { a, b }
    }
}

impl Normaliser for RangeNormaliser {
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
