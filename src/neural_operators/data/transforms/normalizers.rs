//! Normalization strategies for operator-learning data: a shared `normalizer`
//! trait (fit / encode / decode) with three implementations differing in
//! what statistics they compute and over what scope.

use burn::{config::Config, prelude::*};
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
/// spatial points, all examples. No customization of `eps` - see gap noted
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

/// Rank-independent decode: (x * (std + eps)) + mean, on flattened
/// [batch, n_points] tensors. Tensor-native so autodiff traces through it
/// into the model, unlike UnitGaussiannormalizer::decode's ndarray version.
pub fn decode_flat(x: Tensor<2>, mean: &Tensor<1>, std: &Tensor<1>, eps: f64) -> Tensor<2> {
    x * (std.clone().unsqueeze::<2>() + eps) + mean.clone().unsqueeze::<2>()
}

/// Converts a fitted UnitGaussiannormalizer's mean/std into flat rank-1
/// Tensors, once, before training starts - not called per-batch.
pub fn normalizer_to_flat_tensors(
    normalizer: &UnitGaussianNormalizer,
    device: &Device,
) -> (Tensor<1>, Tensor<1>) {
    let mean_data: Vec<f64> = normalizer.mean_ref().iter().copied().collect();
    let std_data: Vec<f64> = normalizer.std_ref().iter().copied().collect();
    let n = mean_data.len();
    (
        Tensor::<1>::from_data(TensorData::new(mean_data, vec![n]), device),
        Tensor::<1>::from_data(TensorData::new(std_data, vec![n]), device),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_round_trips() {
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 3, 5]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64
        });
        let n = UnitGaussianNormalizer::fit(&data);
        let rebuilt = UnitGaussianNormalizer::from_record(&n.to_record());
        assert_eq!(n.encode(data.clone()), rebuilt.encode(data));
    }

    #[test]
    fn encode_decode_round_trips() {
        let data = ArrayD::from_shape_fn(IxDyn(&[4, 3, 5]), |i| {
            (i[0] * 100 + i[1] * 10 + i[2]) as f64
        });
        let n = UnitGaussianNormalizer::fit(&data);
        let restored = n.decode(n.encode(data.clone()));
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-9, "{a} != {b}");
        }
    }

    #[test]
    fn tensor_reshape_matches_ndarray_ordering() {
        let device = Device::default();

        // 0..12 in a [3, 4] ndarray - row-major, so element (i,j) = i*4 + j.
        let arr = ndarray::Array2::from_shape_fn((3, 4), |(i, j)| (i * 4 + j) as f64);
        let flat_nd: Vec<f64> = arr.iter().copied().collect();

        // Same values as a [1, 3, 4] tensor, reshaped to [1, 12].
        let t = Tensor::<3>::from_data(TensorData::new(flat_nd.clone(), vec![1, 3, 4]), &device);
        let flat_t: Vec<f64> = t.reshape([1, 12]).into_data().iter::<f64>().collect();

        assert_eq!(
            flat_nd, flat_t,
            "ndarray and Tensor flatten in different orders"
        );
    }

    #[test]
    fn decode_flat_matches_ndarray_decode() {
        let device = Device::default();

        // Non-square spatial dims - a transpose bug is invisible on square shapes.
        let (n, s1, s2) = (2, 3, 4);
        let raw = ndarray::ArrayD::from_shape_fn(ndarray::IxDyn(&[n, s1, s2]), |idx| {
            (idx[0] * 100 + idx[1] * 10 + idx[2]) as f64
        });

        let normalizer = UnitGaussianNormalizer::fit(&raw);
        let encoded = normalizer.encode(raw.clone());

        // Path A: ndarray decode, the reference implementation.
        let decoded_nd = normalizer.decode(encoded.clone());

        // Path B: Tensor-native decode on the flattened pair, as training uses.
        let (mean, std) = normalizer_to_flat_tensors(&normalizer, &device);
        let flat: Vec<f64> = encoded.iter().copied().collect();
        let t = Tensor::<2>::from_data(TensorData::new(flat, vec![n, s1 * s2]), &device);
        let decoded_flat = decode_flat(t, &mean, &std, normalizer.eps_val());

        let a: Vec<f64> = decoded_nd.iter().copied().collect();
        let b: Vec<f64> = decoded_flat.into_data().iter::<f64>().collect();

        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-4, "element {i}: ndarray {x} != flat {y}");
        }
    }
}
