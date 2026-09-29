//! Normalization strategies for operator-learning data: a shared `normalizer`
//! trait (fit / encode / decode) with three implementations differing in
//! what statistics they compute and over what scope.

use burn::{config::Config, prelude::*};
use ndarray::{ArrayD, Axis, IxDyn, Zip};

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
        let std = data.std_axis(Axis(0), 1.0);
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
/// spatial points, all examples. `eps` is fixed at `DEFAULT_EPS` (1e-5):
/// unlike [`UnitGaussianNormalizer::with_eps`], there is no constructor that
/// sets it.
#[derive(Clone)]
pub struct GaussianNormalizer {
    mean: f64,
    std: f64,
    eps: f64,
}

impl Normalizer for GaussianNormalizer {
    fn fit(data: &ArrayD<f64>) -> Self {
        let mean = data.mean().unwrap();
        let std = data.std(1.0);
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
/// A point that is constant across the batch (`max == min`, e.g. a Dirichlet
/// boundary pixel) has no range to rescale: it gets scale 1 and is shifted to
/// the midpoint `(low + high) / 2`, so `encode`/`decode` stay finite and exact
/// inverses. All other points use the plain min/max map.
#[derive(Clone)]
pub struct RangeNormalizer {
    a: ArrayD<f64>, // scale factor
    b: ArrayD<f64>, // offset
}

impl RangeNormalizer {
    /// # Panics
    ///
    /// If `low >= high`: `decode` would divide by a zero scale.
    pub fn with_range(data: &ArrayD<f64>, low: f64, high: f64) -> Self {
        assert!(
            low < high,
            "RangeNormalizer: need low < high, got [{low}, {high}]"
        );
        let min = data.map_axis(Axis(0), |row| row.fold(f64::INFINITY, |a, &b| a.min(b)));
        let max = data.map_axis(Axis(0), |row| row.fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        let mid = 0.5 * (low + high);

        // a = (high - low) / (max - min); b = -a * max + high. Constant points
        // would give a = inf and NaN on encode, so they get a = 1, b = mid - c.
        let a = Zip::from(&max).and(&min).map_collect(|&mx, &mn| {
            if mx == mn {
                1.0
            } else {
                (high - low) / (mx - mn)
            }
        });
        let b = Zip::from(&a)
            .and(&max)
            .and(&min)
            .map_collect(|&a, &mx, &mn| if mx == mn { mid - mx } else { -a * mx + high });

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

    // --- REVIEW.md 2.6: RangeNormalizer at constant points ---

    /// [3 samples, 4 points]: points 0 and 2 vary; point 1 is constant 2.5
    /// (old code: a = inf, b = -inf) and point 3 is constant 0.0 (old code:
    /// b = inf * 0 = NaN).
    fn range_data_with_constant_points() -> ArrayD<f64> {
        ArrayD::from_shape_vec(
            IxDyn(&[3, 4]),
            vec![
                -1.0, 2.5, 10.0, 0.0, //
                0.5, 2.5, 30.0, 0.0, //
                3.0, 2.5, 20.0, 0.0,
            ],
        )
        .unwrap()
    }

    #[test]
    fn range_constant_points_encode_to_midpoint_and_round_trip() {
        let data = range_data_with_constant_points();
        for (low, high) in [(0.0, 1.0), (-1.0, 1.0)] {
            let n = RangeNormalizer::with_range(&data, low, high);
            let enc = n.encode(data.clone());
            assert!(
                enc.iter().all(|v| v.is_finite()),
                "non-finite encode: {enc:?}"
            );

            let mid = 0.5 * (low + high);
            for sample in 0..3 {
                assert_eq!(enc[[sample, 1]], mid, "[{low}, {high}] point 1");
                assert_eq!(enc[[sample, 3]], mid, "[{low}, {high}] point 3");
            }

            let dec = n.decode(enc);
            for (a, b) in data.iter().zip(dec.iter()) {
                assert!((a - b).abs() < 1e-12, "{a} != {b}");
            }
        }
    }

    #[test]
    fn range_non_constant_points_are_unchanged() {
        // The guard only touches max == min points; the others must match the
        // unguarded formula to the bit, and span [low, high].
        let data = range_data_with_constant_points();
        let (low, high) = (-2.0, 3.0);
        let n = RangeNormalizer::with_range(&data, low, high);
        let enc = n.encode(data.clone());

        for point in [0, 2] {
            let col: Vec<f64> = (0..3).map(|s| data[[s, point]]).collect();
            let min = col.iter().copied().fold(f64::INFINITY, f64::min);
            let max = col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let a = (high - low) / (max - min);
            let b = -a * max + high;
            for (s, &x) in col.iter().enumerate() {
                assert_eq!(enc[[s, point]], a * x + b, "point {point}, sample {s}");
            }
            let enc_col: Vec<f64> = (0..3).map(|s| enc[[s, point]]).collect();
            let lo = enc_col.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = enc_col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            assert!((lo - low).abs() < 1e-12 && (hi - high).abs() < 1e-12);
        }
    }

    #[test]
    #[should_panic(expected = "low < high")]
    fn range_rejects_empty_target_interval() {
        let _ = RangeNormalizer::with_range(&range_data_with_constant_points(), 1.0, 1.0);
    }
}
