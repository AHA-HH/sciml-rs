//! Pointwise Gaussian normalization for operator-learning data
//! ([`UnitGaussianNormalizer`], CONVENTIONS §10), its serialisable
//! [`NormalizerRecord`], and the tensor-side [`FlatDecoder`].

use burn::{config::Config, prelude::*};
use ndarray::{ArrayD, Axis, IxDyn};

// Default guard added to standard deviations to avoid division by zero.
const DEFAULT_EPS: f64 = 1e-5;

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
    /// Fits on `data` with the default `eps` (1e-5).
    ///
    /// # Panics
    ///
    /// As [`with_eps`](Self::with_eps): if `data` has fewer than 2 samples
    /// along axis 0.
    pub fn fit(data: &ArrayD<f64>) -> Self {
        Self::with_eps(data, DEFAULT_EPS)
    }

    /// Fits with a caller-specified `eps` instead of `fit`'s default.
    ///
    /// # Panics
    ///
    /// If `data` has fewer than 2 samples along axis 0: the sample std
    /// (`ddof = 1`) is 0/0 = NaN for one sample, which `eps` can't guard.
    pub fn with_eps(data: &ArrayD<f64>, eps: f64) -> Self {
        let n = data.shape().first().copied().unwrap_or(0);
        assert!(
            n >= 2,
            "UnitGaussianNormalizer: a sample std needs at least 2 samples along axis 0, got {n}"
        );
        let mean = data.mean_axis(Axis(0)).unwrap();
        let std = data.std_axis(Axis(0), 1.0);
        Self { mean, std, eps }
    }

    /// Normalizes `data` (`[batch, ...]`, each example shaped like `mean`)
    /// using the fitted statistics.
    pub fn encode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data - &self.mean) / (&self.std + self.eps)
    }

    /// Denormalizes `data` (`[batch, ...]`, each example shaped like `mean`)
    /// back to physical scale.
    pub fn decode(&self, data: ArrayD<f64>) -> ArrayD<f64> {
        (data * (&self.std + self.eps)) + &self.mean
    }

    /// Fitted per-point mean, shaped like one example. With [`std`](Self::std)
    /// and [`eps`](Self::eps) it bridges into Burn tensors ([`FlatDecoder`]);
    /// `encode`/`decode` stay ndarray-based for data loading, where no
    /// gradient tracking is needed.
    pub fn mean(&self) -> &ArrayD<f64> {
        &self.mean
    }

    /// Fitted per-point sample std (`ddof = 1`), shaped like one example.
    pub fn std(&self) -> &ArrayD<f64> {
        &self.std
    }

    /// Guard added to `std` in `encode`/`decode`.
    pub fn eps(&self) -> f64 {
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

/// Decodes flattened `[batch, n_points]` predictions back to physical scale:
/// `x * (std + eps) + mean`.
///
/// Built once per device from a fitted [`UnitGaussianNormalizer`]; `std + eps`
/// is computed here, once, rather than on every batch. Tensor-native, so
/// autodiff traces through [`decode`](Self::decode) into the model, unlike the
/// ndarray [`UnitGaussianNormalizer::decode`].
#[derive(Clone, Debug)]
pub struct FlatDecoder {
    mean: Tensor<1>,
    /// `std + eps`, `[n_points]`.
    scale: Tensor<1>,
}

impl FlatDecoder {
    /// Uploads the normalizer's mean and std to `device` (converted to its
    /// default float dtype) and adds `eps` on the device, the same operation
    /// the per-batch decode used to do, so results are unchanged.
    pub fn new(normalizer: &UnitGaussianNormalizer, device: &Device) -> Self {
        let upload = |a: &ArrayD<f64>| {
            let values: Vec<f64> = a.iter().copied().collect();
            let n = values.len();
            Tensor::<1>::from_data(TensorData::new(values, vec![n]), device)
        };
        Self {
            mean: upload(normalizer.mean()),
            scale: upload(normalizer.std()) + normalizer.eps(),
        }
    }

    /// The same decoder on the inner (non-autodiff) device, for evaluating a model taken
    /// with `valid()`.
    pub fn inner(&self) -> Self {
        Self {
            mean: self.mean.clone().inner(),
            scale: self.scale.clone().inner(),
        }
    }

    /// `x * (std + eps) + mean`, broadcast over the batch axis.
    pub fn decode(&self, x: Tensor<2>) -> Tensor<2> {
        x * self.scale.clone().unsqueeze::<2>() + self.mean.clone().unsqueeze::<2>()
    }
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
    fn flat_decoder_matches_ndarray_decode() {
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
        let flat: Vec<f64> = encoded.iter().copied().collect();
        let t = Tensor::<2>::from_data(TensorData::new(flat, vec![n, s1 * s2]), &device);
        let decoded_flat = FlatDecoder::new(&normalizer, &device).decode(t);

        let a: Vec<f64> = decoded_nd.iter().copied().collect();
        let b: Vec<f64> = decoded_flat.into_data().iter::<f64>().collect();

        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() < 1e-4, "element {i}: ndarray {x} != flat {y}");
        }
    }

    // --- REVIEW.md 4.9: std + eps precomputed once, results unchanged ---

    /// The pre-4.9 `decode_flat` body: `std + eps` recomputed per call.
    fn per_call_decode(x: Tensor<2>, mean: &Tensor<1>, std: &Tensor<1>, eps: f64) -> Tensor<2> {
        x * (std.clone().unsqueeze::<2>() + eps) + mean.clone().unsqueeze::<2>()
    }

    /// A normalizer whose std values are not exact in f32, with an eps whose
    /// sum with them rounds, plus the old path's uploaded mean/std.
    fn awkward_normalizer(device: &Device) -> (UnitGaussianNormalizer, Tensor<1>, Tensor<1>) {
        let data = ArrayD::from_shape_fn(IxDyn(&[5, 7]), |i| {
            ((i[0] * 7 + i[1]) as f64 * 0.731).sin() / 3.0 + i[1] as f64
        });
        let n = UnitGaussianNormalizer::with_eps(&data, 1.3e-5);
        let up = |a: &ArrayD<f64>| {
            let v: Vec<f64> = a.iter().copied().collect();
            let len = v.len();
            Tensor::<1>::from_data(TensorData::new(v, vec![len]), device)
        };
        let (mean, std) = (up(n.mean()), up(n.std()));
        (n, mean, std)
    }

    fn bits(t: Tensor<2>) -> Vec<u32> {
        t.into_data().iter::<f32>().map(f32::to_bits).collect()
    }

    #[test]
    fn flat_decoder_bit_identical_to_per_call_formula() {
        for device in [Device::default(), Device::default().autodiff()] {
            let (n, mean, std) = awkward_normalizer(&device);
            let x = Tensor::<2>::from_data(
                TensorData::new(
                    (0..21)
                        .map(|i| (i as f32 * 0.37).cos() * 2.5)
                        .collect::<Vec<_>>(),
                    vec![3, 7],
                ),
                &device,
            );
            let new = FlatDecoder::new(&n, &device).decode(x.clone());
            let old = per_call_decode(x, &mean, &std, n.eps());
            assert_eq!(new.is_autodiff(), device.is_autodiff());
            assert_eq!(bits(new), bits(old), "autodiff = {}", device.is_autodiff());
        }
    }

    #[test]
    fn flat_decoder_gradient_matches() {
        let device = Device::default().autodiff();
        let (n, mean, std) = awkward_normalizer(&device);
        let x0 = Tensor::<2>::from_data(
            TensorData::new(
                (0..14).map(|i| (i as f32 * 0.53).sin()).collect::<Vec<_>>(),
                vec![2, 7],
            ),
            &device,
        );
        let probe = Tensor::<2>::from_data(
            TensorData::new(
                (0..14).map(|i| 1.0 + i as f32 * 0.1).collect::<Vec<_>>(),
                vec![2, 7],
            ),
            &device,
        );
        let grad = |f: &dyn Fn(Tensor<2>) -> Tensor<2>| {
            let x = x0.clone().require_grad();
            let g = (f(x.clone()) * probe.clone()).sum().backward();
            bits(x.grad(&g).expect("input gradient"))
        };
        let decoder = FlatDecoder::new(&n, &device);
        let eps = n.eps();
        assert_eq!(
            grad(&|x| decoder.decode(x)),
            grad(&|x| per_call_decode(x, &mean, &std, eps))
        );
    }

    // --- REVIEW1.md N1: one-sample fits gave NaN std under ddof = 1 ---

    #[test]
    #[should_panic(expected = "at least 2")]
    fn unit_gaussian_rejects_single_sample() {
        let data = ArrayD::from_shape_vec(IxDyn(&[1, 3]), vec![1.0, 3.0, 5.0]).unwrap();
        let _ = UnitGaussianNormalizer::fit(&data);
    }

    #[test]
    fn unit_gaussian_two_samples_uses_sample_std() {
        // columns [1, 3] and [3, 7]: ddof = 1 gives [√2, 2√2], ddof = 0 [1, 2]
        let data = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 3.0, 3.0, 7.0]).unwrap();
        let n = UnitGaussianNormalizer::fit(&data);
        assert_eq!(n.mean().as_slice().unwrap(), &[2.0, 5.0]);
        let std = n.std().as_slice().unwrap();
        let expected = [2f64.sqrt(), 2.0 * 2f64.sqrt()];
        for (a, b) in std.iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
        assert!(n.encode(data).iter().all(|v| v.is_finite()));
    }
}
