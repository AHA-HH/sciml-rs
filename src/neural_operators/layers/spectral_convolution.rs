//! Spectral convolution, generic over spatial dimensionality.
//!
//! The core operation of a Fourier Neural Operator: transform to the frequency
//! domain, apply a learned linear map to the channel axis at each retained
//! frequency, and transform back. Because the map is applied per-frequency
//! rather than per-grid-point, the layer is a global convolution and is
//! independent of the input discretisation - the same weights apply at any
//! resolution.
//!
//! # Rank
//!
//! `R` is the total tensor rank: `R = D + 2` for `D` spatial axes, plus the
//! batch and channel axes. Burn encodes rank in the tensor *type*, so it must
//! be a compile-time constant. Mode counts, by contrast, are ordinary runtime
//! data (`modes: Vec<usize>`); `new` asserts the two agree.

use burn::{
    Tensor,
    module::{Module, Param},
    tensor::signal,
    tensor::{Device, Distribution},
};

use std::ops::Range;

use crate::neural_operators::utils::fft::icfft_full_spectrum;

/// Learned spectral convolution over `R - 2` spatial axes.
#[derive(Module, Debug)]
pub struct SpectralConv<const R: usize> {
    /// One weight per mode corner; `len() == 2^(D-1)`.
    /// Each is `[in_channels, out_channels, modes[0], .., modes[D-1]]`.
    weights_re: Vec<Param<Tensor<R>>>,
    weights_im: Vec<Param<Tensor<R>>>,
    /// Retained frequency count per spatial axis; `len() == R - 2`.
    modes: Vec<usize>,
}

impl<const R: usize> SpectralConv<R> {
    /// # Panics
    ///
    /// If `modes.len() != R - 2`.
    pub fn new(device: &Device, in_channels: usize, out_channels: usize, modes: &[usize]) -> Self {
        assert_eq!(
            modes.len(),
            R - 2,
            "modes must have one entry per spatial axis (R - 2)"
        );
        let modes = modes.to_vec();
        let num_corners = 1usize << (modes.len() - 1);

        let mut shape = vec![in_channels, out_channels];
        shape.extend_from_slice(&modes);
        let shape: [usize; R] = shape.try_into().unwrap();

        // Scaling follows the reference implementation.
        let scale = 1.0 / (in_channels * out_channels) as f64;
        let mut weights_re = Vec::with_capacity(num_corners);
        let mut weights_im = Vec::with_capacity(num_corners);
        for _ in 0..num_corners {
            weights_re.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                Distribution::Uniform(0.0, scale),
                device,
            )));
            weights_im.push(Param::from_tensor(Tensor::<R>::random(
                shape,
                Distribution::Uniform(0.0, scale),
                device,
            )));
        }

        Self {
            weights_re,
            weights_im,
            modes,
        }
    }

    /// Spatial-axis slices selecting one mode corner.
    ///
    /// Bit `j` of `mask` picks the low (`0`) or high (`1`) frequency block on
    /// axis `j`. The last axis is always low: the real FFT leaves it with only
    /// non-negative frequencies, so it has no mirror to select. This is why
    /// `mask` spans `2^(D-1)` values rather than `2^D`.
    fn corner_ranges(mask: usize, modes: &[usize], full_dims: &[usize]) -> Vec<Range<usize>> {
        let d = modes.len();
        (0..d)
            .map(|axis| {
                if axis == d - 1 || (mask >> axis) & 1 == 0 {
                    0..modes[axis]
                } else {
                    full_dims[axis] - modes[axis]..full_dims[axis]
                }
            })
            .collect()
    }

    /// Complex channel mixing at every retained frequency.
    ///
    /// `(a + bi)(c + di) = (ac - bd) + (ad + bc)i`, batched over modes.
    ///
    /// The mode axes are flattened to a single axis `M` so the contraction is a
    /// plain batched matmul `[M, B, I] @ [M, I, O]`. This keeps every
    /// intermediate at literal rank 3 - the broadcast-and-sum alternative would
    /// need rank `R + 1`, which stable Rust cannot express as a type.
    fn complex_multiplication(
        x_re: Tensor<R>,
        x_im: Tensor<R>, // [B, I, modes..]
        w_re: Tensor<R>,
        w_im: Tensor<R>, // [I, O, modes..]
        modes: &[usize],
    ) -> (Tensor<R>, Tensor<R>) {
        let (b, i) = (x_re.dims()[0], x_re.dims()[1]);
        let o = w_re.dims()[1];
        let flat: usize = modes.iter().product();

        let x_re = x_re.reshape([b, i, flat]).permute([2, 0, 1]);
        let x_im = x_im.reshape([b, i, flat]).permute([2, 0, 1]);
        let w_re = w_re.reshape([i, o, flat]).permute([2, 0, 1]);
        let w_im = w_im.reshape([i, o, flat]).permute([2, 0, 1]);

        let out_re = x_re.clone().matmul(w_re.clone()) - x_im.clone().matmul(w_im.clone());
        let out_im = x_re.matmul(w_im) + x_im.matmul(w_re);

        let mut out_shape = vec![b, o];
        out_shape.extend_from_slice(modes);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        (
            out_re.permute([1, 2, 0]).reshape(out_shape),
            out_im.permute([1, 2, 0]).reshape(out_shape),
        )
    }

    /// Forward N-d transform: real FFT on the last spatial axis, complex FFT on
    /// the rest. Halves the storage and the corner count versus a full complex
    /// transform, since a real signal's spectrum is Hermitian.
    fn fft_ctensor(x: Tensor<R>) -> (Tensor<R>, Tensor<R>) {
        let dims = x.dims(); // read before x is consumed
        let last = R - 1;
        let (mut re, mut im) = signal::rfft(x, last, Some(dims[last]));
        for axis in (2..last).rev() {
            let (r, i) = signal::cfft(re, im, axis, Some(dims[axis]));
            re = r;
            im = i;
        }
        (re, im)
    }

    /// Inverse of [`fft_nd`], applying the axes in the reverse order.
    fn ifft_ctensor(mut re: Tensor<R>, mut im: Tensor<R>, orig_dims: &[usize]) -> Tensor<R> {
        let last = R - 1;
        for (axis, &dim) in orig_dims.iter().enumerate().take(last).skip(2) {
            let (r, i) = icfft_full_spectrum(re, im, axis, dim);
            re = r;
            im = i;
        }
        signal::irfft(re, im, last, Some(orig_dims[last]))
    }

    /// `[B, in_channels, spatial..] -> [B, out_channels, spatial..]`.
    ///
    /// Spatial extents are preserved; the channel count changes.
    pub fn forward(&self, x: Tensor<R>) -> Tensor<R> {
        // Captured before the transform: irfft needs the original extent to
        // recover the correct length from the truncated half-spectrum.
        let orig_dims = x.dims();
        let (batch, in_ch) = (orig_dims[0], orig_dims[1]);
        let out_ch = self.weights_re[0].val().dims()[1];
        let num_corners = self.weights_re.len();

        let (x_ft_re, x_ft_im) = Self::fft_ctensor(x);
        // Post-transform extents - the last axis is now n/2 + 1.
        let spec_dims: Vec<usize> = x_ft_re.dims()[2..].to_vec();

        let mut out_shape = vec![batch, out_ch];
        out_shape.extend_from_slice(&spec_dims);
        let out_shape: [usize; R] = out_shape.try_into().unwrap();

        // Frequencies outside the retained modes stay zero - this is the
        // low-pass truncation that makes the operator resolution-independent.
        let mut out_ft_re = Tensor::<R>::zeros(out_shape, &x_ft_re.device());
        let mut out_ft_im = Tensor::<R>::zeros(out_shape, &x_ft_im.device());

        for corner in 0..num_corners {
            let ranges = Self::corner_ranges(corner, &self.modes, &spec_dims);

            let mut in_ranges = vec![0..batch, 0..in_ch];
            in_ranges.extend(ranges.clone());
            let in_ranges: [Range<usize>; R] = in_ranges.try_into().unwrap();

            let x_re = x_ft_re.clone().slice(in_ranges.clone());
            let x_im = x_ft_im.clone().slice(in_ranges);

            let (o_re, o_im) = Self::complex_multiplication(
                x_re,
                x_im,
                self.weights_re[corner].val(),
                self.weights_im[corner].val(),
                &self.modes,
            );

            let mut out_ranges = vec![0..batch, 0..out_ch];
            out_ranges.extend(ranges);
            let out_ranges: [Range<usize>; R] = out_ranges.try_into().unwrap();

            out_ft_re = out_ft_re.slice_assign(out_ranges.clone(), o_re);
            out_ft_im = out_ft_im.slice_assign(out_ranges, o_im);
        }

        Self::ifft_ctensor(out_ft_re, out_ft_im, &orig_dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    #[test]
    fn corner_ranges_2d_matches_manual_indices() {
        let modes = vec![4, 3];
        let full_dims = vec![16, 9];

        // Last axis is always the low block — rfft leaves it non-negative only.
        assert_eq!(
            SpectralConv::<4>::corner_ranges(0, &modes, &full_dims),
            vec![0..4, 0..3],
        );
        assert_eq!(
            SpectralConv::<4>::corner_ranges(1, &modes, &full_dims),
            vec![12..16, 0..3],
        );
    }

    #[test]
    fn corner_ranges_3d_covers_all_four_combinations() {
        let modes = vec![4, 3, 2];
        let full_dims = vec![16, 12, 5];
        let expected = [
            vec![0..4, 0..3, 0..2],
            vec![12..16, 0..3, 0..2],
            vec![0..4, 9..12, 0..2],
            vec![12..16, 9..12, 0..2],
        ];
        for (corner, expected) in expected.iter().enumerate() {
            assert_eq!(
                &SpectralConv::<5>::corner_ranges(corner, &modes, &full_dims),
                expected,
                "corner {corner} mismatch",
            );
        }
    }

    #[test]
    fn complex_multiplication_matches_python_einsum() {
        let device = Device::default();

        let x_re = Tensor::<3>::from_data([[[1.0f32, 2.0], [3.0, 4.0]]], &device);
        let x_im = Tensor::<3>::from_data([[[0.5f32, 1.5], [2.5, 3.5]]], &device);
        let w_re = Tensor::<3>::from_data(
            [[[1.0f32, 0.5], [0.5, 1.0]], [[0.5, 1.5], [1.5, 0.5]]],
            &device,
        );
        let w_im = Tensor::<3>::from_data(
            [[[0.1f32, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            &device,
        );

        let (out_re, out_im) =
            SpectralConv::<3>::complex_multiplication(x_re, x_im, w_re, w_im, &[2]);

        let re_vec = out_re.into_data().try_to_vec::<f32>().unwrap();
        let im_vec = out_im.into_data().try_to_vec::<f32>().unwrap();

        let expected_re = [1.2f32, 4.6, 3.1, 0.6];
        let expected_im = [3.35f32, 8.8, 6.4, 7.25];

        for (i, (r, e)) in re_vec.iter().zip(expected_re.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "re[{i}]: got {r}, expected {e}");
        }
        for (i, (r, e)) in im_vec.iter().zip(expected_im.iter()).enumerate() {
            assert!((r - e).abs() < 1e-4, "im[{i}]: got {r}, expected {e}");
        }
    }

    fn round_trip_check<const R: usize>(shape: [usize; R]) {
        let device = Device::default();
        let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let orig_dims = x.dims();

        let (re, im) = SpectralConv::<R>::fft_ctensor(x.clone());
        let recon = SpectralConv::<R>::ifft_ctensor(re, im, &orig_dims);

        let x_vec = x.into_data().try_to_vec::<f32>().unwrap();
        let recon_vec = recon.into_data().try_to_vec::<f32>().unwrap();

        let max_diff = x_vec
            .iter()
            .zip(recon_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        assert!(max_diff < 1e-4, "round-trip diverged: max_diff={max_diff}");
    }

    #[test]
    fn fft_round_trip_1d() {
        round_trip_check::<3>([2, 4, 16]);
    }

    #[test]
    fn fft_round_trip_2d() {
        round_trip_check::<4>([2, 4, 16, 16]);
    }

    #[test]
    fn fft_round_trip_3d() {
        round_trip_check::<5>([2, 4, 8, 8, 8]);
    }

    #[test]
    fn forward_2d_matches_pytorch_reference() {
        let device = Device::default();
        let modes = vec![1, 1];

        // corner 0 = low-H (Python's weights1), corner 1 = high-H (weights2).
        let mut conv = SpectralConv::<4>::new(&device, 1, 1, &modes);
        conv.weights_re[0] = Param::from_tensor(Tensor::<4>::from_data([[[[0.1]]]], &device));
        conv.weights_im[0] = Param::from_tensor(Tensor::<4>::from_data([[[[0.2]]]], &device));
        conv.weights_re[1] = Param::from_tensor(Tensor::<4>::from_data([[[[0.3]]]], &device));
        conv.weights_im[1] = Param::from_tensor(Tensor::<4>::from_data([[[[0.4]]]], &device));

        let x = Tensor::<4>::from_data(
            [[[
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [13.0, 14.0, 15.0, 16.0],
            ]]],
            &device,
        );

        let out_vals = conv.forward(x).into_data().try_to_vec::<f32>().unwrap();

        // Pinned against SpectralConv2d.forward, weights1 = 0.1+0.2i, weights2 = 0.3+0.4i.
        let expected = [
            1.05, 1.05, 1.05, 1.05, -0.55, -0.55, -0.55, -0.55, 0.65, 0.65, 0.65, 0.65, 2.25, 2.25,
            2.25, 2.25,
        ];

        for (i, (a, e)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
        }
    }

    #[test]
    fn forward_3d_matches_numpy_reference() {
        let device = Device::default();
        let modes = vec![1, 1, 1];

        // corner 0 = (low H1, low H2), 1 = (high H1, low H2),
        // 2 = (low H1, high H2), 3 = (high, high).
        let mut conv = SpectralConv::<5>::new(&device, 1, 1, &modes);
        let pairs = [(0.1, 0.2), (0.3, 0.4), (0.5, 0.6), (0.7, 0.8)];
        for (i, (re, im)) in pairs.iter().enumerate() {
            conv.weights_re[i] = Param::from_tensor(Tensor::<5>::from_data([[[[[*re]]]]], &device));
            conv.weights_im[i] = Param::from_tensor(Tensor::<5>::from_data([[[[[*im]]]]], &device));
        }

        let x = Tensor::<5>::from_data(
            [[[
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0],
                    [13.0, 14.0, 15.0, 16.0],
                ],
                [
                    [17.0, 18.0, 19.0, 20.0],
                    [21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0],
                    [29.0, 30.0, 31.0, 32.0],
                ],
                [
                    [33.0, 34.0, 35.0, 36.0],
                    [37.0, 38.0, 39.0, 40.0],
                    [41.0, 42.0, 43.0, 44.0],
                    [45.0, 46.0, 47.0, 48.0],
                ],
                [
                    [49.0, 50.0, 51.0, 52.0],
                    [53.0, 54.0, 55.0, 56.0],
                    [57.0, 58.0, 59.0, 60.0],
                    [61.0, 62.0, 63.0, 64.0],
                ],
            ]]],
            &device,
        );

        let out_vals = conv.forward(x).into_data().try_to_vec::<f32>().unwrap();

        let expected = [
            4.25, 4.25, 4.25, 4.25, 1.85, 1.85, 1.85, 1.85, 3.85, 3.85, 3.85, 3.85, 6.25, 6.25,
            6.25, 6.25, -2.15, -2.15, -2.15, -2.15, -4.55, -4.55, -4.55, -4.55, -2.55, -2.55,
            -2.55, -2.55, -0.15, -0.15, -0.15, -0.15, 2.65, 2.65, 2.65, 2.65, 0.25, 0.25, 0.25,
            0.25, 2.25, 2.25, 2.25, 2.25, 4.65, 4.65, 4.65, 4.65, 9.05, 9.05, 9.05, 9.05, 6.65,
            6.65, 6.65, 6.65, 8.65, 8.65, 8.65, 8.65, 11.05, 11.05, 11.05, 11.05,
        ];

        for (i, (a, e)) in out_vals.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-4, "mismatch at index {i}: {a} vs {e}");
        }
    }

    fn gradient_flow_check<const R: usize>(modes: Vec<usize>, x_shape: [usize; R]) {
        let device = Device::default().autodiff();
        let channels = x_shape[1];

        let conv = SpectralConv::<R>::new(&device, channels, channels, &modes);
        let x =
            Tensor::<R>::random(x_shape, Distribution::Normal(0.0, 1.0), &device).require_grad();

        let out = conv.forward(x.clone());
        let grads = out.powf_scalar(2.0).sum().backward();

        assert!(
            x.grad(&grads).is_some(),
            "gradients did not flow through SpectralConv<{R}>"
        );

        for (idx, w) in conv.weights_re.iter().enumerate() {
            let g = w
                .grad(&grads)
                .unwrap_or_else(|| panic!("weights_re[{idx}] got no gradient"));
            let sum: f32 = g.abs().sum().into_scalar();
            assert!(sum > 1e-9, "weights_re[{idx}] gradient is effectively zero");
        }
        for (idx, w) in conv.weights_im.iter().enumerate() {
            let g = w
                .grad(&grads)
                .unwrap_or_else(|| panic!("weights_im[{idx}] got no gradient"));
            let sum: f32 = g.abs().sum().into_scalar();
            assert!(sum > 1e-9, "weights_im[{idx}] gradient is effectively zero");
        }
    }

    #[test]
    fn gradients_flow_1d() {
        gradient_flow_check::<3>(vec![4], [2, 4, 16]);
    }

    #[test]
    fn gradients_flow_2d() {
        gradient_flow_check::<4>(vec![2, 2], [1, 1, 4, 4]);
    }

    #[test]
    fn gradients_flow_3d() {
        gradient_flow_check::<5>(vec![2, 2, 2], [1, 1, 4, 4, 4]);
    }

    // --- regression: output must carry out_channels, not in_channels ---

    #[test]
    fn forward_shape_with_asymmetric_channels() {
        let device = Device::default();
        let (in_channels, out_channels) = (2, 3);
        let modes = vec![2, 2, 2];
        let (batch, h1, h2, m) = (1, 8, 8, 8);

        let conv = SpectralConv::<5>::new(&device, in_channels, out_channels, &modes);
        let x = Tensor::<5>::random(
            [batch, in_channels, h1, h2, m],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        assert_eq!(
            conv.forward(x).dims(),
            [batch, out_channels, h1, h2, m],
            "output shape should carry out_channels — regression test for the zeros-shape bug"
        );
    }
}
