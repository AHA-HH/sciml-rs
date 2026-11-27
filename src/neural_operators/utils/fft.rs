//! The FFTs of the spectral layers (CONVENTIONS §4): the N-d real transform over the spatial
//! axes and its inverses, and the inverse complex FFT (icfft) Burn does not provide.

use burn::tensor::{Tensor, signal};

/// Inverse complex FFT for a general (possibly non-Hermitian) spectrum, at
/// arbitrary rank `D`, via `conj(ifft(x)) = fft(conj(x)) / n`: negate the
/// imaginary part, run forward `cfft`, negate and scale the result back.
pub fn icfft_full_spectrum<const D: usize>(
    re: Tensor<D>,
    im: Tensor<D>,
    dim: usize,
    n: usize,
) -> (Tensor<D>, Tensor<D>) {
    // conj(x) = (re, -im)
    let conj_im = im.neg();

    let (fwd_re, fwd_im) = signal::cfft(re, conj_im, dim, Some(n));

    // conj(result) = (fwd_re, -fwd_im), scaled by 1/n
    let scale = 1.0 / n as f64;
    let out_re = fwd_re.mul_scalar(scale);
    let out_im = fwd_im.neg().mul_scalar(scale);

    (out_re, out_im)
}

/// Forward N-d transform: real FFT on the last spatial axis, complex FFT on
/// the rest. Halves the storage and the corner count versus a full complex
/// transform, since a real signal's spectrum is Hermitian.
///
/// Each complex FFT runs on the last axis, with its axis swapped there and
/// back (CONVENTIONS §4 is unchanged). On the fused GPU backends, Burn's
/// Bluestein `cfft` along a non-last axis of non-power-of-two length is
/// wrong when the trailing axes' product is even: the fused `cat` ignores
/// broadcasting on the vectorised axis (`upstream/burn/cfft-kernel/`).
///
/// `x`: `[B, C, s..]`; returns `(re, im)`, each `[B, C, s.., s_last / 2 + 1]`.
pub(crate) fn fft_ctensor<const R: usize>(x: Tensor<R>) -> (Tensor<R>, Tensor<R>) {
    let dims = x.dims(); // read before x is consumed
    let last = R - 1;
    let (mut re, mut im) = signal::rfft(x, last, Some(dims[last]));
    for axis in (2..last).rev() {
        let (r, i) = signal::cfft(
            re.swap_dims(axis, last),
            im.swap_dims(axis, last),
            last,
            Some(dims[axis]),
        );
        re = r.swap_dims(axis, last);
        im = i.swap_dims(axis, last);
    }
    (re, im)
}

/// Inverse of [`fft_ctensor`], applying the axes in the reverse order,
/// each complex inverse on the last axis as there. `re`, `im`: the spectrum of a
/// `orig_dims` (`[B, C, s..]`) field, as [`fft_ctensor`] returns it; returns `[B, C, s..]`.
pub(crate) fn ifft_ctensor<const R: usize>(
    mut re: Tensor<R>,
    mut im: Tensor<R>,
    orig_dims: &[usize],
) -> Tensor<R> {
    let last = R - 1;
    for (axis, &dim) in orig_dims.iter().enumerate().take(last).skip(2) {
        let (r, i) = icfft_full_spectrum(
            re.swap_dims(axis, last),
            im.swap_dims(axis, last),
            last,
            dim,
        );
        re = r.swap_dims(axis, last);
        im = i.swap_dims(axis, last);
    }
    signal::irfft(re, im, last, Some(orig_dims[last]))
}

/// [`ifft_ctensor`] to `out_dims` (`[B, C, out..]`) of a spectrum transformed at `in_dims`
/// (`[B, C, s..]`) and already resized to `out_dims`'s spectral extents, scaled by
/// `N_out / N_in` (products of the spatial extents): 2.0.0's `norm="forward"` amplitude, an
/// extension of CONVENTIONS §4's backward norm (library CM1.5b). The resize itself is the
/// caller's: [`SpectralConv::forward_to`] trims or pads at the end of each axis, `resample`
/// keeps each axis's lowest modes.
///
/// [`SpectralConv::forward_to`]: crate::neural_operators::layers::spectral_convolution::SpectralConv::forward_to
pub(crate) fn ifft_ctensor_to<const R: usize>(
    re: Tensor<R>,
    im: Tensor<R>,
    in_dims: &[usize],
    out_dims: &[usize],
) -> Tensor<R> {
    let n_in: usize = in_dims[2..].iter().product();
    let n_out: usize = out_dims[2..].iter().product();
    ifft_ctensor(re, im, out_dims).mul_scalar(n_out as f64 / n_in as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        prelude::*,
        tensor::{DType, Distribution, Tolerance},
    };

    fn roundtrip_check(re_vals: Vec<f32>, im_vals: Vec<f32>) {
        let device = Device::default();
        let n = re_vals.len();

        let re = Tensor::<1>::from_data(re_vals.as_slice(), &device).cast(DType::F32);
        let im = Tensor::<1>::from_data(im_vals.as_slice(), &device).cast(DType::F32);

        let (ft_re, ft_im) = signal::cfft(re.clone(), im.clone(), 0, Some(n));
        let (out_re, out_im) = icfft_full_spectrum(ft_re, ft_im, 0, n);

        let (orig_re, orig_im) = (
            re.into_data().try_to_vec::<f32>().unwrap(),
            im.into_data().try_to_vec::<f32>().unwrap(),
        );
        let (got_re, got_im) = (
            out_re.into_data().try_to_vec::<f32>().unwrap(),
            out_im.into_data().try_to_vec::<f32>().unwrap(),
        );

        for (a, e) in got_re.iter().zip(orig_re.iter()) {
            assert!((a - e).abs() < 1e-4, "real mismatch: {a} vs {e}");
        }
        for (a, e) in got_im.iter().zip(orig_im.iter()) {
            assert!((a - e).abs() < 1e-4, "imag mismatch: {a} vs {e}");
        }
    }

    /// Non-Hermitian input: would not survive a truncate-and-irfft
    /// implementation, so this exercises the full-spectrum path rather than
    /// passing by coincidence.
    #[test]
    fn round_trips_an_asymmetric_signal() {
        roundtrip_check(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![0.5, -1.0, 2.5, 0.0, -3.0, 1.5, 0.0, 4.0],
        );
    }

    #[test]
    fn round_trips_a_short_asymmetric_signal() {
        roundtrip_check(vec![1.0, 2.0, 3.0, 4.0], vec![0.1, 0.2, 0.3, 0.4]);
    }

    fn round_trip_check<const R: usize>(shape: [usize; R]) {
        let device = Device::default();
        let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let orig_dims = x.dims();

        let (re, im) = fft_ctensor(x.clone());
        let recon = ifft_ctensor(re, im, &orig_dims);

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

    /// Unscaled forward DFT of `x` (row-major, `shape`) over every axis from 2
    /// on, keeping `n/2 + 1` bins on the last (CONVENTIONS §4), in f64 on the
    /// host. Returns the real and imaginary parts, row-major.
    fn naive_dft(x: &[f32], shape: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let last = shape.len() - 1;
        let mut out_shape = shape.to_vec();
        out_shape[last] = shape[last] / 2 + 1;
        let strides = |s: &[usize]| -> Vec<usize> {
            let mut st = vec![1; s.len()];
            for a in (0..s.len() - 1).rev() {
                st[a] = st[a + 1] * s[a + 1];
            }
            st
        };
        let (in_st, out_st) = (strides(shape), strides(&out_shape));
        let total_out: usize = out_shape.iter().product();
        let (mut re, mut im) = (vec![0.0; total_out], vec![0.0; total_out]);
        for (o, (re_o, im_o)) in re.iter_mut().zip(im.iter_mut()).enumerate() {
            let k: Vec<usize> = (0..shape.len())
                .map(|a| o / out_st[a] % out_shape[a])
                .collect();
            for (i, &xi) in x.iter().enumerate() {
                let j: Vec<usize> = (0..shape.len()).map(|a| i / in_st[a] % shape[a]).collect();
                if j[..2] != k[..2] {
                    continue;
                }
                let phase: f64 = (2..shape.len())
                    .map(|a| (j[a] * k[a]) as f64 / shape[a] as f64)
                    .sum();
                let theta = -2.0 * std::f64::consts::PI * phase;
                *re_o += xi as f64 * theta.cos();
                *im_o += xi as f64 * theta.sin();
            }
        }
        (re, im)
    }

    /// Regression: on Metal (Burn fork `ddfa9af`), a Bluestein `cfft` along a
    /// non-last axis of non-power-of-two length was wrong when the product of
    /// the axes after it was even (20–85% relative error), because the fused
    /// `cat` ignores broadcasting on the vectorised axis. `fft_ctensor` now
    /// runs each complex FFT on the last axis. The shapes below hit that case
    /// on every complex axis: lengths 6 and 5, rfft half-width 4 (even). Checks
    /// the forward transform against a host DFT and the round trip.
    fn check_fft_at_bluestein_sizes<const R: usize>(shape: [usize; R]) {
        let device = Device::default();
        let x = Tensor::<R>::random(shape, Distribution::Normal(0.0, 1.0), &device);
        let x_vec = x.clone().into_data().try_to_vec::<f32>().unwrap();

        let (re, im) = fft_ctensor(x);
        let got_re = re.clone().into_data().try_to_vec::<f32>().unwrap();
        let got_im = im.clone().into_data().try_to_vec::<f32>().unwrap();
        let (want_re, want_im) = naive_dft(&x_vec, &shape);

        let num: f64 = got_re
            .iter()
            .zip(&want_re)
            .chain(got_im.iter().zip(&want_im))
            .map(|(g, w)| (*g as f64 - w).powi(2))
            .sum();
        let den: f64 = want_re.iter().chain(&want_im).map(|w| w * w).sum();
        let err = (num / den).sqrt();
        assert!(err < 1e-5, "{shape:?}: forward FFT rel err {err:e}");

        let recon = ifft_ctensor(re, im, &shape);
        let recon_vec = recon.into_data().try_to_vec::<f32>().unwrap();
        let max_diff = x_vec
            .iter()
            .zip(&recon_vec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-4, "{shape:?}: round trip max_diff={max_diff}");
    }

    #[test]
    fn fft_correct_at_bluestein_sizes_with_even_trailing_width() {
        check_fft_at_bluestein_sizes::<4>([2, 3, 6, 7]);
        check_fft_at_bluestein_sizes::<5>([1, 2, 6, 5, 7]);
    }

    /// `ifft_ctensor_to` keeps a constant's value at a new size (the forward-norm amplitude:
    /// only the DC bin is non-zero, and it moves unchanged), and at the input's own extents
    /// is `ifft_ctensor` scaled by one.
    #[test]
    fn ifft_ctensor_to_keeps_a_constant_and_is_the_inverse_at_the_same_size() {
        let device = Device::default();
        let x = Tensor::<4>::full([1, 2, 6, 5], 1.5, &device);
        let (re, im) = fft_ctensor(x.clone());
        // The 6 x 3 half-spectrum's DC bin into a 9 x 5 one (out 9 x 8).
        let (out_re, out_im) = (
            Tensor::<4>::zeros([1, 2, 9, 5], &device).slice_assign(
                [0..1, 0..2, 0..1, 0..1],
                re.clone().slice([0..1, 0..2, 0..1, 0..1]),
            ),
            Tensor::<4>::zeros([1, 2, 9, 5], &device),
        );
        ifft_ctensor_to(out_re, out_im, &[1, 2, 6, 5], &[1, 2, 9, 8])
            .into_data()
            .assert_approx_eq::<f32>(
                &Tensor::<4>::full([1, 2, 9, 8], 1.5, &device).into_data(),
                Tolerance::absolute(1e-6),
            );
        ifft_ctensor_to(re.clone(), im.clone(), &[1, 2, 6, 5], &[1, 2, 6, 5])
            .into_data()
            .assert_eq(&ifft_ctensor(re, im, &[1, 2, 6, 5]).into_data(), true);
    }
}
