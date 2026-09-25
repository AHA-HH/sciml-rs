//! Inverse complex FFT (icfft), which Burn does not provide natively.

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{prelude::*, tensor::DType};

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

    /// Deterministic, non-symmetric test signal.
    fn signal_values(len: usize, phase: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (0.7 * i as f32 + phase).sin() + 0.1 * i as f32)
            .collect()
    }

    /// Forward DFT `X[k] = Σ_j x[j] e^{-2πi jk/n}` along `axis` of a
    /// row-major `[rows, cols]` array, in f64 - the unnormalized convention
    /// `signal::cfft` uses.
    fn naive_dft(re: &[f32], im: &[f32], shape: [usize; 2], axis: usize) -> (Vec<f64>, Vec<f64>) {
        let [rows, cols] = shape;
        let n = shape[axis];
        let mut out_re = vec![0.0; rows * cols];
        let mut out_im = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let k = if axis == 0 { r } else { c };
                let (mut sr, mut si) = (0.0f64, 0.0f64);
                for j in 0..n {
                    let idx = if axis == 0 {
                        j * cols + c
                    } else {
                        r * cols + j
                    };
                    let theta = -std::f64::consts::TAU * (j * k) as f64 / n as f64;
                    let (xr, xi) = (re[idx] as f64, im[idx] as f64);
                    sr += xr * theta.cos() - xi * theta.sin();
                    si += xr * theta.sin() + xi * theta.cos();
                }
                out_re[r * cols + c] = sr;
                out_im[r * cols + c] = si;
            }
        }
        (out_re, out_im)
    }

    /// Checks `signal::cfft` along `axis` of a `[rows, cols]` tensor against
    /// [`naive_dft`]. Inputs are O(1) with n <= 64, so f32 error is ~1e-5 at
    /// worst; the defect below is O(1).
    fn cfft_matches_naive(shape: [usize; 2], axis: usize) {
        let device = Device::default();
        let len = shape[0] * shape[1];
        let (re_vals, im_vals) = (signal_values(len, 0.0), signal_values(len, 1.3));

        let re = Tensor::<2>::from_data(TensorData::new(re_vals.clone(), shape), &device);
        let im = Tensor::<2>::from_data(TensorData::new(im_vals.clone(), shape), &device);
        let (got_re, got_im) = signal::cfft(re, im, axis, Some(shape[axis]));
        let got_re = got_re.into_data().try_to_vec::<f32>().unwrap();
        let got_im = got_im.into_data().try_to_vec::<f32>().unwrap();

        let (want_re, want_im) = naive_dft(&re_vals, &im_vals, shape, axis);
        for i in 0..len {
            let err = (got_re[i] as f64 - want_re[i])
                .abs()
                .max((got_im[i] as f64 - want_im[i]).abs());
            assert!(
                err < 1e-3,
                "cfft {shape:?} axis {axis}, flat index {i}: error {err}"
            );
        }
    }

    /// Configurations `SpectralConv` relies on for Navier-Stokes at 64x64xT
    /// and similar: any length on the last axis, power-of-two lengths on
    /// earlier axes, and Bluestein on an earlier axis with odd trailing extent.
    #[test]
    fn cfft_matches_naive_dft_on_supported_layouts() {
        cfft_matches_naive([4, 40], 1); // last axis, Bluestein
        cfft_matches_naive([3, 7], 1); // last axis, odd
        cfft_matches_naive([8, 4], 0); // earlier axis, radix-2
        cfft_matches_naive([64, 6], 0); // earlier axis, radix-2, Navier-Stokes s
        cfft_matches_naive([5, 3], 0); // earlier axis, Bluestein, odd trailing
    }

    /// Known defect in the Burn fork (`ax1s-x1zz/burn`, branch
    /// `feat/bluestein-arbitrary-n`, rev aa9d161): `cfft` with a
    /// non-power-of-two length on a non-last axis gives O(1) errors when the
    /// trailing extent is even. Found 2026-09-24; re-enable once fixed.
    #[test]
    #[ignore = "Burn fork: Bluestein cfft on a non-last axis is wrong for even trailing extents"]
    fn cfft_bluestein_non_last_axis_even_trailing() {
        cfft_matches_naive([5, 4], 0);
        cfft_matches_naive([6, 6], 0);
        cfft_matches_naive([7, 2], 0);
    }
}
