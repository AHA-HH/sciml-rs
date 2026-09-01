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

    fn roundtrip_check(re_vals: Vec<f64>, im_vals: Vec<f64>) {
        let device = Device::default();
        let n = re_vals.len();

        let re = Tensor::<1>::from_data(re_vals.as_slice(), &device).cast(DType::F64);
        let im = Tensor::<1>::from_data(im_vals.as_slice(), &device).cast(DType::F64);

        let (ft_re, ft_im) = signal::cfft(re.clone(), im.clone(), 0, Some(n));
        let (out_re, out_im) = icfft_full_spectrum(ft_re, ft_im, 0, n);

        let (orig_re, orig_im) = (
            re.into_data().try_to_vec::<f64>().unwrap(),
            im.into_data().try_to_vec::<f64>().unwrap(),
        );
        let (got_re, got_im) = (
            out_re.into_data().try_to_vec::<f64>().unwrap(),
            out_im.into_data().try_to_vec::<f64>().unwrap(),
        );

        for (a, e) in got_re.iter().zip(orig_re.iter()) {
            assert!((a - e).abs() < 1e-6, "real mismatch: {a} vs {e}");
        }
        for (a, e) in got_im.iter().zip(orig_im.iter()) {
            assert!((a - e).abs() < 1e-6, "imag mismatch: {a} vs {e}");
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
}
