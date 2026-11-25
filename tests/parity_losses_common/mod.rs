//! Library CM1.6a: `FiniteDiff`, the Sobolev losses (H1, Hdiv), `PointwiseQuantileLoss`,
//! `mse_loss` and `LpLoss::rel` against `neuraloperator` 2.0.0's, on the fixtures
//! `tests/fixtures/parity/losses{1,2,3}d_{f32,f64}` (`tools/parity/generate_losses.py`).
//! Used by `tests/parity_losses.rs` (f32) and `tests/parity_losses_f64.rs`.

use burn::{prelude::*, tensor::TensorData};
use sciml_rs::neural_operators::losses::{
    data_losses::{LpLoss, Reduction},
    differentiation::FiniteDiff,
    pointwise::{PointwiseQuantileLoss, mse_loss},
    sobolev::SobolevLoss,
};

use crate::parity_common::{Fixture, Precision, assert_parity};

/// The fixtures' finite-difference steps, domain measures and quantile level
/// (`generate_losses.py`).
const H: [f64; 3] = [0.1, 0.2, 0.3];
const MEASURE: [f64; 3] = [1.0, 2.0, 0.5];
const ALPHA: f64 = 0.1;
/// The bound on `lp.rel`: 2.0.0's eps of 1e-8 against norms ‖y‖ ≳ 1 (measured: 1.9e-9 at
/// most, in f64).
const LP_EPS_GAP: f64 = 1e-8;

/// The fixture array `key` as a rank-`N` tensor in the fixture's precision.
pub fn tensor<const N: usize>(fx: &mut Fixture, key: &str, device: &Device) -> Tensor<N> {
    let a = fx.array(key);
    let shape = a.shape().to_vec();
    let data = match fx.precision {
        Precision::F32 => TensorData::new(a.iter().map(|&v| v as f32).collect(), shape),
        Precision::F64 => TensorData::new(a.iter().copied().collect(), shape),
    };
    Tensor::from_data(data, device)
}

/// Asserts `actual` matches the fixture's `key` within D3 and returns the error.
fn check<const N: usize>(fx: &mut Fixture, actual: Tensor<N>, key: &str) -> f64 {
    let expected = fx.array(key);
    assert_parity(actual, &expected, fx.precision.d3(), key)
}

/// Periodicity per axis by the fixture's boundary name: `per`, `non`, `mix` (x one-sided).
fn boundaries<const D: usize>() -> Vec<(&'static str, [bool; D])> {
    let mut out = vec![("per", [true; D]), ("non", [false; D])];
    if D >= 2 {
        let mut mix = [true; D];
        mix[0] = false;
        out.push(("mix", mix));
    }
    out
}

/// Every comparison of one fixture, on `device`. `R` = D + 2 (x, y, v), `G` = R + 1 (gradient), `V` =
/// R − 1 (divergence; curl in 2D).
pub fn run<const D: usize, const R: usize, const G: usize, const V: usize>(
    name: &str,
    device: &Device,
) {
    let mut fx = Fixture::open(name);
    let x: Tensor<R> = tensor(&mut fx, "x", device);
    let y: Tensor<R> = tensor(&mut fx, "y", device);
    let v: Tensor<R> = tensor(&mut fx, "v", device);
    let h: [f64; D] = std::array::from_fn(|a| H[a]);
    let measure: [f64; D] = std::array::from_fn(|a| MEASURE[a]);
    let mut worst = 0.0_f64;

    for (bc, periodic) in boundaries::<D>() {
        let fd = FiniteDiff::new(h, periodic);
        for (a, axis) in ["x", "y", "z"].into_iter().take(D).enumerate() {
            for order in [1, 2] {
                let key = format!("fd.d{axis}{order}.{bc}");
                worst = worst.max(check(&mut fx, fd.partial(x.clone(), a, order), &key));
            }
        }
        let key = format!("fd.laplacian.{bc}");
        worst = worst.max(check(&mut fx, fd.laplacian(x.clone()), &key));
        let grad: Tensor<G> = fd.gradient(x.clone());
        let key = format!("fd.gradient.{bc}");
        // 2.0.0's 1D gradient has no component axis; ours keeps one of length 1.
        worst = worst.max(if D == 1 {
            check(&mut fx, grad.reshape(x.dims()), &key)
        } else {
            check(&mut fx, grad, &key)
        });
        let key = format!("fd.divergence.{bc}");
        worst = worst.max(check(&mut fx, fd.divergence::<R, V>(v.clone()), &key));
        let key = format!("fd.curl.{bc}");
        if D == 2 {
            worst = worst.max(check(&mut fx, fd.curl::<R, V>(v.clone()), &key));
        } else if D == 3 {
            worst = worst.max(check(&mut fx, fd.curl::<R, R>(v.clone()), &key));
        }

        for (norm, loss) in [
            ("h1", SobolevLoss::<D>::h1(Reduction::Sum)),
            ("hdiv", SobolevLoss::<D>::hdiv(Reduction::Sum)),
        ] {
            let loss = SobolevLoss {
                measure,
                periodic,
                ..loss
            };
            let key = format!("{norm}.rel.{bc}.sum");
            worst = worst.max(check(&mut fx, loss.rel(x.clone(), y.clone()), &key));
            let key = format!("{norm}.abs.{bc}.sum");
            worst = worst.max(check(&mut fx, loss.abs(x.clone(), y.clone()), &key));
        }
    }
    let h1_mean = SobolevLoss {
        measure,
        ..SobolevLoss::<D>::h1(Reduction::Mean)
    };
    let out = h1_mean.rel(x.clone(), y.clone());
    worst = worst.max(check(&mut fx, out, "h1.rel.per.mean"));

    for (red, reduction) in [("sum", Reduction::Sum), ("mean", Reduction::Mean)] {
        let out = PointwiseQuantileLoss::new(ALPHA, reduction).loss(x.clone(), y.clone());
        worst = worst.max(check(&mut fx, out, &format!("quantile.{red}")));
    }
    worst = worst.max(check(&mut fx, mse_loss(x.clone(), y.clone()), "mse"));

    // 2.0.0 reduces over batch and channel: flatten both into the crate's batch axis.
    let dims = x.dims();
    let rows: usize = dims[..R - D].iter().product();
    let flat = |t: Tensor<R>| t.reshape([rows as i32, -1]);
    let out = LpLoss::new(2, Reduction::Sum).rel(flat(x), flat(y));
    // 2.0.0 divides by ‖y‖ + 1e-8, the crate by ‖y‖ (CONVENTIONS §11, constraint 1): a
    // relative gap of about 1e-8 / ‖y‖, inside the f32 bound but not the f64 one.
    let expected = fx.array("lp.rel");
    let tol = fx.precision.d3().max(LP_EPS_GAP);
    let lp = assert_parity(out, &expected, tol, "lp.rel");

    println!("{name}: worst relative max error {worst:.3e}; lp.rel {lp:.3e}");
}
