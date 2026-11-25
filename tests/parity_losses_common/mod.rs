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

/// An open fixture and the worst D3 error seen on it so far.
struct Checks {
    fx: Fixture,
    worst: f64,
}

impl Checks {
    /// Asserts `actual` matches the fixture's `key` within D3, and keeps the worst error.
    fn check<const N: usize>(&mut self, actual: Tensor<N>, key: &str) {
        let expected = self.fx.array(key);
        let err = assert_parity(actual, &expected, self.fx.precision.d3(), key);
        self.worst = self.worst.max(err);
    }
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

/// Every comparison of one fixture, on `device`. `R` = D + 2 (x, y, v), `G` = R + 1
/// (gradient), `V` = R − 1 (divergence; curl in 2D).
pub fn run<const D: usize, const R: usize, const G: usize, const V: usize>(
    name: &str,
    device: &Device,
) {
    let mut fx = Fixture::open(name);
    let x: Tensor<R> = tensor(&mut fx, "x", device);
    let y: Tensor<R> = tensor(&mut fx, "y", device);
    let v: Tensor<R> = tensor(&mut fx, "v", device);
    let mut c = Checks { fx, worst: 0.0 };
    let h: [f64; D] = std::array::from_fn(|a| H[a]);
    let measure: [f64; D] = std::array::from_fn(|a| MEASURE[a]);

    for (bc, periodic) in boundaries::<D>() {
        let fd = FiniteDiff { h, periodic };
        for (a, axis) in ["x", "y", "z"].into_iter().take(D).enumerate() {
            for order in [1, 2] {
                let key = format!("fd.d{axis}{order}.{bc}");
                c.check(fd.partial(x.clone(), a, order), &key);
            }
        }
        let key = format!("fd.laplacian.{bc}");
        c.check(fd.laplacian(x.clone()), &key);
        let grad: Tensor<G> = fd.gradient(x.clone());
        let key = format!("fd.gradient.{bc}");
        // 2.0.0's 1D gradient has no component axis; ours keeps one of length 1.
        if D == 1 {
            c.check(grad.reshape(x.dims()), &key);
        } else {
            c.check(grad, &key);
        }
        let key = format!("fd.divergence.{bc}");
        c.check(fd.divergence::<R, V>(v.clone()), &key);
        let key = format!("fd.curl.{bc}");
        if D == 2 {
            c.check(fd.curl_2d::<R, V>(v.clone()), &key);
        } else if D == 3 {
            c.check(fd.curl_3d(v.clone()), &key);
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
            c.check(loss.rel(x.clone(), y.clone()), &key);
            let key = format!("{norm}.abs.{bc}.sum");
            c.check(loss.abs(x.clone(), y.clone()), &key);
        }
    }
    let h1_mean = SobolevLoss {
        measure,
        ..SobolevLoss::<D>::h1(Reduction::Mean)
    };
    let out = h1_mean.rel(x.clone(), y.clone());
    c.check(out, "h1.rel.per.mean");

    for (red, reduction) in [("sum", Reduction::Sum), ("mean", Reduction::Mean)] {
        let out = PointwiseQuantileLoss::new(ALPHA, reduction).loss(x.clone(), y.clone());
        c.check(out, &format!("quantile.{red}"));
    }
    c.check(mse_loss(x.clone(), y.clone()), "mse");

    // 2.0.0 reduces over batch and channel: flatten both into the crate's batch axis.
    let dims = x.dims();
    let rows: usize = dims[..R - D].iter().product();
    let flat = |t: Tensor<R>| t.reshape([rows as i32, -1]);
    let out = LpLoss::new(2, Reduction::Sum).rel(flat(x), flat(y));
    // 2.0.0 divides by ‖y‖ + 1e-8, the crate by ‖y‖ (CONVENTIONS §11, constraint 1): a
    // relative gap of about 1e-8 / ‖y‖, inside the f32 bound but not the f64 one.
    let expected = c.fx.array("lp.rel");
    let tol = c.fx.precision.d3().max(LP_EPS_GAP);
    let lp = assert_parity(out, &expected, tol, "lp.rel");

    println!(
        "{name}: worst relative max error {:.3e}; lp.rel {lp:.3e}",
        c.worst
    );
}
