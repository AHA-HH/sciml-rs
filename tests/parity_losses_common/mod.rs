//! Library CM1.6a: `FiniteDiff`, the Sobolev losses (H1, Hdiv), `PointwiseQuantileLoss`,
//! `mse_loss` and `LpLoss::rel` against `neuraloperator` 2.0.0's, on the fixtures
//! `tests/fixtures/parity/losses{1,2,3}d_{f32,f64}`; CM1.6b: the aggregators on
//! `meta_losses_{f32,f64}`; CM2.5: the equation losses on `equation_losses_{f32,f64}` (all
//! written by `tools/parity/generate_losses.py`).
//! Used by `tests/parity_losses.rs` (f32) and `tests/parity_losses_f64.rs`.

use burn::{prelude::*, tensor::TensorData};
use sciml_rs::neural_operators::losses::{
    data_losses::{LpLoss, Reduction},
    differentiation::FiniteDiff,
    equation_losses::{
        BurgersEqnLoss, PoissonEqnLoss, ic_loss, poisson_boundary_loss, poisson_interior_loss,
    },
    meta_losses::{Field, FieldwiseAggregatorLoss, LossFn, Relobralo, SoftAdapt, WeightedSumLoss},
    pointwise::{PointwiseQuantileLoss, mse_loss},
    sobolev::SobolevLoss,
};

use crate::parity_common::{Fixture, assert_parity};

/// The fixtures' finite-difference steps, domain measures and quantile level
/// (`generate_losses.py`).
const H: [f64; 3] = [0.1, 0.2, 0.3];
const MEASURE: [f64; 3] = [1.0, 2.0, 0.5];
const ALPHA: f64 = 0.1;
/// The bound on `lp.rel`: 2.0.0's eps of 1e-8 against norms ‖y‖ ≳ 1 (measured: 1.9e-9 at
/// most, in f64).
const LP_EPS_GAP: f64 = 1e-8;

/// The Poisson point cloud's boundary points, first in the list (`generate_losses.py`).
const POISSON_BOUNDARY: usize = 6;

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
    let x: Tensor<R> = fx.tensor("x", device);
    let y: Tensor<R> = fx.tensor("y", device);
    let v: Tensor<R> = fx.tensor("v", device);
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

/// The aggregator comparisons of `meta_losses_<precision>` (CM1.6b), on `device`; the
/// adaptive ones on its autodiff form, for the gradient with respect to each loss.
pub fn run_meta(name: &str, device: &Device) {
    let mut fx = Fixture::open(name);
    let x: Tensor<3> = fx.tensor("x", device);
    let y: Tensor<3> = fx.tensor("y", device);
    let pred: Tensor<2> = fx.tensor("pred", device);
    let truth: Tensor<2> = fx.tensor("truth", device);
    let rho_default = fx.array("relobralo.rho");
    let rho_w = fx.array("relobralo_w.rho");
    let ad = device.clone().autodiff();
    let table: Tensor<2> = fx.tensor("table", &ad);
    let mut c = Checks { fx, worst: 0.0 };

    let pair = || -> Vec<LossFn<3>> {
        vec![
            Box::new(|a, b| SobolevLoss::<1>::h1(Reduction::Sum).rel(a, b)),
            Box::new(mse_loss),
        ]
    };
    let out = WeightedSumLoss::new(pair(), None).loss(x.clone(), y.clone());
    c.check(out, "wsum.default");
    let out = WeightedSumLoss::new(pair(), Some(vec![0.3, 0.7])).loss(x, y);
    c.check(out, "wsum.weighted");

    let fieldwise = FieldwiseAggregatorLoss::new(vec![
        Field {
            axis: 1,
            channels: 0..1,
            loss: Box::new(mse_loss),
        },
        Field {
            axis: 1,
            channels: 2..3,
            loss: Box::new(|a, b| PointwiseQuantileLoss::new(ALPHA, Reduction::Sum).loss(a, b)),
        },
    ]);
    c.check(fieldwise.loss(pred, truth), "fieldwise");

    let mut soft = SoftAdapt::new(3, None);
    adaptive(&mut c, &table, "softadapt", |l, s| soft.forward(l, s));
    let mut soft = SoftAdapt::new(3, Some(vec![1.0, 2.0, 1.0]));
    adaptive(&mut c, &table, "softadapt_w", |l, s| soft.forward(l, s));

    // torch's recorded draws for steps 1.. replace the seeded ones.
    let rho = |draws: &ndarray::ArrayD<f64>, s: usize| s > 0 && draws[[s - 1]] == 1.0;
    let mut relo = Relobralo::new(3, None, 0);
    adaptive(&mut c, &table, "relobralo", |l, s| {
        relo.forward_with_rho(l, s, rho(&rho_default, s))
    });
    let mut relo = Relobralo::new(3, Some(vec![1.0, 1.0, 0.5]), 0);
    (relo.alpha, relo.beta, relo.tau) = (0.9, 0.5, 2.0);
    adaptive(&mut c, &table, "relobralo_w", |l, s| {
        relo.forward_with_rho(l, s, rho(&rho_w, s))
    });

    println!("{name}: worst relative max error {:.3e}", c.worst);
}

/// Runs an adaptive aggregator over the rows of `table` (`[steps, losses]`), each loss a
/// leaf, and checks `<name>.loss`, `<name>.weights` and `<name>.grad` (d aggregate / d loss).
fn adaptive(
    c: &mut Checks,
    table: &Tensor<2>,
    name: &str,
    mut step: impl FnMut(Vec<Tensor<1>>, usize) -> (Tensor<1>, Vec<f64>),
) {
    let [steps, n] = table.dims();
    let (mut loss, mut weights, mut grads) = (vec![], vec![], vec![]);
    for s in 0..steps {
        let leaves: Vec<Tensor<1>> = (0..n)
            .map(|i| {
                let l: Tensor<1> = table.clone().slice([s..s + 1, i..i + 1]).reshape([1]);
                l.detach().require_grad()
            })
            .collect();
        let (out, w) = step(leaves.clone(), s);
        let g = out.backward();
        grads.extend(
            leaves
                .iter()
                .map(|l| l.grad(&g).unwrap().into_scalar::<f64>()),
        );
        loss.push(out.into_scalar::<f64>());
        weights.extend(w);
    }
    let device = table.device();
    let host = |v: Vec<f64>| Tensor::<2>::from_data(TensorData::new(v, [steps, n]), &device);
    let loss = Tensor::<1>::from_data(TensorData::new(loss, [steps]), &device);
    c.check(loss, &format!("{name}.loss"));
    c.check(host(weights), &format!("{name}.weights"));
    c.check(host(grads), &format!("{name}.grad"));
}

/// The equation-loss comparisons of `equation_losses_<precision>` (CM2.5), on `device`.
/// 2.0.0 differentiates the Poisson u by autograd with respect to the query points; the
/// fixture stores those derivatives, from which the crate's left-hand side is formed.
pub fn run_equation(name: &str, device: &Device) {
    let mut fx = Fixture::open(name);
    let u: Tensor<4> = fx.tensor("u", device);
    let x: Tensor<4> = fx.tensor("x", device);
    let mut p = |key: &str| -> Tensor<1> { fx.tensor(&format!("poisson.{key}"), device) };
    let (pu, ux, uy, lap, f) = (p("u"), p("u_x"), p("u_y"), p("lap"), p("f"));
    let y: Tensor<3> = fx.tensor("poisson.y", device);
    let mut c = Checks { fx, worst: 0.0 };

    c.check(BurgersEqnLoss::default().loss(u.clone()), "burgers.default");
    let visc = BurgersEqnLoss {
        visc: 0.05,
        domain_length: [2.0, 3.0],
    };
    c.check(visc.loss(u.clone()), "burgers.visc");
    c.check(ic_loss(u, x), "ic");

    let n = pu.dims()[0];
    let y = y.reshape([n]);
    let inner = pu.clone().narrow(0, POISSON_BOUNDARY, n - POISSON_BOUNDARY);
    let interior = poisson_interior_loss(inner, [ux.clone(), uy.clone()], lap.clone(), f.clone());
    c.check(interior, "poisson.interior");
    let boundary = poisson_boundary_loss(pu.clone(), y.clone(), POISSON_BOUNDARY);
    c.check(boundary, "poisson.boundary");
    let eqn = PoissonEqnLoss {
        interior_weight: 0.5,
        boundary_weight: 2.0,
    };
    let out = eqn.points(pu, y, POISSON_BOUNDARY, [ux, uy], lap, f);
    c.check(out, "poisson.eqn");

    println!("{name}: worst relative max error {:.3e}", c.worst);
}
