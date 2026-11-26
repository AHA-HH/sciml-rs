//! Library CM1.6a: `FiniteDiff`, the Sobolev losses (H1, Hdiv), `PointwiseQuantileLoss`,
//! `mse_loss` and `LpLoss::rel` against `neuraloperator` 2.0.0's, on the fixtures
//! `tests/fixtures/parity/losses{1,2,3}d_{f32,f64}`; CM1.6b: the aggregators on
//! `meta_losses_{f32,f64}`; CM2.5: the equation losses on `equation_losses_{f32,f64}` (all
//! written by `tools/parity/generate_losses.py`); CM1.6c: `FourierDiff` and `non_uniform_fd`
//! on `derivatives_{f32,f64}`.
//! Used by `tests/parity_losses.rs` (f32) and `tests/parity_losses_f64.rs`.

use burn::{prelude::*, tensor::TensorData};
use sciml_rs::neural_operators::losses::{
    data_losses::{LpLoss, Reduction},
    differentiation::{FiniteDiff, FourierDiff, GridDiff},
    equation_losses::{
        BurgersEqnLoss, PoissonEqnLoss, ic_loss, poisson_boundary_loss, poisson_interior_loss,
    },
    meta_losses::{Field, FieldwiseAggregatorLoss, LossFn, Relobralo, SoftAdapt, WeightedSumLoss},
    non_uniform_fd::NonUniformFd,
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

/// `FourierDiff`'s domain lengths and derivative orders in `derivatives_*`.
const FOURIER_L: [f64; 3] = [2.0, 3.0, 1.5];
const FOURIER_ORDERS: [usize; 3] = [1, 2, 3];

/// The bound on 2.0.0's regularised `non_uniform_fd` in f64: its condition ≈ 5e6 times ε
/// (measured: 6.4e-11).
const RIDGE_BOUND_F64: f64 = 1e-9;

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

/// The Laplacian, gradient, divergence and curl (D = 2, 3) of `diff` on `u` and `v` against
/// the keys `key("laplacian")` and so on. `drop_1d_axis`: the 1D reference gradient has no
/// component axis.
fn vector_calculus<O, const D: usize, const R: usize, const G: usize, const V: usize>(
    c: &mut Checks,
    diff: &O,
    u: &Tensor<R>,
    v: &Tensor<R>,
    key: impl Fn(&str) -> String,
    drop_1d_axis: bool,
) where
    O: GridDiff<D>,
{
    c.check(diff.laplacian(u.clone()), &key("laplacian"));
    let grad: Tensor<G> = diff.gradient(u.clone());
    if D == 1 && drop_1d_axis {
        c.check(grad.reshape(u.dims()), &key("gradient"));
    } else {
        c.check(grad, &key("gradient"));
    }
    c.check(diff.divergence::<R, V>(v.clone()), &key("divergence"));
    if D == 2 {
        c.check(diff.curl_2d::<R, V>(v.clone()), &key("curl"));
    } else if D == 3 {
        c.check(diff.curl_3d(v.clone()), &key("curl"));
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
        let key = |what: &str| format!("fd.{what}.{bc}");
        // 2.0.0's `FiniteDiff` gradient has no component axis in 1D; ours keeps one of length 1.
        vector_calculus::<_, D, R, G, V>(&mut c, &fd, &x, &v, key, true);

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

/// The `FourierDiff` comparisons of `derivatives_<precision>` (CM1.6c) in `D` dimensions, on
/// `device`. `R` = D + 2 (u, v), `G` = R + 1 (gradient), `V` = R − 1 (divergence; curl in 2D).
pub fn run_fourier<const D: usize, const R: usize, const G: usize, const V: usize>(
    name: &str,
    device: &Device,
) {
    let mut fx = Fixture::open(name);
    let u: Tensor<R> = fx.tensor(&format!("u{D}"), device);
    let v: Tensor<R> = fx.tensor(&format!("v{D}"), device);
    let mut c = Checks { fx, worst: 0.0 };
    let fd = FourierDiff::<D> {
        l: std::array::from_fn(|a| FOURIER_L[a]),
    };
    let key = |what: &str| format!("fourier{D}.{what}");
    for (a, axis) in ["x", "y", "z"].into_iter().take(D).enumerate() {
        for order in FOURIER_ORDERS {
            c.check(
                fd.partial(u.clone(), a, order),
                &key(&format!("d{axis}{order}")),
            );
        }
    }
    // 2.0.0's `FourierDiff` keeps the 1D gradient's component axis, as ours.
    vector_calculus::<_, D, R, G, V>(&mut c, &fd, &u, &v, key, false);
    println!(
        "{name} fourier{D}: worst relative max error {:.3e}",
        c.worst
    );
}

/// The `non_uniform_fd` comparisons of `derivatives_<precision>` (CM1.6c), on `device`: the
/// weights in f64 on the host, the derivatives on `device`. The ridge case
/// (`regularize_lstsq`) is checked against `ridge_reference`'s result: 2.0.0's system has
/// condition ≈ ‖AᵀA‖ / 1e-6 ≈ 5e6, and its own f32 result is about 4e-2 from its f64 one on
/// the same inputs (the f32 points cast), so the f32 test passes the f64 fixture, within D3,
/// and the f64 test its own, within [`RIDGE_BOUND_F64`].
pub fn run_non_uniform(name: &str, ridge_reference: &str, device: &Device) {
    let base = |derivative_indices: Vec<usize>| NonUniformFd {
        derivative_indices,
        ..NonUniformFd::default()
    };
    let cases = [
        ("nu2", base(vec![0, 1])),
        (
            "nu2_radius",
            NonUniformFd {
                radius: Some(0.2),
                ..base(vec![0, 1])
            },
        ),
        (
            "nu2_reg",
            NonUniformFd {
                regularize: true,
                ..base(vec![0, 1])
            },
        ),
        (
            "nu3",
            NonUniformFd {
                num_neighbors: 7,
                ..base(vec![0, 2])
            },
        ),
    ];
    let mut c = Checks {
        fx: Fixture::open(name),
        worst: 0.0,
    };
    let mut ridge_fx = Fixture::open(ridge_reference);
    for (case, op) in cases {
        let points = c.fx.array(&format!("{case}.points"));
        let points = points.into_dimensionality().unwrap();
        let values: Tensor<1> = c.fx.tensor(&format!("{case}.values"), device);
        let out = op.derivatives(&points, values);
        let key = format!("{case}.d");
        if !op.regularize {
            c.check(out, &key);
            continue;
        }
        let tol = match c.fx.precision {
            Precision::F32 => Precision::F32.d3(),
            Precision::F64 => RIDGE_BOUND_F64,
        };
        let err = assert_parity(out, &ridge_fx.array(&key), tol, &key);
        println!("{name} {key} against {ridge_reference}: relative max error {err:.3e}");
        c.worst = c.worst.max(err);
    }
    println!(
        "{name} non_uniform_fd: worst relative max error {:.3e}",
        c.worst
    );
}
