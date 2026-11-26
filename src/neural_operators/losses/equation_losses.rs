//! Physics (equation) losses, as `neuraloperator` 2.0.0's `equation_losses`
//! (`neuralop/losses/equation_losses.py`; library CM2.5): the residual of a PDE evaluated on
//! a model's output, for physics-informed training (PINO).
//!
//! The derivatives come from a [`GridDiff`]: 2.0.0's finite differences ([`FiniteDiff`]),
//! or a differentiation matrix per axis ([`MatrixDiff`]), such as the crate's Chebyshev
//! matrices on the CGL grid (CONVENTIONS §12). 2.0.0's base loss is `F.mse_loss`, the mean
//! over every entry: [`mean_squared_error`], not 2.0.0's `MSELoss`
//! ([`mse_loss`](super::pointwise::mse_loss)).
//!
//! 2.0.0's Poisson interior loss differentiates a point-cloud model's output by autograd
//! with respect to the query points. Here the left-hand side takes the derivatives as
//! arguments ([`nonlinear_poisson_lhs`]), and [`PoissonEqnLoss::grid`] supplies them on a
//! grid.
//!
//! [`MatrixDiff`]: super::differentiation::MatrixDiff

use burn::tensor::Tensor;

use super::differentiation::{FiniteDiff, GridDiff};
use super::pointwise::mean_squared_error;

/// 2.0.0's `BurgersEqnLoss`: the residual of the viscous Burgers equation
/// ∂u/∂t = −u ∂u/∂x + ν ∂²u/∂x², by [`mean_squared_error`] between the two sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BurgersEqnLoss {
    /// The viscosity ν, default 0.01.
    pub visc: f64,
    /// The lengths (L_t, L_x) of the time interval and the spatial domain, default (1, 1).
    pub domain_length: [f64; 2],
}

impl Default for BurgersEqnLoss {
    fn default() -> Self {
        Self {
            visc: 0.01,
            domain_length: [1.0, 1.0],
        }
    }
}

impl BurgersEqnLoss {
    /// 2.0.0's finite differences for an `n_t` × `n_x` grid: non-periodic on both axes, with
    /// steps L_t / (n_t − 1) (both ends of the interval on the grid) and L_x / n_x.
    ///
    /// # Panics
    /// If `n_t < 2` or `n_x` is 0.
    pub fn fdm(&self, n_t: usize, n_x: usize) -> FiniteDiff<2> {
        assert!(n_t >= 2 && n_x >= 1, "BurgersEqnLoss: a {n_t} × {n_x} grid");
        let [l_t, l_x] = self.domain_length;
        FiniteDiff {
            h: [l_t / (n_t - 1) as f64, l_x / n_x as f64],
            periodic: [false, false],
        }
    }

    /// The loss of `u` `[.., n_t, n_x]` (2.0.0: `[batch, 1, n_t, n_x]`) with 2.0.0's finite
    /// differences ([`fdm`](Self::fdm)). Returns `[1]`.
    ///
    /// # Panics
    /// If `R < 2`, or an axis has fewer than 4 points.
    pub fn loss<const R: usize>(&self, u: Tensor<R>) -> Tensor<1> {
        assert!(R >= 2, "BurgersEqnLoss: rank {R} below (t, x)");
        let dims = u.dims();
        self.loss_with(&self.fdm(dims[R - 2], dims[R - 1]), u)
    }

    /// The loss of `u` `[.., n_t, n_x]` with the derivatives of `diff`, axis 0 being t and
    /// axis 1 x. Returns `[1]`.
    ///
    /// # Panics
    /// As `diff`'s [`partial`](GridDiff::partial).
    pub fn loss_with<G: GridDiff<2>, const R: usize>(&self, diff: &G, u: Tensor<R>) -> Tensor<1> {
        let dudt = diff.partial(u.clone(), 0, 1);
        let dudx = diff.partial(u.clone(), 1, 1);
        let dudxx = diff.partial(u.clone(), 1, 2);
        let rhs = dudxx.mul_scalar(self.visc) - dudx * u;
        mean_squared_error(dudt, rhs)
    }
}

/// 2.0.0's `ICLoss`: the [`mean_squared_error`] between the prediction and the input at
/// channel 0 and t = 0, `y_pred[:, 0, 0, :]` against `x[:, 0, 0, :]`. `y_pred`, `x`:
/// `[batch, channels, n_t, n_x]`. Returns `[1]`.
///
/// # Panics
/// If the batch or x extents differ.
pub fn ic_loss(y_pred: Tensor<4>, x: Tensor<4>) -> Tensor<1> {
    let initial = |t: Tensor<4>| t.narrow(1, 0, 1).narrow(2, 0, 1);
    mean_squared_error(initial(y_pred), initial(x))
}

/// The left-hand side of 2.0.0's nonlinear Poisson equation ∇·((1 + 0.1u²)∇u) = f, in its
/// expanded form Δu (1 + 0.1u²) + 0.2u |∇u|². `u`, each component of `grad` (∂u/∂xₐ) and
/// `lap` (Δu): one shape, the result's.
///
/// # Panics
/// If `D` is 0, or the shapes differ.
pub fn nonlinear_poisson_lhs<const D: usize, const R: usize>(
    u: Tensor<R>,
    grad: [Tensor<R>; D],
    lap: Tensor<R>,
) -> Tensor<R> {
    let grad_sq = grad
        .into_iter()
        .map(|t| t.square())
        .reduce(|a, b| a + b)
        .expect("nonlinear_poisson_lhs: no gradient components");
    lap.clone() + lap.mul_scalar(0.1) * u.clone().square() + u.mul_scalar(0.2) * grad_sq
}

/// 2.0.0's `PoissonBoundaryLoss` (with `out_sub_level` 1): the [`mean_squared_error`] of the
/// first `num_boundary` points, the boundary ones of a point list. `y_pred`, `y`:
/// `[n_points]`. Returns `[1]`.
///
/// # Panics
/// If the shapes differ, or `num_boundary` is 0 or more than the points.
pub fn poisson_boundary_loss(y_pred: Tensor<1>, y: Tensor<1>, num_boundary: usize) -> Tensor<1> {
    assert_eq!(
        y_pred.dims(),
        y.dims(),
        "poisson_boundary_loss: shapes differ"
    );
    let n = y.dims()[0];
    assert!(
        (1..=n).contains(&num_boundary),
        "poisson_boundary_loss: {num_boundary} boundary points of {n}"
    );
    mean_squared_error(
        y_pred.narrow(0, 0, num_boundary),
        y.narrow(0, 0, num_boundary),
    )
}

/// 2.0.0's `PoissonInteriorLoss` given the derivatives: the [`mean_squared_error`] of
/// [`nonlinear_poisson_lhs`] against the source `f`. `u`, each component of `grad`, `lap`
/// and `f`: one shape, at the interior points. Returns `[1]`.
///
/// # Panics
/// As [`nonlinear_poisson_lhs`], or if `f`'s shape differs.
pub fn poisson_interior_loss<const D: usize, const R: usize>(
    u: Tensor<R>,
    grad: [Tensor<R>; D],
    lap: Tensor<R>,
    f: Tensor<R>,
) -> Tensor<1> {
    mean_squared_error(nonlinear_poisson_lhs(u, grad, lap), f)
}

/// 2.0.0's `PoissonEqnLoss`: the weighted sum of the interior equation loss and the
/// boundary loss of the nonlinear Poisson equation ([`nonlinear_poisson_lhs`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoissonEqnLoss {
    /// The weight of the interior (equation) loss.
    pub interior_weight: f64,
    /// The weight of the boundary loss.
    pub boundary_weight: f64,
}

impl PoissonEqnLoss {
    /// 2.0.0's `PoissonEqnLoss` on a point list (`out_sub_level` 1), the first
    /// `num_boundary` points the boundary: [`poisson_interior_loss`] on the others and
    /// [`poisson_boundary_loss`] against `y`. `u`, `y`: `[n_points]`; each component of
    /// `grad` (∂u/∂xₐ), `lap` (Δu) and `f`: `[n_points − num_boundary]`, at the interior
    /// points (2.0.0 takes them by autograd). Returns `[1]`.
    ///
    /// # Panics
    /// As [`poisson_boundary_loss`] and [`poisson_interior_loss`].
    pub fn points<const D: usize>(
        &self,
        u: Tensor<1>,
        y: Tensor<1>,
        num_boundary: usize,
        grad: [Tensor<1>; D],
        lap: Tensor<1>,
        f: Tensor<1>,
    ) -> Tensor<1> {
        let n = u.dims()[0];
        let boundary = poisson_boundary_loss(u.clone(), y, num_boundary);
        let inner = u.narrow(0, num_boundary, n - num_boundary);
        self.weigh(poisson_interior_loss(inner, grad, lap, f), boundary)
    }

    /// The loss on a grid: the interior loss is the [`mean_squared_error`] of the left-hand
    /// side against `f` on the interior nodes, with the derivatives of `diff`; the boundary
    /// loss the mean of (u − g)² over the boundary nodes (those on a face of the spatial
    /// box). `u`, `g`, `f`: `[.., s_1, .., s_D]`, one shape; only `f`'s interior and `g`'s
    /// boundary nodes are read. Returns `[1]`.
    ///
    /// # Panics
    /// If the shapes differ, a spatial axis has fewer than 3 points, or as `diff`'s
    /// [`partial`](GridDiff::partial).
    pub fn grid<G: GridDiff<D>, const D: usize, const R: usize>(
        &self,
        diff: &G,
        u: Tensor<R>,
        g: Tensor<R>,
        f: Tensor<R>,
    ) -> Tensor<1> {
        let dims = u.dims();
        assert!(
            dims == g.dims() && dims == f.dims(),
            "PoissonEqnLoss::grid: shapes differ"
        );
        assert!(
            R >= D,
            "PoissonEqnLoss::grid: rank {R} below the {D} spatial axes"
        );
        assert!(
            dims[R - D..].iter().all(|&n| n >= 3),
            "PoissonEqnLoss::grid: a spatial axis below 3 points: {dims:?}"
        );
        let interior: [std::ops::Range<usize>; R] = std::array::from_fn(|i| {
            if i >= R - D {
                1..dims[i] - 1
            } else {
                0..dims[i]
            }
        });

        let grad: [Tensor<R>; D] = std::array::from_fn(|a| diff.partial(u.clone(), a, 1));
        let lap = diff.laplacian(u.clone());
        let lhs = nonlinear_poisson_lhs(u.clone(), grad, lap);
        let interior_loss =
            mean_squared_error(lhs.slice(interior.clone()), f.slice(interior.clone()));

        let n_all: usize = dims.iter().product();
        let n_inner: usize = interior.iter().map(|r| r.len()).product();
        let inner_shape: [usize; R] = std::array::from_fn(|i| interior[i].len());
        let device = u.device();
        let error = (u - g).slice_assign(interior, Tensor::zeros(inner_shape, &device));
        let boundary_loss = error.square().sum().div_scalar((n_all - n_inner) as f64);

        self.weigh(interior_loss, boundary_loss)
    }

    fn weigh(&self, interior: Tensor<1>, boundary: Tensor<1>) -> Tensor<1> {
        interior.mul_scalar(self.interior_weight) + boundary.mul_scalar(self.boundary_weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Device;
    use burn::tensor::TensorData;
    use std::f64::consts::PI;

    /// A `[1, nx, ny]` grid function of the node coordinates.
    fn field(xs: &[f64], ys: &[f64], f: impl Fn(f64, f64) -> f64, device: &Device) -> Tensor<3> {
        let values = xs
            .iter()
            .flat_map(|&x| ys.iter().map(move |&y| (x, y)))
            .map(|(x, y)| f(x, y))
            .collect();
        Tensor::from_data(TensorData::new(values, [1, xs.len(), ys.len()]), device)
    }

    /// u = sin(πx) sin(πy) and its nonlinear Poisson left-hand side, in closed form.
    fn sine(x: f64, y: f64) -> f64 {
        (PI * x).sin() * (PI * y).sin()
    }
    fn sine_lhs(x: f64, y: f64) -> f64 {
        let u = sine(x, y);
        let ux = PI * (PI * x).cos() * (PI * y).sin();
        let uy = PI * (PI * x).sin() * (PI * y).cos();
        let lap = -2.0 * PI * PI * u;
        lap * (1.0 + 0.1 * u * u) + 0.2 * u * (ux * ux + uy * uy)
    }

    /// The interior part of `PoissonEqnLoss::grid` for `sine` on `xs` × `xs`.
    fn sine_interior_loss<G: GridDiff<2>>(diff: &G, xs: &[f64]) -> f64 {
        let device = Device::default();
        let u = field(xs, xs, sine, &device);
        let f = field(xs, xs, sine_lhs, &device);
        let eqn = PoissonEqnLoss {
            interior_weight: 1.0,
            boundary_weight: 0.0,
        };
        eqn.grid(diff, u.clone(), u, f).into_scalar::<f64>()
    }

    fn uniform(n: usize, a: f64, b: f64) -> Vec<f64> {
        (0..n)
            .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
            .collect()
    }

    /// Finite differences converge at second order on the interior: halving h divides the
    /// mean squared residual by about 16. The boundary part vanishes for g = u and is the
    /// mean of (u − g)² = 1 over the boundary nodes for g = u + 1.
    #[test]
    fn poisson_grid_finite_differences_second_order() {
        let loss = |n: usize| {
            let fd = FiniteDiff {
                h: [1.0 / (n - 1) as f64; 2],
                periodic: [false, false],
            };
            sine_interior_loss(&fd, &uniform(n, 0.0, 1.0))
        };
        let (coarse, fine) = (loss(17), loss(33));
        let ratio = coarse / fine;
        assert!(
            (13.0..19.0).contains(&ratio),
            "{coarse:e} / {fine:e} = {ratio}"
        );

        let device = Device::default();
        let xs = uniform(9, 0.0, 1.0);
        let u = field(&xs, &xs, sine, &device);
        let f = field(&xs, &xs, sine_lhs, &device);
        let fd = FiniteDiff {
            h: [1.0 / 8.0; 2],
            periodic: [false, false],
        };
        let boundary = PoissonEqnLoss {
            interior_weight: 0.0,
            boundary_weight: 1.0,
        };
        let same = boundary.grid(&fd, u.clone(), u.clone(), f.clone());
        assert_eq!(same.into_scalar::<f64>(), 0.0);
        let shifted = boundary.grid(&fd, u.clone(), u.add_scalar(1.0), f);
        assert!((shifted.into_scalar::<f64>() - 1.0).abs() < 1e-6);
    }

    /// The Chebyshev path is spectral: at 17 CGL nodes per axis its mean squared residual
    /// (5.6e-11 measured, f32, Flex) is far below that of finite differences at 65 points
    /// (5.4e-5).
    #[cfg(feature = "datagen")]
    #[test]
    fn poisson_grid_chebyshev_spectral() {
        use crate::neural_operators::chebyshev::nodes;
        use crate::neural_operators::losses::differentiation::MatrixDiff;
        let device = Device::default();
        let n = 17;
        let cgl = nodes(n).to_vec();
        let cheb = sine_interior_loss(&MatrixDiff::chebyshev([n, n], &device), &cgl);
        let fd = FiniteDiff {
            h: [2.0 / 64.0; 2],
            periodic: [false, false],
        };
        let finite = sine_interior_loss(&fd, &uniform(65, -1.0, 1.0));
        assert!(
            cheb < finite * 1e-4,
            "Chebyshev {cheb:e}, finite differences {finite:e}"
        );
    }

    /// Gradients flow through the Burgers loss and the grid Poisson loss.
    #[test]
    fn gradients_flow() {
        let device = Device::default().autodiff();
        let data: Vec<f64> = (0..2 * 8 * 6)
            .map(|k| ((k * 13 % 7) as f64).cos())
            .collect();
        let u: Tensor<4> = Tensor::from_data(TensorData::new(data, [2, 1, 8, 6]), &device);
        let u = u.require_grad();
        let loss = BurgersEqnLoss::default().loss(u.clone());
        let g = u
            .grad(&loss.backward())
            .unwrap()
            .abs()
            .max()
            .into_scalar::<f64>();
        assert!(g.is_finite() && g > 0.0, "{g}");

        let fd = FiniteDiff {
            h: [0.2; 2],
            periodic: [false, false],
        };
        let v: Tensor<3> = Tensor::ones([1, 6, 6], &device).require_grad();
        let eqn = PoissonEqnLoss {
            interior_weight: 1.0,
            boundary_weight: 1.0,
        };
        let zero = Tensor::zeros([1, 6, 6], &device);
        let loss = eqn.grid(&fd, v.clone(), zero.clone(), zero.add_scalar(1.0));
        let g = v
            .grad(&loss.backward())
            .unwrap()
            .abs()
            .max()
            .into_scalar::<f64>();
        assert!(g.is_finite() && g > 0.0, "{g}");
    }

    /// By hand: the first two points are the boundary, errors (1, −3): mean 5.
    #[test]
    fn boundary_loss_known_value() {
        let device = Device::default();
        let t = |v: [f64; 4]| Tensor::<1>::from_data(v, &device);
        let out = poisson_boundary_loss(t([1.0, -3.0, 9.0, 9.0]), t([0.0; 4]), 2);
        assert!((out.into_scalar::<f64>() - 5.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "5 boundary points of 4")]
    fn boundary_beyond_points_panics() {
        let t = Tensor::<1>::zeros([4], &Device::default());
        poisson_boundary_loss(t.clone(), t, 5);
    }
}
