//! [`cg`]: preconditioned conjugate gradients on a symmetric positive (semi-)definite
//! operator given as a closure, with Martens' stopping rule and the Lanczos estimates of
//! the operator's extreme eigenvalues.

use super::param_vec::ParamVec;

/// Settings of [`cg`].
#[derive(Clone, Debug)]
pub struct CgConfig {
    /// The most iterations (operator applications after the first) to take.
    pub max_iters: usize,
    /// Stop when ‖b − A x‖ / ‖b‖ falls to this.
    pub rel_tol: f64,
    /// Martens' relative-progress tolerance ε: stop at iteration i when
    /// q(xᵢ) < 0 and (q(xᵢ) − q(xᵢ₋ₖ)) / q(xᵢ) < k ε, with k = max(`progress_window`, ⌈i/10⌉)
    /// and q(x) = ½ xᵀA x − bᵀx (Martens 2010, §4.4). Zero disables the rule.
    pub progress_tol: f64,
    /// The smallest window k of the progress rule.
    pub progress_window: usize,
    /// Iterates are kept at iterations ⌈γʲ⌉, j = 0, 1, …, for backtracking; this is γ (> 1).
    pub keep_ratio: f64,
}

impl Default for CgConfig {
    /// 50 iterations, relative residual 1e-6, Martens' ε = 5e-4 with window 10, γ = 1.3.
    fn default() -> Self {
        Self {
            max_iters: 50,
            rel_tol: 1e-6,
            progress_tol: 5e-4,
            progress_window: 10,
            keep_ratio: 1.3,
        }
    }
}

/// Why [`cg`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CgStop {
    /// The relative residual reached [`CgConfig::rel_tol`].
    Converged,
    /// Martens' relative-progress rule fired.
    Progress,
    /// [`CgConfig::max_iters`] iterations were taken.
    MaxIters,
    /// A search direction p had pᵀA p ≤ 0: the operator is not positive definite there.
    /// The iterate before that direction is returned.
    NegativeCurvature,
    /// b = 0, so x = 0 solves the system; no iteration was taken.
    ZeroRhs,
}

/// One iterate [`cg`] kept for backtracking.
#[derive(Clone, Debug)]
pub struct CgIterate {
    /// The iteration it was reached at (0 for an unchanged warm start).
    pub iteration: usize,
    /// The iterate.
    pub x: ParamVec,
    /// The quadratic model q(x) = ½ xᵀA x − bᵀx there, from CG's recurrence residual.
    pub quadratic: f64,
}

/// The outcome of [`cg`].
#[derive(Clone, Debug)]
pub struct CgResult {
    /// The final iterate.
    pub x: ParamVec,
    /// Iterates for backtracking, in order: those at iterations ⌈γʲ⌉ (γ =
    /// [`CgConfig::keep_ratio`]), then the final iterate if it is not among them. The
    /// final iterate is always last: a warm start that already meets the tolerance gives
    /// one entry at iteration 0. Empty only if no iteration was taken from x = 0.
    pub iterates: Vec<CgIterate>,
    /// Iterations taken (operator applications, not counting the one for a warm start's
    /// initial residual).
    pub iterations: usize,
    /// ‖r‖ / ‖b‖ at the final iterate, r being CG's recurrence residual (no extra
    /// application). It equals b − A x in exact arithmetic and can drift from it in f32 or
    /// with a noisy (finite-difference) operator. 0 for [`CgStop::ZeroRhs`].
    pub rel_residual: f64,
    /// The smallest and largest Ritz values of the Lanczos tridiagonal built from CG's
    /// coefficients: estimates of the extreme eigenvalues of the preconditioned operator
    /// M⁻¹A on the Krylov space, inside its spectrum. `None` if no iteration completed.
    pub ritz: Option<(f64, f64)>,
    /// Why CG stopped.
    pub stop: CgStop,
}

/// Solves A x = b by preconditioned conjugate gradients.
///
/// - `apply` computes A v for a symmetric positive (semi-)definite A, e.g. the damped
///   Gauss–Newton curvature (JᵀJ + λI) v from a [`jvp`](super::jvp) and a
///   [`vjp`](super::vjp).
/// - `x0` is a warm start (e.g. the previous step); `None` starts from 0. A warm start
///   costs one extra application for its initial residual.
/// - `precond` is the diagonal of a preconditioner M ≈ A (applied as r / diag, so every
///   entry must be positive); `None` is M = I.
///
/// Scalars (α, β, the norms) are on the host in f64 from [`ParamVec`]'s reductions; the
/// vectors stay on their device.
///
/// # Panics
/// If `x0` or `precond` has a different structure from `b`, if an entry of `precond` is not positive,
/// or if `keep_ratio` ≤ 1.
pub fn cg(
    mut apply: impl FnMut(&ParamVec) -> ParamVec,
    b: &ParamVec,
    x0: Option<&ParamVec>,
    precond: Option<&ParamVec>,
    cfg: &CgConfig,
) -> CgResult {
    assert!(cfg.keep_ratio > 1.0, "cg: keep_ratio must exceed 1");
    if let Some(d) = precond {
        let min = d.to_flat().min().into_scalar::<f64>();
        assert!(
            min > 0.0,
            "cg: preconditioner entries must be positive, min {min}"
        );
    }
    let b_norm = b.norm();
    if b_norm == 0.0 {
        return CgResult {
            x: b.zeros_like(),
            iterates: Vec::new(),
            iterations: 0,
            rel_residual: 0.0,
            ritz: None,
            stop: CgStop::ZeroRhs,
        };
    }
    let precondition = |r: &ParamVec| match precond {
        Some(d) => r.div(d),
        None => r.clone(),
    };
    // q(x) = ½ xᵀA x − bᵀx = −½ (xᵀb + xᵀr) with r = b − A x: no extra application.
    let quadratic = |x: &ParamVec, r: &ParamVec| -0.5 * (x.dot(b) + x.dot(r));

    let (mut x, mut r) = match x0 {
        Some(x0) => (x0.clone(), b.axpy(-1.0, &apply(x0))),
        None => (b.zeros_like(), b.clone()),
    };
    let mut rel_residual = r.norm() / b_norm;
    let mut q_history = vec![quadratic(&x, &r)];
    let mut z = precondition(&r);
    let mut p = z.clone();
    let mut rz = r.dot(&z);
    let (mut alphas, mut betas) = (Vec::new(), Vec::new());
    let mut iterates = Vec::new();
    let mut next_keep = 1.0_f64;
    let mut stop = CgStop::MaxIters;

    if rel_residual <= cfg.rel_tol {
        stop = CgStop::Converged;
    } else {
        for i in 1..=cfg.max_iters {
            let ap = apply(&p);
            let p_ap = p.dot(&ap);
            if p_ap <= 0.0 {
                stop = CgStop::NegativeCurvature;
                break;
            }
            let alpha = rz / p_ap;
            x = x.axpy(alpha, &p);
            r = r.axpy(-alpha, &ap);
            alphas.push(alpha);
            rel_residual = r.norm() / b_norm;
            let q = quadratic(&x, &r);
            q_history.push(q);
            if i as f64 >= next_keep.ceil() {
                iterates.push(CgIterate {
                    iteration: i,
                    x: x.clone(),
                    quadratic: q,
                });
                while next_keep.ceil() <= i as f64 {
                    next_keep *= cfg.keep_ratio;
                }
            }

            if rel_residual <= cfg.rel_tol {
                stop = CgStop::Converged;
                break;
            }
            let k = cfg.progress_window.max(i.div_ceil(10));
            if cfg.progress_tol > 0.0
                && i > k
                && q < 0.0
                && (q - q_history[i - k]) / q < k as f64 * cfg.progress_tol
            {
                stop = CgStop::Progress;
                break;
            }
            if i == cfg.max_iters {
                break;
            }
            z = precondition(&r);
            let rz_next = r.dot(&z);
            let beta = rz_next / rz;
            rz = rz_next;
            betas.push(beta);
            p = z.axpy(beta, &p);
        }
    }

    let iterations = alphas.len();
    let final_kept = iterates.last().is_some_and(|it| it.iteration == iterations);
    if !final_kept && (iterations > 0 || x0.is_some()) {
        iterates.push(CgIterate {
            iteration: iterations,
            x: x.clone(),
            quadratic: q_history[iterations],
        });
    }
    CgResult {
        x,
        iterates,
        iterations,
        rel_residual,
        ritz: extreme_ritz_values(&alphas, &betas[..iterations.saturating_sub(1)]),
        stop,
    }
}

/// The Ritz values of a symmetric positive (semi-)definite A on the Krylov space of `v0`,
/// ascending: `iters` steps of Lanczos with full reorthogonalisation (each new vector is
/// orthogonalised twice against every earlier one), then the eigenvalues of the
/// tridiagonal. The largest converge first, to the top of A's spectrum; reorthogonalising
/// keeps a noisy or f32 operator from producing duplicate copies of them (ghosts), at the
/// cost of storing every Lanczos vector (`iters` × [`ParamVec`]). Stops early, with fewer
/// values, when the Krylov space is exhausted (the next vector's norm below 1e-10 of the
/// largest |αⱼ| so far, the Rayleigh quotients on the diagonal).
///
/// `apply` computes A v, as for [`cg`]; it is called `iters` times at most.
///
/// # Panics
/// If `iters` is 0 or `v0` is zero.
pub fn lanczos(
    mut apply: impl FnMut(&ParamVec) -> ParamVec,
    v0: &ParamVec,
    iters: usize,
) -> Vec<f64> {
    assert!(iters > 0, "lanczos: iters must be positive");
    let norm = v0.norm();
    assert!(norm > 0.0, "lanczos: v0 must be nonzero");
    let mut basis = vec![v0.scale(1.0 / norm)];
    let (mut alphas, mut betas) = (Vec::new(), Vec::new());
    let mut scale = 0.0_f64;
    for j in 0..iters {
        let mut w = apply(&basis[j]);
        let alpha = basis[j].dot(&w);
        alphas.push(alpha);
        scale = scale.max(alpha.abs());
        if j + 1 == iters {
            break;
        }
        // twice is enough (Kahan, "twice is enough"): the first pass leaves O(ε) residue
        for _ in 0..2 {
            for q in &basis {
                w = w.axpy(-q.dot(&w), q);
            }
        }
        let beta = w.norm();
        if beta <= 1e-10 * scale {
            break;
        }
        betas.push(beta);
        basis.push(w.scale(1.0 / beta));
    }
    (0..alphas.len())
        .map(|i| tridiagonal_eigenvalue(&alphas, &betas, i))
        .collect()
}

/// The smallest and largest eigenvalues of the Lanczos tridiagonal of k CG steps:
/// diagonal 1/α₁ and 1/αⱼ + βⱼ₋₁/αⱼ₋₁ (j ≥ 2), off-diagonal √βⱼ / αⱼ. `betas` holds
/// k − 1 values.
fn extreme_ritz_values(alphas: &[f64], betas: &[f64]) -> Option<(f64, f64)> {
    if alphas.is_empty() {
        return None;
    }
    debug_assert_eq!(betas.len() + 1, alphas.len());
    let diag: Vec<f64> = (0..alphas.len())
        .map(|j| {
            let tail = if j == 0 {
                0.0
            } else {
                betas[j - 1] / alphas[j - 1]
            };
            1.0 / alphas[j] + tail
        })
        .collect();
    let off: Vec<f64> = (0..betas.len())
        .map(|j| betas[j].sqrt() / alphas[j])
        .collect();
    Some((
        tridiagonal_eigenvalue(&diag, &off, 0),
        tridiagonal_eigenvalue(&diag, &off, diag.len() - 1),
    ))
}

/// The `index`-th smallest eigenvalue of the symmetric tridiagonal matrix with diagonal
/// `diag` and off-diagonal `off`, by bisection on the Sturm count inside the Gershgorin
/// interval.
fn tridiagonal_eigenvalue(diag: &[f64], off: &[f64], index: usize) -> f64 {
    let n = diag.len();
    let radius = |j: usize| {
        let left = if j > 0 { off[j - 1].abs() } else { 0.0 };
        let right = if j + 1 < n { off[j].abs() } else { 0.0 };
        left + right
    };
    let mut lo = (0..n)
        .map(|j| diag[j] - radius(j))
        .fold(f64::INFINITY, f64::min);
    let mut hi = (0..n)
        .map(|j| diag[j] + radius(j))
        .fold(f64::NEG_INFINITY, f64::max);
    // The number of eigenvalues below x: the negative pivots of T − xI (Sturm).
    let count_below = |x: f64| {
        let mut count = 0;
        let mut d = 1.0;
        for j in 0..n {
            let coupling = if j > 0 {
                off[j - 1] * off[j - 1] / d
            } else {
                0.0
            };
            d = diag[j] - x - coupling;
            if d == 0.0 {
                d = -f64::EPSILON * (x.abs() + 1.0);
            }
            if d < 0.0 {
                count += 1;
            }
        }
        count
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if count_below(mid) > index {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::optim::gauss_newton::params;
    use burn::{
        nn::{Linear, LinearConfig},
        prelude::*,
    };

    /// A = H diag(λ) H with the Householder reflection H = I − 2wwᵀ/‖w‖²: symmetric with
    /// known eigenvalues λ and no zero entries. n = 8, the size of a `Linear(3, 2)`.
    const EIGENVALUES: [f64; 8] = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0];

    fn matrix() -> Vec<Vec<f64>> {
        let n = EIGENVALUES.len();
        let w: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * i as f64).collect();
        let ww: f64 = w.iter().map(|x| x * x).sum();
        let h = |i: usize, j: usize| f64::from(u8::from(i == j)) - 2.0 * w[i] * w[j] / ww;
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (0..n).map(|k| h(i, k) * EIGENVALUES[k] * h(k, j)).sum())
                    .collect()
            })
            .collect()
    }

    /// A vector with a `Linear(3, 2)`'s structure, so `ParamVec` has two parts.
    fn template(device: &Device) -> ParamVec {
        let model: Linear = LinearConfig::new(3, 2).init(device);
        params(&model)
    }

    fn vector(template: &ParamVec, values: &[f64], device: &Device) -> ParamVec {
        let values: Vec<f32> = values.iter().map(|x| *x as f32).collect();
        template.from_flat_like(Tensor::from_floats(values.as_slice(), device))
    }

    fn host(v: &ParamVec) -> Vec<f64> {
        v.to_flat()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// v ↦ (A + λI) v, on the host in f64 then back.
    fn operator<'a>(
        a: &'a [Vec<f64>],
        lambda: f64,
        template: &'a ParamVec,
        device: &'a Device,
    ) -> impl FnMut(&ParamVec) -> ParamVec + 'a {
        move |v| {
            let v = host(v);
            let av: Vec<f64> = a
                .iter()
                .zip(&v)
                .map(|(row, vi)| row.iter().zip(&v).map(|(x, y)| x * y).sum::<f64>() + lambda * vi)
                .collect();
            vector(template, &av, device)
        }
    }

    fn solution() -> Vec<f64> {
        (0..EIGENVALUES.len())
            .map(|i| (i as f64 * 0.7).cos())
            .collect()
    }

    fn apply_host(a: &[Vec<f64>], lambda: f64, x: &[f64]) -> Vec<f64> {
        a.iter()
            .zip(x)
            .map(|(row, xi)| row.iter().zip(x).map(|(p, q)| p * q).sum::<f64>() + lambda * xi)
            .collect()
    }

    fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).fold(0.0, |m, (x, y)| m.max((x - y).abs()))
    }

    fn exact(max_iters: usize) -> CgConfig {
        CgConfig {
            max_iters,
            rel_tol: 1e-6,
            progress_tol: 0.0,
            ..CgConfig::default()
        }
    }

    /// Lanczos on the 8 × 8 matrix of known eigenvalues: 8 steps recover all of them,
    /// with no duplicates; 3 steps give the largest first and stay inside the spectrum.
    #[test]
    fn lanczos_recovers_the_known_spectrum() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let v0 = vector(&t, &[1.0; 8], &device);
        let ritz = lanczos(operator(&a, 0.0, &t, &device), &v0, 8);
        assert_eq!(ritz.len(), 8);
        for (r, e) in ritz.iter().zip(EIGENVALUES) {
            assert!((r - e).abs() < 1e-3 * e, "{ritz:?}");
        }
        let few = lanczos(operator(&a, 0.0, &t, &device), &v0, 3);
        assert_eq!(few.len(), 3);
        assert!(few.windows(2).all(|w| w[0] <= w[1]));
        assert!(
            (few[2] - 21.0).abs() < 0.5 && few[0] >= 0.5 - 1e-6,
            "{few:?}"
        );
        // the Krylov space of an 8 × 8 matrix is exhausted after 8 steps: 12 asked, 8 given
        let all = lanczos(operator(&a, 0.0, &t, &device), &v0, 12);
        assert_eq!(all.len(), 8, "{all:?}");
        assert!((all[7] - 21.0).abs() < 1e-3 * 21.0, "{all:?}");
    }

    #[test]
    fn cg_reaches_the_known_solution() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let x_star = solution();
        let b = vector(&t, &apply_host(&a, 0.0, &x_star), &device);
        let result = cg(operator(&a, 0.0, &t, &device), &b, None, None, &exact(50));
        assert_eq!(result.stop, CgStop::Converged);
        assert!(result.iterations <= 2 * EIGENVALUES.len());
        assert!(max_abs_diff(&host(&result.x), &x_star) < 1e-4);
        assert_eq!(result.iterates.last().unwrap().iteration, result.iterations);

        // every kept iterate's q equals ½ xᵀA x − bᵀx computed on the host
        let b_host = host(&b);
        for it in &result.iterates {
            let x = host(&it.x);
            let ax = apply_host(&a, 0.0, &x);
            let q: f64 = x
                .iter()
                .zip(&ax)
                .map(|(xi, axi)| 0.5 * xi * axi)
                .sum::<f64>()
                - x.iter().zip(&b_host).map(|(xi, bi)| xi * bi).sum::<f64>();
            assert!(
                (it.quadratic - q).abs() < 1e-4 * q.abs().max(1.0),
                "iteration {}: {} vs {q}",
                it.iteration,
                it.quadratic
            );
        }
    }

    #[test]
    fn damped_cg_solves_the_damped_system() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let (lambda, x_star) = (2.5, solution());
        let b = vector(&t, &apply_host(&a, lambda, &x_star), &device);
        let damped = cg(
            operator(&a, lambda, &t, &device),
            &b,
            None,
            None,
            &exact(50),
        );
        assert!(max_abs_diff(&host(&damped.x), &x_star) < 1e-4);
        let undamped = cg(operator(&a, 0.0, &t, &device), &b, None, None, &exact(50));
        assert!(max_abs_diff(&host(&undamped.x), &x_star) > 1e-2);
    }

    #[test]
    fn ritz_values_bracket_and_converge_to_the_extreme_eigenvalues() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let b = vector(&t, &[1.0; 8], &device);
        let (lo, hi) = (EIGENVALUES[0], EIGENVALUES[7]);
        let mut previous = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 1..=EIGENVALUES.len() {
            let cfg = CgConfig {
                rel_tol: 0.0,
                ..exact(k)
            };
            let result = cg(operator(&a, 0.0, &t, &device), &b, None, None, &cfg);
            let (min, max) = result.ritz.unwrap();
            assert!(
                min >= lo * (1.0 - 1e-3) && max <= hi * (1.0 + 1e-3),
                "k={k}: {min}, {max}"
            );
            assert!(min <= previous.0 * (1.0 + 1e-3) && max >= previous.1 * (1.0 - 1e-3));
            previous = (min, max);
        }
        assert!((previous.0 - lo).abs() < 1e-2 * lo, "{}", previous.0);
        assert!((previous.1 - hi).abs() < 1e-3 * hi, "{}", previous.1);
    }

    #[test]
    fn preconditioning_warm_start_and_zero_rhs() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let x_star = solution();
        let b = vector(&t, &apply_host(&a, 0.0, &x_star), &device);

        let diag: Vec<f64> = (0..8).map(|i| a[i][i]).collect();
        let pre = vector(&t, &diag, &device);
        let result = cg(
            operator(&a, 0.0, &t, &device),
            &b,
            None,
            Some(&pre),
            &exact(50),
        );
        assert_eq!(result.stop, CgStop::Converged);
        assert!(max_abs_diff(&host(&result.x), &x_star) < 1e-4);

        let warm = vector(&t, &x_star, &device);
        let result = cg(
            operator(&a, 0.0, &t, &device),
            &b,
            Some(&warm),
            None,
            &exact(50),
        );
        assert!(result.iterations <= 1, "{}", result.iterations);

        let zero = cg(
            operator(&a, 0.0, &t, &device),
            &b.zeros_like(),
            None,
            None,
            &exact(50),
        );
        assert_eq!((zero.stop, zero.iterations), (CgStop::ZeroRhs, 0));
    }

    #[test]
    fn negative_curvature_and_iterate_schedule() {
        let device = Device::default();
        let t = template(&device);
        let b = vector(&t, &[1.0; 8], &device);
        let result = cg(|v: &ParamVec| v.scale(-1.0), &b, None, None, &exact(50));
        assert_eq!(
            (result.stop, result.iterations),
            (CgStop::NegativeCurvature, 0)
        );

        let a = matrix();
        let cfg = CgConfig {
            rel_tol: 0.0,
            ..exact(8)
        };
        let result = cg(operator(&a, 0.0, &t, &device), &b, None, None, &cfg);
        let kept: Vec<usize> = result.iterates.iter().map(|it| it.iteration).collect();
        // ⌈1.3ʲ⌉: 1, 2, 3, 4, 5, 7, then the final 8.
        assert_eq!(kept, [1, 2, 3, 4, 5, 7, 8]);
    }

    #[test]
    #[should_panic(expected = "preconditioner entries must be positive")]
    fn cg_rejects_a_non_positive_preconditioner() {
        let device = Device::default();
        let (a, t) = (matrix(), template(&device));
        let b = vector(&t, &[1.0; 8], &device);
        let mut diag = [1.0; 8];
        diag[3] = 0.0;
        let pre = vector(&t, &diag, &device);
        let _ = cg(
            operator(&a, 0.0, &t, &device),
            &b,
            None,
            Some(&pre),
            &exact(50),
        );
    }

    #[test]
    fn tridiagonal_eigenvalues_match_a_known_matrix() {
        // tridiag(-1, 2, -1) of size n: eigenvalues 2 − 2cos(kπ/(n+1)).
        let n = 6;
        let (diag, off) = (vec![2.0; n], vec![-1.0; n - 1]);
        let eig = |k: usize| 2.0 - 2.0 * (k as f64 * std::f64::consts::PI / (n + 1) as f64).cos();
        assert!((tridiagonal_eigenvalue(&diag, &off, 0) - eig(1)).abs() < 1e-12);
        assert!((tridiagonal_eigenvalue(&diag, &off, n - 1) - eig(n)).abs() < 1e-12);
    }
}
