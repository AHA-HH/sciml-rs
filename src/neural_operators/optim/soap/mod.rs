//! SOAP (Vyas et al. 2024, arXiv 2409.11321, Algorithm 3): Adam run in the eigenbasis of
//! Shampoo's preconditioner, as Burn's per-parameter [`Optimizer`] (optimisers CO3.2;
//! proposal `docs/design/optimisers.md` §5).
//!
//! For a parameter of any rank D with gradient G, every dimension k with
//! dₖ ≤ `max_precond_dim` (and D ≥ 2) gets a factor Lₖ, the moving average of G's mode-k Gram
//! matrix, and an eigenbasis Qₖ of it (decision D8, option A: one factor per dimension; the
//! FNO's `weights_re` and `weights_im` are separate parameters, so each has its own). A
//! dimension without a factor keeps Qₖ = I, so a bias (rank 1) gets Adam. Each step
//! projects G into the bases, takes Adam's step there and projects it back; the bases are
//! refreshed every `precondition_frequency` steps by one power step and a QR.
//!
//! The reference implementation is nikhilvyas/SOAP @a1e5535 (`soap.py`); this follows it
//! except that the first moment is kept in the original space and projected each step (the
//! paper's Algorithm 3; the reference keeps it in the basis and rotates it each step, the
//! same up to round-off). Its options `merge_dims`, `precondition_1d`, `normalize_grads`
//! and `correct_bias` are at their defaults (off, off, off, on) and not offered.
//!
//! Linear algebra on the host for every backend (decision D9): the first basis by Burn's
//! `linalg::svd` (host code), the refreshes by a Householder QR in f64 here
//! (`spikes/linalg/REPORT.md`: Burn's device `qr` is 18–240× slower than a host round trip
//! on metal and cuda at these sizes).

use burn::{
    config::Config,
    optim::{LearningRate, ModuleOptimizer, Optimizer, RecordState, StateSink, StateSource},
    prelude::*,
    store::burn_pack::Scalar,
    tensor::{Int, TensorData},
};

#[cfg(test)]
mod tests;

/// The QR sweeps allowed to Burn's host `svd` per singular value (as in CO3.1's spike).
const SVD_SWEEPS: usize = 30;

/// SOAP's hyperparameters; the defaults are the reference implementation's. The learning
/// rate is not here: Burn passes it to every step (the trainer's schedule).
#[derive(Config, Debug)]
pub struct SoapConfig {
    /// Adam's β₁ (first moment).
    #[config(default = 0.95)]
    pub beta_1: f32,
    /// Adam's β₂ (second moment).
    #[config(default = 0.95)]
    pub beta_2: f32,
    /// The factors' moving-average rate; `None` uses β₂ (the reference's −1).
    pub shampoo_beta: Option<f32>,
    /// Added to √V in Adam's denominator.
    #[config(default = 1e-8)]
    pub epsilon: f32,
    /// Decoupled weight decay: W ← W (1 − lr · weight_decay) after each step.
    #[config(default = 0.01)]
    pub weight_decay: f32,
    /// Steps between basis refreshes (f).
    #[config(default = 10)]
    pub precondition_frequency: usize,
    /// A dimension longer than this gets no factor (its basis stays the identity); 0 makes
    /// SOAP Adam.
    #[config(default = 10000)]
    pub max_precond_dim: usize,
}

impl SoapConfig {
    /// The bare [`Soap`], for [`ModuleOptimizer::with_group`]; [`init`](Self::init) is
    /// the whole-module optimiser.
    pub fn build(&self) -> Soap {
        Soap {
            config: self.clone(),
        }
    }

    /// The whole-module optimiser.
    pub fn init(&self) -> ModuleOptimizer {
        ModuleOptimizer::from(self.build())
    }
}

/// The SOAP optimiser (see the [module documentation](self)).
#[derive(Clone, Debug)]
pub struct Soap {
    config: SoapConfig,
}

/// One dimension's preconditioner: its factor and that factor's eigenbasis.
#[derive(Clone, Debug)]
pub struct Factor {
    /// Lₖ, the moving average of the mode-k Gram matrix, shape `[dₖ, dₖ]`.
    pub gram: Tensor<2>,
    /// Qₖ, orthonormal columns in descending order of their eigenvalue estimates,
    /// shape `[dₖ, dₖ]`.
    pub basis: Tensor<2>,
}

/// SOAP's state for one parameter of rank D.
#[derive(Clone, Debug)]
pub struct SoapState<const D: usize> {
    /// Adam steps taken (the first call takes none).
    pub step: usize,
    /// Adam's first moment M, in the original space, the parameter's shape.
    pub moment_1: Tensor<D>,
    /// Adam's second moment V, in the bases, the parameter's shape.
    pub moment_2: Tensor<D>,
    /// Per dimension, its [`Factor`], or `None` where the basis is the identity; length D.
    pub factors: Vec<Option<Factor>>,
}

impl Optimizer for Soap {
    type State<const D: usize> = SoapState<D>;

    /// One SOAP step for `tensor` with gradient `grad`, both of any shape.
    ///
    /// The first call (no state) only builds the factors and their first bases and returns
    /// `tensor` unchanged, as the reference does, so a gradient never enters its own
    /// projection. Later calls take Adam's step in the bases, then decay the weights, then
    /// update the factors, and every `precondition_frequency`-th step refresh the bases.
    ///
    /// # Panics
    /// If Burn's `svd` does not converge on a first factor.
    fn step<const D: usize>(
        &self,
        lr: LearningRate,
        tensor: Tensor<D>,
        grad: Tensor<D>,
        state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        let Some(mut state) = state else {
            return (tensor, Some(self.first_state(grad)));
        };
        let c = &self.config;
        let (b1, b2) = (f64::from(c.beta_1), f64::from(c.beta_2));

        let grad_rot = project(grad.clone(), &state.factors, false);
        state.step += 1;
        state.moment_1 = state
            .moment_1
            .mul_scalar(b1)
            .add(grad.clone().mul_scalar(1.0 - b1));
        state.moment_2 = state
            .moment_2
            .mul_scalar(b2)
            .add(grad_rot.square().mul_scalar(1.0 - b2));

        let t = state.step as i32;
        let step_size = lr * (1.0 - b2.powi(t)).sqrt() / (1.0 - b1.powi(t));
        let moment_1_rot = project(state.moment_1.clone(), &state.factors, false);
        let normalised = moment_1_rot.div(state.moment_2.clone().sqrt().add_scalar(c.epsilon));
        let direction = project(normalised, &state.factors, true);
        let mut tensor = tensor - direction.mul_scalar(step_size);
        if c.weight_decay > 0.0 {
            tensor = tensor.mul_scalar(1.0 - lr * f64::from(c.weight_decay));
        }

        let beta_s = f64::from(c.shampoo_beta.unwrap_or(c.beta_2));
        for (k, f) in state.factors.iter_mut().enumerate() {
            if let Some(f) = f {
                f.gram = f
                    .gram
                    .clone()
                    .mul_scalar(beta_s)
                    .add(mode_gram(grad.clone(), k).mul_scalar(1.0 - beta_s));
            }
        }
        if state.step % c.precondition_frequency == 0 {
            refresh(&mut state);
        }
        (tensor, Some(state))
    }

    fn to_device<const D: usize>(state: Self::State<D>, device: &Device) -> Self::State<D> {
        SoapState {
            step: state.step,
            moment_1: state.moment_1.to_device(device),
            moment_2: state.moment_2.to_device(device),
            factors: state
                .factors
                .into_iter()
                .map(|f| {
                    f.map(|f| Factor {
                        gram: f.gram.to_device(device),
                        basis: f.basis.to_device(device),
                    })
                })
                .collect(),
        }
    }
}

impl Soap {
    /// The state after the first call: zero moments, and for each dimension that gets one,
    /// the factor (1 − β_s) Gₖ Gₖᵀ with its eigenbasis from Burn's host `svd` (decision D9;
    /// for a symmetric positive semidefinite matrix the left singular vectors are
    /// eigenvectors, in descending order).
    fn first_state<const D: usize>(&self, grad: Tensor<D>) -> SoapState<D> {
        let c = &self.config;
        let beta_s = f64::from(c.shampoo_beta.unwrap_or(c.beta_2));
        let dims = grad.dims();
        let factors = (0..D)
            .map(|k| {
                (D >= 2 && dims[k] <= c.max_precond_dim).then(|| {
                    let gram = mode_gram(grad.clone(), k).mul_scalar(1.0 - beta_s);
                    let (basis, _, _) = burn::linalg::svd::<2, 1>(gram.clone(), SVD_SWEEPS);
                    Factor { gram, basis }
                })
            })
            .collect();
        SoapState {
            step: 0,
            moment_1: grad.zeros_like(),
            moment_2: grad.zeros_like(),
            factors,
        }
    }
}

/// The mode-k Gram matrix Gₖ Gₖᵀ of `x`, shape `[dₖ, dₖ]`.
fn mode_gram<const D: usize>(x: Tensor<D>, k: usize) -> Tensor<2> {
    let d = x.dims()[k];
    let unfolded: Tensor<2> = x.swap_dims(0, k).reshape([d as i32, -1]);
    unfolded.clone().matmul(unfolded.transpose())
}

/// `x` projected into the bases (`back` false: the mode-k product with Qₖᵀ for every k with
/// a factor) or back out of them (`back` true: with Qₖ), the reference's `tensordot` loop:
/// each pass contracts the leading dimension and moves it last, so D passes restore the
/// order. Identity everywhere returns `x` untouched.
fn project<const D: usize>(mut x: Tensor<D>, factors: &[Option<Factor>], back: bool) -> Tensor<D> {
    if factors.iter().all(Option::is_none) {
        return x;
    }
    for f in factors {
        let mut dims = x.dims();
        let rows: Tensor<2> = x.reshape([dims[0] as i32, -1]).transpose();
        let rows = match f {
            Some(f) if back => rows.matmul(f.basis.clone().transpose()),
            Some(f) => rows.matmul(f.basis.clone()),
            None => rows,
        };
        dims.rotate_left(1);
        x = rows.reshape(dims);
    }
    x
}

/// Every basis refreshed on the host (decision D9; reference `get_orthogonal_matrix_QR`):
/// the columns of Qₖ sorted by the estimates diag(Qₖᵀ Lₖ Qₖ), descending, V permuted along k
/// to follow, then Qₖ ← the Q of a QR of Lₖ Qₖ (one power step).
fn refresh<const D: usize>(state: &mut SoapState<D>) {
    let device = state.moment_2.device();
    for (k, f) in state.factors.iter_mut().enumerate() {
        let Some(f) = f else { continue };
        let n = f.gram.dims()[0];
        let (order, basis) = host::refresh(&host::read(&f.gram), &host::read(&f.basis), n);
        f.basis = Tensor::from_data(TensorData::new(basis, [n, n]), &device);
        let order: Vec<i64> = order.into_iter().map(|i| i as i64).collect();
        let order = Tensor::<1, Int>::from_data(TensorData::new(order, [n]), &device);
        state.moment_2 = state.moment_2.clone().select(k, order);
    }
}

/// The refresh's host arithmetic, in f64.
mod host {
    use burn::prelude::*;

    /// A square f32 matrix to the host, row-major, in f64.
    pub(super) fn read(t: &Tensor<2>) -> Vec<f64> {
        t.clone()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .expect("an f32 matrix")
            .into_iter()
            .map(f64::from)
            .collect()
    }

    /// For n × n row-major `gram` (L) and `basis` (Q): the column order by descending
    /// diag(QᵀLQ) and the Q of a Householder QR of L Q[:, order], rounded to f32.
    pub(super) fn refresh(gram: &[f64], basis: &[f64], n: usize) -> (Vec<usize>, Vec<f32>) {
        let lq = matmul(gram, basis, n);
        let estimate: Vec<f64> = (0..n)
            .map(|j| (0..n).map(|r| basis[r * n + j] * lq[r * n + j]).sum())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| estimate[b].total_cmp(&estimate[a]));
        let sorted: Vec<f64> = (0..n * n)
            .map(|idx| basis[(idx / n) * n + order[idx % n]])
            .collect();
        let q = householder_q(matmul(gram, &sorted, n), n);
        (order, q.into_iter().map(|v| v as f32).collect())
    }

    fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for p in 0..n {
                let a_ip = a[i * n + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b[p * n + j];
                }
            }
        }
        c
    }

    /// Q of the Householder QR of an n × n row-major matrix: orthogonal even when the
    /// matrix is rank-deficient (a zero column leaves its reflector out).
    fn householder_q(mut a: Vec<f64>, n: usize) -> Vec<f64> {
        let mut reflectors = Vec::with_capacity(n);
        for j in 0..n {
            let mut v: Vec<f64> = (j..n).map(|i| a[i * n + j]).collect();
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            v[0] += if v[0] >= 0.0 { norm } else { -norm };
            let vv: f64 = v.iter().map(|x| x * x).sum();
            if vv == 0.0 {
                reflectors.push(None);
                continue;
            }
            reflect(&mut a, &v, vv, j, n);
            reflectors.push(Some((v, vv)));
        }
        let mut q: Vec<f64> = (0..n * n)
            .map(|idx| if idx / n == idx % n { 1.0 } else { 0.0 })
            .collect();
        for (j, r) in reflectors.iter().enumerate().rev() {
            if let Some((v, vv)) = r {
                reflect(&mut q, v, *vv, j, n);
            }
        }
        q
    }

    /// Rows j.. of the n × n row-major `m` ← (I − 2 v vᵀ / vᵀv) times them.
    fn reflect(m: &mut [f64], v: &[f64], vv: f64, j: usize, n: usize) {
        for col in 0..n {
            let dot: f64 = (j..n).map(|i| v[i - j] * m[i * n + col]).sum();
            let s = 2.0 * dot / vv;
            for i in j..n {
                m[i * n + col] -= s * v[i - j];
            }
        }
    }
}

impl<const D: usize> RecordState for SoapState<D> {
    fn state_flatten(&self, prefix: &str, out: &mut StateSink) {
        out.push_scalar(prefix, "step", Scalar::from(self.step));
        out.push_tensor(prefix, "moment_1", self.moment_1.clone().into_data());
        out.push_tensor(prefix, "moment_2", self.moment_2.clone().into_data());
        for (k, f) in self.factors.iter().enumerate() {
            if let Some(f) = f {
                out.push_tensor(prefix, &format!("gram.{k}"), f.gram.clone().into_data());
                out.push_tensor(prefix, &format!("basis.{k}"), f.basis.clone().into_data());
            }
        }
    }

    fn state_unflatten(prefix: &str, src: &mut StateSource, device: &Device) -> Option<Self> {
        let step = usize::try_from(src.take_scalar(prefix, "step")?).ok()?;
        let moment_1 = Tensor::from_data(src.take_tensor(prefix, "moment_1")?, device);
        let moment_2 = Tensor::from_data(src.take_tensor(prefix, "moment_2")?, device);
        let factors = (0..D)
            .map(|k| {
                let gram = src.take_tensor(prefix, &format!("gram.{k}"));
                let basis = src.take_tensor(prefix, &format!("basis.{k}"));
                match (gram, basis) {
                    (Some(gram), Some(basis)) => Some(Some(Factor {
                        gram: Tensor::from_data(gram, device),
                        basis: Tensor::from_data(basis, device),
                    })),
                    (None, None) => Some(None),
                    _ => None,
                }
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            step,
            moment_1,
            moment_2,
            factors,
        })
    }
}
