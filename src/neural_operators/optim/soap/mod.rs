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

    fn step<const D: usize>(
        &self,
        _lr: LearningRate,
        _tensor: Tensor<D>,
        _grad: Tensor<D>,
        _state: Option<Self::State<D>>,
    ) -> (Tensor<D>, Option<Self::State<D>>) {
        todo!()
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
