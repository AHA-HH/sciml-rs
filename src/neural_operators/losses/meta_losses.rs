//! The loss aggregators of `neuraloperator` 2.0.0 (`neuralop/losses/meta_losses.py`;
//! library CM1.6b): fixed weighted sums, per-field losses, and the adaptive weightings
//! SoftAdapt and ReLoBRaLo.
//!
//! The adaptive weights are computed from the losses' values without gradient, as 2.0.0
//! does under `torch.no_grad`: the aggregate is Σ λᵢ Lᵢ with the λᵢ constants, so its
//! gradient is Σ λᵢ ∇Lᵢ. Their state (previous and initial losses, the EMA of the weights)
//! lives on the host in f64, read from the scalar losses each step.

use std::ops::Range;

use burn::tensor::Tensor;
use rand_chacha::{
    ChaCha8Rng,
    rand_core::{Rng, SeedableRng},
};

/// A loss of a prediction against a target, both of rank `R`, returning `[1]`: any of the
/// crate's losses behind a closure, e.g. `Box::new(mse_loss)`.
pub type LossFn<const R: usize> = Box<dyn Fn(Tensor<R>, Tensor<R>) -> Tensor<1>>;

/// 2.0.0's `WeightedSumLoss`: Σᵢ wᵢ Lᵢ(pred, truth).
pub struct WeightedSumLoss<const R: usize> {
    /// Each loss with its weight.
    pub terms: Vec<(LossFn<R>, f64)>,
}

impl<const R: usize> WeightedSumLoss<R> {
    /// The losses with `weights`, or 1/n each when `None` (2.0.0's default).
    ///
    /// # Panics
    /// If `weights` is given with a length other than the number of losses.
    pub fn new(losses: Vec<LossFn<R>>, weights: Option<Vec<f64>>) -> Self {
        let n = losses.len();
        let weights = weights.unwrap_or_else(|| vec![1.0 / n as f64; n]);
        assert_eq!(
            weights.len(),
            n,
            "WeightedSumLoss: each loss must have a weight"
        );
        Self {
            terms: losses.into_iter().zip(weights).collect(),
        }
    }

    /// The weighted sum on `pred` and `truth` (any shape the losses take). Returns `[1]`.
    ///
    /// # Panics
    /// If there are no losses, or as the losses do.
    pub fn loss(&self, pred: Tensor<R>, truth: Tensor<R>) -> Tensor<1> {
        self.terms
            .iter()
            .map(|(loss, w)| loss(pred.clone(), truth.clone()).mul_scalar(*w))
            .reduce(|a, b| a + b)
            .expect("WeightedSumLoss: no losses")
    }
}

/// One output field of a [`FieldwiseAggregatorLoss`]: its channels on one axis and its loss.
pub struct Field<const R: usize> {
    /// The field's name (2.0.0's dict key).
    pub name: String,
    /// The axis the channels are on.
    pub axis: usize,
    /// The field's channels on that axis.
    pub channels: Range<usize>,
    /// The loss of the field's prediction against its target.
    pub loss: LossFn<R>,
}

/// 2.0.0's `FieldwiseAggregatorLoss`: the mean over fields of each field's loss on its slice
/// of the prediction and the target.
///
/// 2.0.0 indexes with any tuple of slices and reshapes the prediction, not the target, to
/// `(-1, 1)`; the two agree only for one-channel slices of `(N, C)` outputs, its use. Here
/// a field is a channel range on one axis, and prediction and target are sliced alike and
/// keep their shapes. 2.0.0's `logging` (which returns only the last field's loss) is
/// [`field_losses`](Self::field_losses).
pub struct FieldwiseAggregatorLoss<const R: usize> {
    /// The fields.
    pub fields: Vec<Field<R>>,
}

impl<const R: usize> FieldwiseAggregatorLoss<R> {
    /// The aggregator over `fields`.
    ///
    /// # Panics
    /// If `fields` is empty.
    pub fn new(fields: Vec<Field<R>>) -> Self {
        assert!(!fields.is_empty(), "FieldwiseAggregatorLoss: no fields");
        Self { fields }
    }

    /// Each field's loss, in order. `pred`, `truth`: the same shape. Returns `[1]` each.
    ///
    /// # Panics
    /// If a field's axis or channels lie outside the tensors, or as its loss does.
    pub fn field_losses(&self, pred: Tensor<R>, truth: Tensor<R>) -> Vec<Tensor<1>> {
        self.fields
            .iter()
            .map(|f| {
                let len = f.channels.len();
                let slice = |t: Tensor<R>| t.narrow(f.axis, f.channels.start, len);
                (f.loss)(slice(pred.clone()), slice(truth.clone()))
            })
            .collect()
    }

    /// The mean of [`field_losses`](Self::field_losses). Returns `[1]`.
    ///
    /// # Panics
    /// As [`field_losses`](Self::field_losses).
    pub fn loss(&self, pred: Tensor<R>, truth: Tensor<R>) -> Tensor<1> {
        let n = self.fields.len() as f64;
        sum(self.field_losses(pred, truth)).mul_scalar(1.0 / n)
    }
}

/// 2.0.0's `SoftAdapt` (Heydari et al. 2019).
///
/// Step 0 returns Σ Lᵢ and weights 1. Each later step, with Lᵢ the (weighted) losses and
/// Pᵢ the previous step's, λᵢ = exp(Lᵢ / (Pᵢ + eps) − maxⱼ Lⱼ / Pⱼ), and the aggregate is
/// n Σ λᵢ Lᵢ / (Σ λᵢ + eps). The λᵢ returned are before that normalisation, as 2.0.0's.
#[derive(Clone, Debug)]
pub struct SoftAdapt {
    /// Guards the divisions, default 1e-8.
    pub eps: f64,
    /// Fixed per-loss weights applied first (2.0.0's `weights` dict, by position; `None`
    /// is all 1).
    pub weights: Option<Vec<f64>>,
    num_losses: usize,
    prev: Vec<f64>,
}

impl SoftAdapt {
    /// SoftAdapt over `num_losses` losses, eps 1e-8.
    ///
    /// # Panics
    /// If `weights` is given with a length other than `num_losses`.
    pub fn new(num_losses: usize, weights: Option<Vec<f64>>) -> Self {
        check_weights(&weights, num_losses);
        Self {
            eps: 1e-8,
            weights,
            num_losses,
            prev: Vec::new(),
        }
    }

    /// The aggregate of `losses` (`[1]` each) at `step`, and the weights λ.
    ///
    /// # Panics
    /// If the number of losses is not `num_losses`, or `step > 0` before any step 0.
    pub fn forward(&mut self, losses: Vec<Tensor<1>>, step: usize) -> (Tensor<1>, Vec<f64>) {
        let losses = weigh(losses, &self.weights, self.num_losses);
        let values = scalars(&losses);
        if step == 0 {
            self.prev = values;
            return (sum(losses), vec![1.0; self.num_losses]);
        }
        assert!(
            !self.prev.is_empty(),
            "SoftAdapt: step {step} before step 0"
        );
        let normalizer = max_ratio(&values, &self.prev, 1.0);
        let lmbda: Vec<f64> = values
            .iter()
            .zip(&self.prev)
            .map(|(l, p)| (l / (p + self.eps) - normalizer).exp())
            .collect();
        let scale = self.num_losses as f64 / (lmbda.iter().sum::<f64>() + self.eps);
        let loss = weighted_sum(losses, &lmbda).mul_scalar(scale);
        self.prev = values;
        (loss, lmbda)
    }
}

/// 2.0.0's `Relobralo` (ReLoBRaLo, Bischof and Kraus 2021).
///
/// Step 0 returns Σ Lᵢ and the current weights and records the losses as initial and
/// previous. Each later step draws ρ ~ Bernoulli(β) and, with Lᵢ the (weighted) losses,
/// Pᵢ the previous and Iᵢ the initial ones: λᵢ^P = exp(Lᵢ / (τ Pᵢ + eps) − maxⱼ Lⱼ / (τ Pⱼ)),
/// λ^I alike with I, each scaled to n / (Σ λ + eps); the weights become
/// α (ρ wᵢ + (1 − ρ) λᵢ^I) + (1 − α) λᵢ^P, and the aggregate is Σ wᵢ Lᵢ.
#[derive(Clone, Debug)]
pub struct Relobralo {
    /// The weights' EMA rate α, default 0.95.
    pub alpha: f64,
    /// The probability β of keeping the history (ρ = 1), default 0.99.
    pub beta: f64,
    /// The temperature τ, default 1.
    pub tau: f64,
    /// Guards the divisions, default 1e-8.
    pub eps: f64,
    /// Fixed per-loss weights applied first, as [`SoftAdapt::weights`].
    pub weights: Option<Vec<f64>>,
    num_losses: usize,
    init: Vec<f64>,
    prev: Vec<f64>,
    ema: Vec<f64>,
    rng: ChaCha8Rng,
}

impl Relobralo {
    /// ReLoBRaLo over `num_losses` losses at 2.0.0's defaults, its draws of ρ from
    /// `ChaCha8Rng::seed_from_u64(seed)`.
    ///
    /// # Panics
    /// If `weights` is given with a length other than `num_losses`.
    pub fn new(num_losses: usize, weights: Option<Vec<f64>>, seed: u64) -> Self {
        check_weights(&weights, num_losses);
        Self {
            alpha: 0.95,
            beta: 0.99,
            tau: 1.0,
            eps: 1e-8,
            weights,
            num_losses,
            init: Vec::new(),
            prev: Vec::new(),
            ema: vec![1.0; num_losses],
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// The aggregate of `losses` (`[1]` each) at `step`, and the weights; ρ drawn from the
    /// seeded generator at every step after 0.
    ///
    /// # Panics
    /// As [`forward_with_rho`](Self::forward_with_rho).
    pub fn forward(&mut self, losses: Vec<Tensor<1>>, step: usize) -> (Tensor<1>, Vec<f64>) {
        // A uniform draw in [0, 1) from 53 random bits: exact at β = 0 and β = 1.
        let rho = step > 0 && ((self.rng.next_u64() >> 11) as f64) * 2f64.powi(-53) < self.beta;
        self.forward_with_rho(losses, step, rho)
    }

    /// [`forward`](Self::forward) with the draw ρ given (ignored at step 0), to replay a
    /// sequence of draws.
    ///
    /// # Panics
    /// If the number of losses is not `num_losses`, or `step > 0` before any step 0.
    pub fn forward_with_rho(
        &mut self,
        losses: Vec<Tensor<1>>,
        step: usize,
        rho: bool,
    ) -> (Tensor<1>, Vec<f64>) {
        let losses = weigh(losses, &self.weights, self.num_losses);
        let values = scalars(&losses);
        if step == 0 {
            self.init = values.clone();
            self.prev = values;
            return (sum(losses), self.ema.clone());
        }
        assert!(
            !self.prev.is_empty(),
            "Relobralo: step {step} before step 0"
        );
        let balance = |reference: &[f64]| {
            let normalizer = max_ratio(&values, reference, self.tau);
            let lmbda: Vec<f64> = values
                .iter()
                .zip(reference)
                .map(|(l, r)| (l / (self.tau * r + self.eps) - normalizer).exp())
                .collect();
            let scale = self.num_losses as f64 / (lmbda.iter().sum::<f64>() + self.eps);
            lmbda.into_iter().map(|l| l * scale).collect::<Vec<f64>>()
        };
        let (lmbda_prev, lmbda_init) = (balance(&self.prev), balance(&self.init));
        let rho = if rho { 1.0 } else { 0.0 };
        for i in 0..self.num_losses {
            self.ema[i] = self.alpha * (rho * self.ema[i] + (1.0 - rho) * lmbda_init[i]);
            self.ema[i] += (1.0 - self.alpha) * lmbda_prev[i];
        }
        let loss = weighted_sum(losses, &self.ema);
        self.prev = values;
        (loss, self.ema.clone())
    }
}

/// # Panics
/// If `weights` is given with a length other than `n`.
fn check_weights(weights: &Option<Vec<f64>>, n: usize) {
    if let Some(w) = weights {
        assert_eq!(w.len(), n, "aggregator: {} weights for {n} losses", w.len());
    }
}

/// 2.0.0's `Aggregator.weigh_losses`: each loss times its weight.
///
/// # Panics
/// If the number of losses is not `n`.
fn weigh(losses: Vec<Tensor<1>>, weights: &Option<Vec<f64>>, n: usize) -> Vec<Tensor<1>> {
    assert_eq!(
        losses.len(),
        n,
        "aggregator: {} losses, expected {n}",
        losses.len()
    );
    match weights {
        None => losses,
        Some(w) => losses
            .into_iter()
            .zip(w)
            .map(|(l, w)| l.mul_scalar(*w))
            .collect(),
    }
}

/// The losses' values, `[1]` each, on the host.
fn scalars(losses: &[Tensor<1>]) -> Vec<f64> {
    losses
        .iter()
        .map(|l| l.clone().into_scalar::<f64>())
        .collect()
}

/// maxᵢ lᵢ / (τ rᵢ), unguarded as in 2.0.0.
fn max_ratio(values: &[f64], reference: &[f64], tau: f64) -> f64 {
    values
        .iter()
        .zip(reference)
        .map(|(l, r)| l / (tau * r))
        .fold(f64::NEG_INFINITY, f64::max)
}

fn sum(losses: Vec<Tensor<1>>) -> Tensor<1> {
    losses
        .into_iter()
        .reduce(|a, b| a + b)
        .expect("aggregator: no losses")
}

/// Σᵢ cᵢ Lᵢ with constant coefficients.
fn weighted_sum(losses: Vec<Tensor<1>>, coefficients: &[f64]) -> Tensor<1> {
    sum(losses
        .into_iter()
        .zip(coefficients)
        .map(|(l, c)| l.mul_scalar(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::*;

    fn losses(values: &[f64]) -> Vec<Tensor<1>> {
        values
            .iter()
            .map(|&v| Tensor::from_data([v], &Device::default()))
            .collect()
    }

    /// β = 1 always keeps the history (ρ = 1), β = 0 never; at β = 0.5 a seed replays the
    /// same weights.
    #[test]
    fn seeded_draws_are_exact_at_the_ends_and_reproducible() {
        let run = |beta: f64, seed: u64| {
            let mut r = Relobralo::new(2, None, seed);
            r.beta = beta;
            (0..8)
                .map(|s| {
                    r.forward(losses(&[1.0 + s as f64, 2.0 / (1.0 + s as f64)]), s)
                        .1
                })
                .collect::<Vec<_>>()
        };
        let replay = |beta: f64, rho: bool| {
            let mut r = Relobralo::new(2, None, 99);
            r.beta = beta;
            (0..8)
                .map(|s| {
                    let l = losses(&[1.0 + s as f64, 2.0 / (1.0 + s as f64)]);
                    r.forward_with_rho(l, s, rho).1
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(1.0, 7), replay(1.0, true));
        assert_eq!(run(0.0, 7), replay(0.0, false));
        assert_eq!(run(0.5, 7), run(0.5, 7));
        assert_ne!(run(0.5, 7), run(0.5, 8));
    }

    /// SoftAdapt by hand at step 1: losses (2, 1) after (1, 1), eps = 0: normaliser 2,
    /// λ = (1, e⁻¹), aggregate 2 (2 + e⁻¹) / (1 + e⁻¹).
    #[test]
    fn softadapt_known_step() {
        let mut s = SoftAdapt::new(2, None);
        s.eps = 0.0;
        s.forward(losses(&[1.0, 1.0]), 0);
        let (loss, lmbda) = s.forward(losses(&[2.0, 1.0]), 1);
        let e = (-1.0f64).exp();
        assert!(
            (lmbda[0] - 1.0).abs() < 1e-12 && (lmbda[1] - e).abs() < 1e-12,
            "{lmbda:?}"
        );
        let expected = 2.0 * (2.0 + e) / (1.0 + e);
        let v = loss.into_scalar::<f64>();
        assert!((v - expected).abs() < 1e-5, "{v} vs {expected}");
    }

    #[test]
    #[should_panic(expected = "before step 0")]
    fn softadapt_step_before_step_0_panics() {
        SoftAdapt::new(2, None).forward(losses(&[1.0, 1.0]), 1);
    }

    #[test]
    #[should_panic(expected = "before step 0")]
    fn relobralo_step_before_step_0_panics() {
        Relobralo::new(2, None, 0).forward(losses(&[1.0, 1.0]), 1);
    }

    #[test]
    #[should_panic(expected = "3 losses, expected 2")]
    fn wrong_number_of_losses_panics() {
        SoftAdapt::new(2, None).forward(losses(&[1.0, 1.0, 1.0]), 0);
    }

    #[test]
    #[should_panic(expected = "1 weights for 2 losses")]
    fn mismatched_weights_panic() {
        Relobralo::new(2, Some(vec![1.0]), 0);
    }

    #[test]
    #[should_panic(expected = "each loss must have a weight")]
    fn weighted_sum_mismatched_weights_panic() {
        let l: Vec<LossFn<2>> = vec![Box::new(
            crate::neural_operators::losses::pointwise::mse_loss,
        )];
        WeightedSumLoss::new(l, Some(vec![0.5, 0.5]));
    }
}
