//! L-BFGS training (optimisers CO2.1; proposal §4): [`train_epoch_lbfgs`], one Burn
//! `LBFGS::step` per epoch on the full batch, its closure Gauss–Newton's objective Φ
//! ([`GridResidual`], D3) plus ½μ‖θ‖² (D13), the counterpart of
//! [`train_epoch_gn`](super::gn::train_epoch_gn).

use std::sync::Arc;
use std::time::Instant;

use burn::{
    data::dataloader::DataLoader,
    module::{ModuleVisitor, Param},
    optim::{GradientsParams, LBFGS},
    prelude::*,
};

use crate::neural_operators::{
    data::{batcher::Batch, transforms::normalizers::FlatDecoder},
    models::operator::GridOperator,
    optim::gauss_newton::LeastSquares,
    runs::LbfgsStepRecord,
    training::{
        config::LbfgsArm,
        gn::GridResidual,
        step::{StepLoss, forward_loss},
        trainer::EpochMeans,
    },
};

/// ½μ‖θ‖² over every float parameter, in the autodiff graph, summed in [`Module::visit`]
/// order.
struct HalfSquaredNorm {
    sum: Option<Tensor<1>>,
}

impl ModuleVisitor for HalfSquaredNorm {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        let sq = param.val().powf_scalar(2.0).sum();
        self.sum = Some(match self.sum.take() {
            Some(s) => s + sq,
            None => sq,
        });
    }
}

/// The closure's objective at `model` on `batch`: Φ = ½‖r‖² of `residual`, plus ½μ‖θ‖²
/// when `mu` > 0 (at μ = 0 exactly Φ, with no extra operations). A scalar, `[1]`.
fn objective<const R: usize, const RM1: usize, M: GridOperator<R>>(
    model: &M,
    residual: &GridResidual<'_, R, RM1>,
    batch: &Batch<R, RM1>,
    mu: f64,
) -> Tensor<1> {
    let phi = residual.residual(model, batch).powf_scalar(2.0).sum() * 0.5;
    if mu > 0.0 {
        let mut norm = HalfSquaredNorm { sum: None };
        model.visit(&mut norm);
        let sq = norm.sum.expect("the model has float parameters");
        return phi + sq * (0.5 * mu);
    }
    phi
}

/// One epoch of L-BFGS: one `LBFGS::step` (up to `max_iter` inner iterations) on the one
/// batch `loader` yields, the full training set (D4: the curvature pairs assume the same
/// function at every evaluation), with the closure Φ + ½μ‖θ‖² at μ = `arm.weight_decay`
/// and `arm.learning_rate`. Its row goes to `on_step`; the closure evaluations are its
/// passes, one forward and one backward each (D7).
///
/// The returned [`EpochMeans`] are those [`train_epoch`](super::trainer::train_epoch)
/// reports, at the epoch's incoming parameters, as
/// [`train_epoch_gn`](super::gn::train_epoch_gn) measures them (one extra forward without
/// autodiff); `last_lr` is `arm.learning_rate`.
///
/// `model`, `loader` and `decoder` are on the autodiff device.
///
/// # Panics
/// If the loader yields no batch, a failed one, or more than one (the batch size must
/// cover the training set); as `LBFGS::step`.
pub fn train_epoch_lbfgs<const R: usize, const RM1: usize, M: GridOperator<R>>(
    model: M,
    loader: &Arc<dyn DataLoader<Batch<R, RM1>>>,
    lbfgs: &mut LBFGS,
    arm: &LbfgsArm,
    decoder: Option<&FlatDecoder>,
    epoch: usize,
    on_step: &mut dyn FnMut(&LbfgsStepRecord),
) -> (M, EpochMeans) {
    let mut batches = loader.iter();
    let batch = batches
        .next()
        .expect("training loader produced no batches")
        .expect("dataset error during training");
    assert!(
        batches.next().is_none(),
        "L-BFGS trains on the full batch (D4): the batch size must cover the training set"
    );

    let plain = Batch {
        inputs: batch.inputs.clone().inner(),
        targets: batch.targets.clone().inner(),
    };
    let StepLoss {
        l2_sum,
        out,
        target,
    } = forward_loss(
        &model.valid(),
        plain,
        decoder.map(FlatDecoder::inner).as_ref(),
    );
    let train_mse = (out - target).powf_scalar(2.0).mean().into_scalar::<f32>();
    let train_l2 = l2_sum.into_scalar::<f32>() / loader.num_items() as f32;

    let residual = GridResidual::<R, RM1>::new(decoder);
    let (mut closure_evals, mut phi_before) = (0, None);
    let start = Instant::now();
    let (model, phi_after) = lbfgs.step(arm.learning_rate, model, |m: M| {
        let phi = objective(&m, &residual, &batch, arm.weight_decay);
        let grads = GradientsParams::from_grads(phi.backward(), &m);
        let value = phi.into_scalar::<f64>();
        closure_evals += 1;
        phi_before.get_or_insert(value);
        (value, grads)
    });
    on_step(&LbfgsStepRecord {
        epoch,
        phi_before: phi_before.expect("LBFGS::step evaluates the closure"),
        phi_after,
        closure_evals,
        wall_time_s: start.elapsed().as_secs_f64(),
    });

    (
        model,
        EpochMeans {
            train_mse,
            train_l2,
            last_lr: arm.learning_rate,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neural_operators::{
        data::device_batcher::device_loader,
        optim::gauss_newton::params,
        training::fixtures::{tiny_dataset, tiny_model_cfg},
    };
    use burn::optim::{LBFGSConfig, LineSearchFn};

    const N: usize = 6;

    /// `epochs` epochs of `arm` on the tiny Burgers fixture, full batch; the step rows.
    fn run(arm: &LbfgsArm, epochs: usize, batch_size: usize) -> Vec<LbfgsStepRecord> {
        let device = Device::default().autodiff();
        device.seed(7);
        let mut model = tiny_model_cfg().init::<3>(&device);
        let loader = device_loader::<3, 2, _>(&tiny_dataset(N), &device, batch_size, Some(7));
        let mut lbfgs = arm.lbfgs.init();
        let mut rows = Vec::new();
        for epoch in 0..epochs {
            let (m, means) =
                train_epoch_lbfgs(model, &loader, &mut lbfgs, arm, None, epoch, &mut |r| {
                    rows.push(r.clone())
                });
            assert!(means.train_l2.is_finite() && means.train_mse.is_finite());
            assert_eq!(means.last_lr, arm.learning_rate);
            model = m;
        }
        rows
    }

    /// Proposal §9: on a small least-squares problem it converges. Strong Wolfe makes
    /// every step decrease Φ, and five steps take it far below its start.
    #[test]
    fn strong_wolfe_converges_on_a_small_problem() {
        let arm = LbfgsArm::new(
            LBFGSConfig::new()
                .with_history_size(10)
                .with_line_search_fn(LineSearchFn::StrongWolfe),
        );
        let rows = run(&arm, 5, N);
        for r in &rows {
            assert!(r.phi_after < r.phi_before, "{r:?}");
            assert!(r.closure_evals >= 2, "{r:?}");
        }
        // the next step starts where the last ended
        for w in rows.windows(2) {
            let rel = (w[1].phi_before - w[0].phi_after).abs() / w[0].phi_after;
            assert!(rel < 1e-5, "{w:?}");
        }
        let (first, last) = (rows[0].phi_before, rows[4].phi_after);
        assert!(last < 0.1 * first, "Φ {first} -> {last}");
    }

    /// Proposal §9: the passes add up. Without a line search each inner iteration
    /// evaluates the closure once, after the step's first evaluation: `max_iter` + 1 per
    /// step, one forward and one backward each, unless `max_eval` (Burn's default
    /// ⌊1.25 `max_iter`⌋) stops the step first.
    #[test]
    fn without_line_search_each_step_costs_max_iter_plus_one_evaluations() {
        let lbfgs = LBFGSConfig::new().with_max_iter(3);
        let arm = LbfgsArm::new(lbfgs.clone().with_max_eval(Some(10))).with_learning_rate(1e-3);
        let rows = run(&arm, 2, N);
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().all(|r| r.closure_evals == 4), "{rows:?}");
        let capped = run(&LbfgsArm::new(lbfgs).with_learning_rate(1e-3), 1, N);
        assert_eq!(capped[0].closure_evals, 3, "max_eval = 3 * 5 / 4");
    }

    /// D13: with μ > 0 the closure's objective is Φ + ½μ‖θ‖², ‖θ‖ over every float
    /// parameter, to f32 round-off; at μ = 0 it is Φ.
    #[test]
    fn weight_decay_adds_half_mu_squared_norm() {
        let device = Device::default().autodiff();
        let model = tiny_model_cfg().init::<3>(&device);
        let batch = device_loader::<3, 2, _>(&tiny_dataset(N), &device, N, None)
            .iter()
            .next()
            .unwrap()
            .unwrap();
        let residual = GridResidual::<3, 2>::new(None);
        let at = |mu| objective(&model, &residual, &batch, mu).into_scalar::<f64>();
        let phi = 0.5
            * residual
                .residual(&model, &batch)
                .powf_scalar(2.0)
                .sum()
                .into_scalar::<f64>();
        assert_eq!(at(0.0), phi);
        let norm = params(&model).norm();
        let expected = phi + 0.5 * 0.1 * norm * norm;
        assert!(
            (at(0.1) - expected).abs() <= 1e-5 * expected,
            "{} != {expected}",
            at(0.1)
        );
    }

    /// D4: mini-batches are refused.
    #[test]
    #[should_panic(expected = "the batch size must cover the training set")]
    fn more_than_one_batch_panics() {
        let arm = LbfgsArm::new(LBFGSConfig::new());
        run(&arm, 1, N - 1);
    }
}
