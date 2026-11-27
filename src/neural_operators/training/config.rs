//! Optimizer and schedule settings for a training run.

use burn::{config::Config, optim::LBFGSConfig};

use crate::neural_operators::optim::{gauss_newton::GaussNewtonConfig, soap::SoapConfig};

/// Which optimiser trains the model (Phase 10 T3).
#[derive(Config, Debug)]
pub enum OptimizerConfig {
    /// Adam with coupled weight decay and the run's schedule (the cosine unless
    /// [`TrainingConfig::lr_schedule`] says otherwise), from [`TrainingConfig`]'s
    /// learning-rate and weight-decay fields: the baseline.
    Adam,
    /// Damped Gauss–Newton ([`GaussNewton`](crate::neural_operators::optim::gauss_newton::GaussNewton)),
    /// one step per batch; [`TrainingConfig`]'s learning-rate fields are unused, and the
    /// weight decay is the config's own μ.
    GaussNewton(GaussNewtonConfig),
    /// Adam with decoupled weight decay (Burn's `AdamWConfig`, ε 1e-8, the decay scaled by
    /// the learning rate), from [`TrainingConfig`]'s learning-rate and weight-decay fields:
    /// PETScML's reference solver (optimisers CO1.1, D11).
    AdamW,
    /// Burn's L-BFGS on Gauss–Newton's objective Φ, one `step` per epoch on the full batch
    /// (optimisers CO2.1; D3, D4): [`TrainingConfig::batch_size`] must cover the training
    /// set, and [`TrainingConfig`]'s learning-rate and weight-decay fields are unused.
    Lbfgs(LbfgsArm),
    /// SOAP ([`Soap`](crate::neural_operators::optim::soap::Soap)) in place of Adam, on
    /// Adam's path: [`TrainingConfig`]'s learning rate, schedule and batch size drive it;
    /// its weight decay is the [`SoapConfig`]'s own (decoupled), and
    /// [`TrainingConfig::weight_decay`] is unused (optimisers CO3.3).
    Soap(SoapConfig),
}

/// The L-BFGS arm (optimisers CO2.1): Burn's settings and the objective's.
#[derive(Config, Debug)]
pub struct LbfgsArm {
    /// Burn's L-BFGS settings: history, inner iterations, tolerances, line search.
    pub lbfgs: LBFGSConfig,
    /// The learning rate `LBFGS::step` takes: the step a line search starts from, or the
    /// fixed step without one (D5: 1).
    #[config(default = 1.0)]
    pub learning_rate: f64,
    /// μ in the closure's Φ + ½μ‖θ‖² (D3, D13: 0 or 1e-4); 0 adds nothing.
    #[config(default = 0.0)]
    pub weight_decay: f64,
}

/// A learning-rate schedule other than the default cosine (optimisers CO1.1).
#[derive(Config, Debug)]
pub enum LrSchedule {
    /// [`TrainingConfig::learning_rate`] halved every `every_epochs` epochs (Burn's
    /// `StepLrSchedulerConfig`, stepped once per batch): PETScML's reference (D11).
    Step {
        /// Epochs between two halvings; at least 1, or building the run's components panics.
        every_epochs: usize,
    },
}

/// The objective the Adam path back-propagates (optimisers CO1.1). The reported train and
/// test L² are the unsquared relative L² either way.
#[derive(Config, Debug, PartialEq)]
pub enum TrainLoss {
    /// The relative L² summed over the batch (CONVENTIONS §11; Li et al.): the default.
    RelL2,
    /// The mean over the batch of the squared relative L², PETScML's eq. 6 (D10).
    SquaredRelL2,
}

/// Optimizer, schedule and batching settings, saved with every run
/// ([`RunDir::save_configs`](crate::neural_operators::runs::RunDir::save_configs)).
#[derive(Config, Debug)]
pub struct TrainingConfig {
    /// Number of training epochs.
    #[config(default = 100)]
    pub epochs: usize,
    /// Training batch size; the last batch of an epoch may be partial.
    #[config(default = 20)]
    pub batch_size: usize,
    /// Evaluation batch size; it does not change the test loss, which is a per-sample mean.
    #[config(default = 20)]
    pub test_batch_size: usize,
    /// Initial learning rate: the cosine's peak, or the step schedule's starting rate.
    #[config(default = 1e-3)]
    pub learning_rate: f64,
    /// f32 to match `WeightDecayConfig::penalty`, unlike the f64 types above.
    #[config(default = 1e-4)]
    pub weight_decay: f32,
    /// Final learning rate of the cosine schedule; unused under [`LrSchedule::Step`].
    #[config(default = 1e-7)]
    pub min_lr: f64,
    /// Seeds the device RNG (weight initialisation) and the training loader's shuffle.
    #[config(default = 42)]
    pub seed: u64,
    /// The optimiser; `None` (the default) means [`OptimizerConfig::Adam`]. An `Option`
    /// so that saved configs without this field (before Phase 10 T3) still load.
    pub optimizer: Option<OptimizerConfig>,
    /// A wall-clock budget in seconds (Phase 10 T4, G4's equal wall-clock): training stops
    /// after the first epoch that ends with the run's training time (earlier sessions'
    /// included) at or over it, so it overshoots by at most one epoch; `epochs` stays the
    /// cap. `None` (the default, and every config before Phase 10 T4) trains all `epochs`.
    /// Adam's cosine schedule still spans `epochs`, so a budget cuts it short.
    pub time_budget_s: Option<f64>,
    /// With Gauss–Newton, the first `adam_epochs` epochs train with Adam instead, its cosine
    /// schedule spanning those epochs' steps, and Gauss–Newton trains the rest: the Hybrid
    /// arm (Phase 10 T5). `None` (the default, and every config before Phase 10 T5) trains
    /// every epoch with the run's optimiser.
    pub adam_epochs: Option<usize>,
    /// The learning-rate schedule; `None` (the default, and every config before optimisers
    /// CO1.1) is the cosine from `learning_rate` to `min_lr` over
    /// [`schedule_steps`](Self::schedule_steps).
    pub lr_schedule: Option<LrSchedule>,
    /// The Adam path's training objective; `None` (the default, and every config before
    /// optimisers CO1.1) is [`TrainLoss::RelL2`]. Gauss–Newton's Φ is unchanged by it.
    pub loss: Option<TrainLoss>,
}

impl TrainingConfig {
    /// Optimisation steps a run of all `epochs` takes on `n_train` samples: `epochs` ×
    /// ⌈`n_train` / `batch_size`⌉, since an epoch's last batch may be partial. The trainer's cosine
    /// schedule spans exactly these steps, but in a hybrid run ([`schedule_steps`](Self::schedule_steps)).
    pub fn total_steps(&self, n_train: usize) -> usize {
        self.epochs * self.steps_per_epoch(n_train)
    }

    /// Steps of one epoch on `n_train` samples: ⌈`n_train` / `batch_size`⌉, the last batch
    /// possibly partial.
    pub fn steps_per_epoch(&self, n_train: usize) -> usize {
        n_train.div_ceil(self.batch_size)
    }

    /// Steps the Adam cosine schedule spans on `n_train` samples: [`total_steps`](Self::total_steps),
    /// or a hybrid run's `adam_epochs` × ⌈`n_train` / `batch_size`⌉.
    pub fn schedule_steps(&self, n_train: usize) -> usize {
        self.adam_epochs.unwrap_or(self.epochs) * self.steps_per_epoch(n_train)
    }

    /// Whether epoch `epoch` (from 0) of a Gauss–Newton run trains with Adam: a hybrid run's
    /// first [`adam_epochs`](Self::adam_epochs).
    pub fn adam_epoch(&self, epoch: usize) -> bool {
        self.adam_epochs.is_some_and(|a| epoch < a)
    }

    /// Whether `trained_s` seconds of training have spent [`time_budget_s`](Self::time_budget_s);
    /// always `false` without a budget.
    pub fn budget_spent(&self, trained_s: f64) -> bool {
        self.time_budget_s.is_some_and(|b| trained_s >= b)
    }

    /// The Gauss–Newton settings if this run trains with Gauss–Newton, else `None` (Adam).
    pub fn gauss_newton(&self) -> Option<&GaussNewtonConfig> {
        match &self.optimizer {
            Some(OptimizerConfig::GaussNewton(gn)) => Some(gn),
            Some(
                OptimizerConfig::Adam
                | OptimizerConfig::AdamW
                | OptimizerConfig::Lbfgs(_)
                | OptimizerConfig::Soap(_),
            )
            | None => None,
        }
    }

    /// The L-BFGS arm if this run trains with L-BFGS, else `None`.
    pub fn lbfgs(&self) -> Option<&LbfgsArm> {
        match &self.optimizer {
            Some(OptimizerConfig::Lbfgs(arm)) => Some(arm),
            _ => None,
        }
    }

    /// The SOAP settings if this run trains with SOAP, else `None`.
    pub fn soap(&self) -> Option<&SoapConfig> {
        match &self.optimizer {
            Some(OptimizerConfig::Soap(soap)) => Some(soap),
            _ => None,
        }
    }

    /// Whether the Adam path back-propagates [`TrainLoss::SquaredRelL2`].
    pub fn squared_loss(&self) -> bool {
        self.loss == Some(TrainLoss::SquaredRelL2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_steps_counts_the_partial_last_batch() {
        let cfg = TrainingConfig::new().with_epochs(500).with_batch_size(20);
        assert_eq!(cfg.total_steps(25), 1000);
        assert_eq!(cfg.total_steps(1000), 25_000);
        assert_eq!(cfg.total_steps(20), 500);
        assert_eq!(cfg.schedule_steps(1000), 25_000);
        let hybrid = cfg.with_adam_epochs(Some(250));
        assert_eq!(hybrid.schedule_steps(1000), 12_500);
        assert_eq!(hybrid.total_steps(1000), 25_000);
        assert!(hybrid.adam_epoch(249) && !hybrid.adam_epoch(250));
        assert!(!TrainingConfig::new().adam_epoch(0));
    }

    /// A `train_cfg.json` as runs wrote it before Phase 10 T3 loads as Adam; a
    /// Gauss–Newton config round-trips.
    #[test]
    fn old_config_loads_as_adam_and_gauss_newton_round_trips() {
        let old = r#"{"epochs": 500, "batch_size": 20, "test_batch_size": 20,
            "learning_rate": 0.001, "weight_decay": 0.0001, "min_lr": 1e-5, "seed": 42}"#;
        let cfg = TrainingConfig::load_binary(old.as_bytes()).unwrap();
        assert!(cfg.optimizer.is_none() && cfg.gauss_newton().is_none());
        assert_eq!(cfg.epochs, 500);

        let gn = GaussNewtonConfig::new().with_cg_max_iters(7);
        let cfg = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::GaussNewton(gn)));
        let back: TrainingConfig =
            TrainingConfig::load_binary(burn::config::config_to_json(&cfg).as_bytes()).unwrap();
        assert_eq!(back.gauss_newton().map(|g| g.cg_max_iters), Some(7));
        let adam = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::Adam));
        assert!(adam.gauss_newton().is_none());
    }

    /// Optimisers CO1.1: an old config has the cosine, the summed loss and no AdamW; the
    /// new fields round-trip.
    #[test]
    fn co1_1_fields_default_to_today_and_round_trip() {
        let old = r#"{"epochs": 500, "batch_size": 20, "test_batch_size": 20,
            "learning_rate": 0.001, "weight_decay": 0.0001, "min_lr": 1e-5, "seed": 42}"#;
        let cfg = TrainingConfig::load_binary(old.as_bytes()).unwrap();
        assert!(cfg.lr_schedule.is_none() && cfg.loss.is_none() && !cfg.squared_loss());

        let cfg = TrainingConfig::new()
            .with_optimizer(Some(OptimizerConfig::AdamW))
            .with_lr_schedule(Some(LrSchedule::Step { every_epochs: 100 }))
            .with_loss(Some(TrainLoss::SquaredRelL2));
        let back: TrainingConfig =
            TrainingConfig::load_binary(burn::config::config_to_json(&cfg).as_bytes()).unwrap();
        assert!(matches!(back.optimizer, Some(OptimizerConfig::AdamW)));
        assert!(matches!(
            back.lr_schedule,
            Some(LrSchedule::Step { every_epochs: 100 })
        ));
        assert!(back.squared_loss() && back.gauss_newton().is_none());
    }

    /// Optimisers CO2.1: an L-BFGS arm round-trips, and is neither Gauss–Newton nor Adam.
    #[test]
    fn lbfgs_arm_round_trips() {
        use burn::optim::LineSearchFn;
        let arm = LbfgsArm::new(
            LBFGSConfig::new()
                .with_history_size(10)
                .with_line_search_fn(LineSearchFn::StrongWolfe),
        )
        .with_weight_decay(1e-4);
        let cfg = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::Lbfgs(arm)));
        let back: TrainingConfig =
            TrainingConfig::load_binary(burn::config::config_to_json(&cfg).as_bytes()).unwrap();
        let arm = back.lbfgs().expect("an L-BFGS arm");
        assert_eq!(arm.lbfgs.history_size, 10);
        assert_eq!(arm.lbfgs.line_search_fn, LineSearchFn::StrongWolfe);
        assert_eq!((arm.learning_rate, arm.weight_decay), (1.0, 1e-4));
        assert!(back.gauss_newton().is_none() && TrainingConfig::new().lbfgs().is_none());
    }

    /// Optimisers CO3.3: a SOAP arm round-trips, and is no other arm.
    #[test]
    fn soap_arm_round_trips() {
        let soap = SoapConfig::new()
            .with_precondition_frequency(50)
            .with_weight_decay(0.0);
        let cfg = TrainingConfig::new().with_optimizer(Some(OptimizerConfig::Soap(soap)));
        let back: TrainingConfig =
            TrainingConfig::load_binary(burn::config::config_to_json(&cfg).as_bytes()).unwrap();
        let soap = back.soap().expect("a SOAP arm");
        assert_eq!((soap.precondition_frequency, soap.weight_decay), (50, 0.0));
        assert!(!soap.log_first_refresh);
        assert!(back.gauss_newton().is_none() && back.lbfgs().is_none());
        assert!(TrainingConfig::new().soap().is_none());
    }
}
