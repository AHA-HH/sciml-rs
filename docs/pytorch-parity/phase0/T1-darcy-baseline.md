# pytorch-parity phase 0 / T1 - PyTorch Darcy baseline (C0.1)

Can start at once. **This task needs the user.** Its numbers come from running
neuraloperator in Python. An agent may prepare and run the script only if the user's
machine has a suitable Python environment and the user agrees. Otherwise the agent
writes the snippet below and stops, and the user runs it.

Read first:
- `docs/design/pytorch-parity.md` §1.5 (A1), §8.3 and §9.2 (Q2);
- [`examples/models/plot_FNO_darcy.py`](https://github.com/neuraloperator/neuraloperator/blob/2.0.0/examples/models/plot_FNO_darcy.py)
  at tag `2.0.0`, lines 52-162;
- `neuralop/training/trainer.py` at `2.0.0`: `train`, lines 229-262, and `evaluate_all`,
  line 336.

This task delivers the PyTorch reference for the benchmark: neuraloperator 2.0.0's Darcy
example run with five seeds. Phase 3's `examples/train/darcy_neuralop.rs` must match it
within the design §8.3 criterion. It is also the user's guided tour of the
neuraloperator pipeline (dataset, `DataProcessor`, `FNO`, `Trainer`, losses).

Do:
- **Set up the environment, outside the repository.** For example, a virtualenv in
  `~/Code/neuralop-ref/`: `pip install neuraloperator==2.0.0 torch matplotlib`, CPU
  wheels. Record `torch.__version__`, `neuralop.__version__`, `tensorly.__version__`,
  `tltorch.__version__`, the Python version and the machine.
- **Run this script there, not in the repository.** It reproduces `plot_FNO_darcy.py`'s
  training with a seed, and adds a final evaluation. In the example, `eval_interval=3`
  means epoch 19 is never evaluated (`trainer.py:243`).

  ```python
  import sys, torch, numpy as np
  from neuralop.models import FNO
  from neuralop import Trainer, LpLoss, H1Loss
  from neuralop.training import AdamW
  from neuralop.data.datasets import load_darcy_flow_small
  from neuralop.utils import count_model_params

  def run(seed):
      torch.manual_seed(seed); np.random.seed(seed)
      train_loader, test_loaders, dp = load_darcy_flow_small(
          n_train=1000, batch_size=32, test_resolutions=[16, 32],
          n_tests=[100, 50], test_batch_sizes=[32, 32])
      model = FNO(n_modes=(8, 8), in_channels=1, out_channels=1,
                  hidden_channels=32, projection_channel_ratio=2)
      opt = AdamW(model.parameters(), lr=8e-3, weight_decay=1e-4)
      sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
      h1, l2 = H1Loss(d=2), LpLoss(d=2, p=2)
      losses = {"h1": h1, "l2": l2}
      tr = Trainer(model=model, n_epochs=20, device="cpu", data_processor=dp,
                   wandb_log=False, eval_interval=3, use_distributed=False, verbose=False)
      tr.train(train_loader=train_loader, test_loaders=test_loaders, optimizer=opt,
               scheduler=sched, regularizer=False, training_loss=h1, eval_losses=losses)
      final = tr.evaluate_all(epoch=19, eval_losses=losses,
                              test_loaders=test_loaders, eval_modes={})
      return count_model_params(model), final

  for seed in range(5):
      n, m = run(seed)
      print(seed, n, {k: float(v) for k, v in m.items()})
  ```

  Before running, check the `evaluate_all` signature and its metric key names
  (`16_h1`, `16_l2`, `32_h1`, `32_l2`) against `trainer.py` at 2.0.0. If they differ,
  adapt the script and say so in the PR.
- **`docs/design/pytorch-parity.md` §8.3.** After the "Setup" paragraph, add a
  "PyTorch baseline (C0.1, <date>)" table with:
  - a row per seed with the four metrics, and a row with the mean and the sample std
    (ddof = 1);
  - the parameter count;
  - the versions and hardware recorded above;
  - the script, verbatim, in a collapsed `<details>` block, so the run can be repeated
    without a file in the repo.
- **Design §7, Phase 0.** Mark C0.1 done, with the date.

Notes for the record in §8.3:
- **Seeds affect only the model initialisation.** The train loader is built without
  `shuffle`, so it defaults to False (`neuralop/data/datasets/darcy.py:171-177` at
  2.0.0). So `torch.manual_seed` changes only the model initialisation, and
  `np.random.seed` has no effect.
- **The five-seed std is therefore initialisation variance only.** Say so in §8.3. Phase
  3's Rust run must also train without shuffling, or the pooled-σ criterion compares
  different sources of noise.
- **The dataset is downloaded on first use** (`download=True`, `darcy.py:167`), so the
  first run needs network access.

Tests that define done (error measure: `baseline`, see the phase README):
- Five seeds were run, and each printed all four metrics.
- **The parameter count is the same for all five seeds.** It is then the reference for
  the Phase 1 exit gate.
- **Sanity check.** Each metric's mean is finite and in (0, 1). If a mean is ≥ 1, stop
  and report: the run diverged.

Must pass:
- No Rust code changes, so `cargo fmt -- --check` and `cargo doc --no-deps` are enough
  to show nothing else was touched.
- `git diff --stat main` shows only `docs/design/pytorch-parity.md`.

Do not:
- Commit any Python file, virtualenv, dataset or output to the repository.
- Change any other section of the design document.
- Change the example's hyperparameters. If something has to change for the script to
  run, record exactly what in the PR and in §8.3.
