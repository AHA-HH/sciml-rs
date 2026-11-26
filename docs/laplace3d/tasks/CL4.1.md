# CL4.1 3D training and prediction examples

Started 2026-11-26 on `laplace3d/cl4-1`. Roadmap: "`train_poisson3d` and the 3D predict
example (`train_poisson` unchanged). *Done when:* a short run at n = 33 on the Mac writes
a run directory that the 3D predict example evaluates. Near the three-module limit: split
when it starts if it exceeds it. *Needs:* CL0.2, CL1.3, CL3.3, D4. The 3D `evaluate` and
errors record are new 3D types (proposal §6). Also one per-sample loop in
`metrics/poisson` for 2D and 3D, keeping a frozen copy of today's `poisson_errors` as the
D2 oracle, and passing max |u| to `boundary_errors_nd` (CL1.3 review F1, N2); it needs no
3D types (the generic loop already gives the 2D bits, L1 gate Q2), and first gains a D2
case with a prediction exactly zero on ∂Ω (a −0.0 sum)." Proposal: §1.3, §1.4, §2, §3.5,
§6, §9, §10.

## Goal
Train an `FNO<5>` on the 3D Poisson splits and evaluate the run with errors 2–5 on the
CGL grid, with one per-sample error loop for 2D and 3D.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- "A short run at n = 33 on the Mac writes a run directory that the 3D predict example
  evaluates."
- Proposal §6: "a 3D training example `train_poisson3d` (`FNO<5>`; not a rank switch in
  `train_poisson`, which stays as it is), 3D predict and evaluation examples"; "New fields
  go on new 3D types, not on `PoissonConfig` or `PoissonErrorsRecord`, so 2D run files
  keep their JSON."
- Roadmap: one per-sample loop for 2D and 3D; a frozen copy of today's `poisson_errors` as
  the D2 oracle; max |u| passed to `boundary_errors_nd`; first, a D2 case with a
  prediction exactly zero on ∂Ω.
- Constraint 1 (§1.4, D2): every 2D output bit for bit identical.

## Read first
- `src/neural_operators/metrics/poisson/{mod.rs,tests.rs}` (`poisson_errors`,
  `poisson_errors_nd`, `boundary_errors{,_nd}`, `evaluate`, `PoissonErrorsRecord`).
- `examples/train/poisson.rs`, `examples/predict/poisson.rs` (the 2D examples mirrored).
- `data/loaders/poisson/{uniform3d.rs,cheb.rs}` (CL3.3), `data/catalogue.rs`
  (`Poisson3dSet`, `GENERATED_3D`).
- `training/trainer.rs` (`build_training_components::<R, RM1, _>`, `train_components`),
  `runs` (`load_fno::<R>`, `checkpoint::<R>`): rank-generic since CM0.3, used as they are.
- `docs/laplace3d/gates/L1.md` (Q2), `docs/laplace3d/results/cl0-1.md` (Mac epoch times),
  `docs/laplace3d/results/cl2-1.md` (A3's floor, the 3D error 1 quote).

## Changes
1. **`metrics/poisson` (one loop).** First commit, a test only: the D2 oracle
   `poisson_errors_frozen` (today's `poisson_errors` and `boundary_errors`, copied into
   `tests.rs`) and new D2 cases: a prediction exactly zero on ∂Ω (interior nonzero) and a
   prediction ≡ 0, through `poisson_errors` and `poisson_errors_nd`, bit for bit against
   the oracle. Then: `poisson_errors` and `poisson_errors_nd` keep their checks and panic
   messages and call one private per-sample loop (today's `_nd` loop); the rank-3
   shortcut and the 2D `boundary_errors` go; `boundary_errors_nd` takes max |u| in place
   of u. `nd_boundary_at_rank_two_matches_2d` and `nd_rank_two_is_bit_identical_to_2d`
   compare against the frozen copy instead of the removed functions.
2. **`metrics/poisson` (3D evaluate).** `evaluate`'s body becomes a private
   `evaluate_core` taking the error 1 quote; `evaluate` calls it with `ERROR1_QUOTE` and
   sets `PoissonEvaluation::plot_run_dir` (which replaces `run_dir`; 2D record and report
   unchanged), a new `evaluate_3d` (no run directory) with `ERROR1_QUOTE_3D` (A3's floor,
   CL2.1) and no plot hint (no 3D plot example). The record type: see Open questions.
3. **`examples/train/poisson3d.rs`** (`train_poisson3d`, registered in `Cargo.toml`):
   `FNO<5>`, D4's width 32, 4 layers, modes min(12, s/2)³, batch 20, Adam with 2D's
   literals (lr 1e-3, weight decay 1e-4, cosine to 1e-5, 500 epochs, padding 4).
   Arguments `[<padding>] [<n>] [<s>] [<seed>]`, n one of `GENERATED_3D` (default 65,
   s = n − 1); `EPOCHS` overrides the epochs for a short run. Data through
   `load_poisson3d_uniform`, `PoissonConfig` and its `DatasetSource` as in 2D (CL3.3), so
   `data_cfg.json` is the 2D type. Checkpoints every 25 epochs and `--resume <run_dir>`
   (CL4.2's VM runs are 1.6–8.9 h each). Writes `costs.json`. Run name
   `poisson3d_fno_n{n}_s{s}_p{p}[_seed{seed}]`. Not taken (proposal §1.3 non-goals, or
   not needed): Gauss–Newton, `TRAIN_DEVICES`, subsets, MLflow.
4. **`examples/predict/poisson3d.rs`** (`predict_poisson3d`): `predict_poisson` for a 3D
   run: `load_fno::<5>`, `load_poisson3d_uniform_test`, `load_poisson3d_cheb_field`,
   `predict_host`, `poisson_errors_nd`, `evaluate_3d`; prints the report, writes
   `errors.json`, records the inference time in `costs.json`, exits 1 on a failed
   cross-check.
5. Root `CLAUDE.md` Layout and Commands; no change to `training/`, `runs`, the loaders or
   `models/`.

Conventions: none changed (CONVENTIONS §12's 3D norms and error 5 are CL1.1's).

## Tests that prove it
- D2, the oracle being today's `poisson_errors` frozen in `tests.rs`:
  `nd_rank_two_is_bit_identical_to_2d` (extended: its cases plus the zero-on-∂Ω and zero
  predictions, through both entry points, standard and transposed layouts, against the
  frozen oracle; it asserts that the zero cases give û = 0 on ∂Ω). Bit identity.
- The existing 2D `metrics::poisson` tests unedited (except the two whose reference was a
  removed function), and the whole 2D suite.
- `evaluate`: the existing tests unedited (the 2D record and report unchanged);
  `evaluate_3d_differs_only_in_error1_and_the_plot_hint`.
- The examples: built by `cargo clippy --examples`; the Done when by the short run below.

## Checks
`/verify` (fmt, the three CI clippy jobs, `cargo clippy --all-targets`, `cargo test`,
`cargo test --features datagen`, `cargo doc`); `cargo clippy --features metal --examples`.
The short run, release, metal, on the Mac (about 60 s an epoch at s = 32, CL0.1):

```sh
EPOCHS=2 cargo run --release --features metal --example train_poisson3d -- 4 33
cargo run --release --features metal --example predict_poisson3d -- runs/poisson3d_fno_n33_s32_p4_<t>
```

Expected: the run directory with the configs, `metrics.csv`, normalizers, weights,
checkpoint and `costs.json`; `predict_poisson3d` prints errors 2–5, both cross-checks ok
(error 2 compared with the sidecar at s = n − 1 = 32), writes `errors.json`, exits 0.
Errors after two epochs are not a result; CL4.2 is the baseline.

## VM runs
None (CL4.2).

## Constraints and risks
- Constraint 1: the per-sample loop change is on a 2D path; the frozen oracle and the
  zero-on-∂Ω case guard it (a −0.0 surface sum would change error 5's sign bit).
  `train_poisson`, `predict_poisson` and the 2D record are not edited.
- `training/` is held by optimisers CO1.2: not touched (the rank-generic path exists).
- Error 3's cross-check compares f64 host errors with f32 device test_l2 to 1e-5, as in 2D;
  in 3D the sums run over 32³ points, so round-off is larger. If it fails on the short
  run, that is a finding for the author, not a tolerance to widen here.

## Short run (the Done when), 2026-11-26
Mac (M2 Pro), metal, release, at 3ae7a04: `EPOCHS=2 train_poisson3d -- 4 33` wrote
`runs/poisson3d_fno_n33_s32_p4_1795700678/` (configs, `metrics.csv`, normalizers, weights,
checkpoint, `costs.json`; 56,631,841 parameters); test_l2 0.341 after epoch 0, 0.211 after
epoch 1. `predict_poisson3d` on it exited 0: errors 2 / 3 / 4 mean 1.91e-4 / 0.2105 /
0.2105, error 5 relative RMS 5.9e-5; error 2 against the sidecar rel diff 1.4e-16 (mean),
0 (max); mean error 3 against test_l2 rel diff 1.7e-7 (tol 1e-5); `errors.json` written.
Epochs took 2,009 s and 1,594 s against CL0.1's ~60 s on metal: the Mac was loaded
(load average 27 on 10 cores, the optimisers lane's training runs), so these times are not
a measurement. A first attempt was killed by the tool's 10-minute limit during epoch 0 and
its directory removed. The final code changes after this run (review fixes F1, F2) touch
neither the training path nor the numbers `predict_poisson3d` computes.

## Pending runs
None.

## Open questions
- Q1 (answered by the author 2026-11-26: "Reuse the 2D record"). The 3D errors record
  has exactly the 2D record's fields (only error 1's quote differs), so `evaluate_3d`
  returns the existing `PoissonEvaluation` / `PoissonErrorsRecord`, no new type, as CL3.3
  did with `PoissonConfig`. The roadmap line's "new 3D types" is amended in this PR.
