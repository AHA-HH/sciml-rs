# T3: `predict_poisson` and the five errors

Phase 3, task 3. Branch `phase3/T3-predict-poisson`. Depends on T1 (`clencurt`,
`cc_rel_l2_error`) and T2 (a saved 65² run).

## Read first
- Design §3.4 (norms), §7 (the five errors, the boundary condition), §11, §12 decisions
  10, 14, 15 and 16.
- `docs/CONVENTIONS.md` §10 (the `y_test` asymmetry), §11 (`LpLoss`) and §12.
- `examples/predict/darcy.rs`, the model to follow; `training/trainer.rs` (`eval_epoch`,
  `flatten_pair`); `data/transforms/normalizers.rs` (`UnitGaussianNormalizer`,
  `NormalizerRecord`, `FlatDecoder`).
- `src/neural_operators/data/loaders/poisson.rs` (the split reader and its validation);
  `chebyshev::{cheb_to_uniform, uniform_to_cheb, apply}` and T1's `clencurt`,
  `cc_rel_l2_error`.
- `datasets/README.md` (the sidecar fields) and the Results of `docs/phase1/README.md`
  (error 1) and `docs/phase3/README.md` (T2's runs).

## Goal
For a saved T2 run, the five errors of design §7 on the stage 2 test split, mean and max
over the samples, and the verdict on whether the boundary-condition ablation is warranted.

## Deliverables
1. **`load_poisson_cheb(path, n, count) -> Result<Array3<f64>, LoadError>`** in
   `data/loaders/poisson.rs`: the first `count` raw solutions `u` `[count, n, n]` of one
   split, on the CGL grid, with the same `x`/`y` node and shape validation as
   `load_poisson_uniform`. Share that validation rather than copy it. The behaviour of
   `load_poisson_uniform` and its tests do not change.
2. **`src/neural_operators/metrics/poisson.rs`**, ungated, ndarray, f64. Given decoded
   uniform predictions `[N, s, s]` and raw `u` `[N, n, n]` (s = n − 1, d = 2), per sample
   and as mean / max:
   - **error 2 (transfer):** `cc_rel_l2_error(T_uc(T_cu u), u)`;
   - **error 3 (model):** relative L² of the prediction against `T_cu u` on the uniform
     grid, equal weights, as `LpLoss::rel` computes it;
   - **error 4 (total):** `cc_rel_l2_error(T_uc(prediction), u)`;
   - **error 5 (boundary):** with û = T_uc(prediction), max over ∂Ω of |û| / max over Ω
     of |u|; and the relative boundary RMS, √(∮ û² ds / 8) / √(∫∫ u² dA / 4), with
     `clencurt` weights along each of the four edges (each corner counted once) and over
     Ω.
   T_cu and T_uc are built once per call. Shapes and panics documented.
3. **`examples/predict/poisson.rs`**, registered as `predict_poisson`, no required
   features: `cargo run --release --example predict_poisson -- runs/<dir>`.
   - Loads the model, config and `y_normalizer.json` as `predict_darcy` does; reads the
     `PoissonConfig` from `data_cfg.json`; runs on `Device::default()`.
   - Uniform test inputs from `load_poisson_uniform`, raw `u` from `load_poisson_cheb`.
     Predictions are decoded with the y normalizer and copied to the host as f64.
   - Prints a table of errors 1–5 (error 1 quoted from Phase 1: A floor ≤ 1e-12, A vs C
     order 1.993–1.999; ungated, so not recomputed, decision 16). Cross-checks error 2
     against the test sidecar's max and mean to 1e-12 relative, and error 3's mean
     against the run's last `test_l2` in `metrics.csv` to 1e-5 relative (f32). Prints
     the boundary verdict.
   - Writes `errors.json` into the run directory (serde via `burn::config::Config`, as
     the other run files are written).
4. **The boundary verdict** (decision 16, fixed before the run): the ablation is
   warranted if the mean relative boundary RMS is at least 10% of the mean error 4.
5. Docs: Layout lines for `metrics::poisson` and `load_poisson_cheb`, and the
   `predict_poisson` command, in `CLAUDE.md`.

## Tests
In `metrics/poisson.rs` (ungated, synthetic, no dataset):
- `exact_prediction_has_only_transfer_error`: prediction = T_cu u for a smooth u at
  n = 17. Error 3 is ≤ 1e-14 and error 4 equals error 2 to 1e-14. If u vanishes on ∂Ω,
  error 5 is ≤ 1e-14: both grids share the endpoints ±1, so T_cu and T_uc both carry
  boundary values exactly.
- `boundary_offset_is_measured`: prediction = T_cu u + c gives a max ratio and a
  relative RMS that match the closed form for the constant c, to 1e-12.
- `error3_matches_lploss`: on the same random fields, error 3 equals
  `LpLoss::new(2, 2, Reduction::Sum).rel` divided by N, computed on Flex, to 1e-6.
- Shape mismatches panic with a clear message.

In `data/loaders/poisson.rs`: `cheb_loader_reads_raw_u` (shape and values against the
fixture) and its error cases (missing file, non-CGL nodes, too few samples), each a
`LoadError`.

## Acceptance
- All tests pass under `cargo test` and `cargo test --features chebyshev`, on macOS and
  in both CI jobs; nothing new needs `chebyshev`.
- `predict_poisson` run on T2's chosen-p runs from metal and flex. The five errors (mean
  and max), both cross-checks and the verdict are recorded in the T3 rows of Results in
  `docs/phase3/README.md`.
- If a value contradicts the design (for example error 4 well below error 3, or error 2
  off from the sidecar), it is recorded in the design's §12 before the phase closes.
- The Phase 3 exit checklist is ready to tick, apart from the CLAUDE.md pointer.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters
(`neural_operators::metrics::poisson::tests::<name>`,
`neural_operators::data::loaders::poisson::tests::<name>`) and check the "running N
tests" lines. Inference reaches a backend: run `error3_matches_lploss` and the example
with `--features metal` too, and say whether that was done.

## Do not
- Add the hard-constraint ablation, even if the verdict warrants it: a warranted
  ablation is planned as a new task.
- Change the model, the training code, T2's runs or the existing loader behaviour.
- Depend on the `chebyshev` feature, or recompute error 1 with solvers A or C.
- Store transferred fields on disk.
