# T5: the θ-map model

Phase 9, task 5 (Phase 8 T2 until 2026-11-09). Branch `phase9/T5-theta-map`. Depends on Phase 5 T7 (the baseline) and
Phase 4s T16 (the ungated `chebyshev` module), both merged. Option 2 of design §6, old
Phase 6 T3, then Phase 8 T2 (§10, decision 30, amendment of 2026-11-09). The comparison against the Phase 5 baseline is T6, not
this task. The open choices are in "Decisions for the author" at the end. The author
decided them on 2026-11-09 (all as recommended), and the brief matches.

## Read first
- This phase's README ("Inputs from earlier phases", the T5 row) and the Phase 8 README
  ("Inputs from earlier phases", "Open items").
- Design §6.1 (the option 2 row), §10 Phase 9, §13.6 (confounders), §13.11; decisions 2,
  10 ("Phase 5 reframed"), 23, 24 and 30 (D2, D3 and the amendment of 2026-10-29: the
  baseline).
- `docs/CONVENTIONS.md`, all of it. T5 changes §1–§5, §7, §9 and §12.
- `src/neural_operators/models/fno/mod.rs` and `models/fno/tests/padding.rs` (the
  pattern T5 copies); `layers/spectral_convolution.rs` (`corner_ranges`,
  `check_modes_fit`, `fft_ctensor`).
- `data/loaders/poisson/{mod,uniform,cheb}.rs`; `metrics/poisson.rs` (`poisson_errors`,
  `evaluate`).
- `examples/train/poisson.rs`, `examples/predict/poisson.rs`,
  `examples/predict/poisson_resolution.rs`; `tests/gpu_parity.rs` (`replay_2d_poisson`).
- Burn at `ddfa9af`: `crates/burn-signal/src/functions/fft.rs` (`rfft`, `cfft`) and
  `crates/burn-tensor/src/tensor/api/base.rs` (`flip`).

## Facts verified at `13d2b72`
- **The padding slot.** `FNO::forward` pads once after `fc0` and crops once after the
  last spectral layer, before `fc1` (`models/fno/mod.rs:258-272`), using `pad_spatial`
  and `crop_spatial` (`:166-179`). `padding` is a plain `usize` field (`:65-67`), not a
  parameter. Burn makes both `usize` and `bool` empty modules
  (`burn-core/src/module/param/constant.rs:64,73`), so a `bool` field leaves checkpoints
  unchanged too.
- **The grid is already θ.** `grid_cl` builds j/(s_a − 1) along each axis
  (`mod.rs:196-223`). On the n CGL nodes x_j = −cos(πj/(n − 1)) (CONVENTIONS §12), so
  θ_j = πj/(n − 1), and the grid `grid_cl` builds on an n-node input is θ/π.
- **No Bluestein.** `rfft` takes the radix-2 path at power-of-two lengths and Bluestein
  otherwise (`fft.rs:64-69`). `cfft` is two `rfft`s (`fft.rs:372-375`). The extension of
  n = 2^k + 1 nodes has L = 2(n − 1) = 2^(k+1): 64, 128, 256 and 512 for n = 33, 65, 129
  and 257. So no θ-map FFT reaches the Metal Bluestein bug (`spectral_convolution.rs:306-308`),
  and the rfft half-width L/2 + 1 = n is odd. Not known: whether the extension's own
  `flip` and `cat` are correct on Metal. The fused-`cat` bugs in `tests/gpu_parity.rs`
  are why T5 adds a parity case.
- **Mode limits.** `check_modes_fit` asserts m ≤ L/2 on non-last axes and m ≤ L/2 + 1 on
  the last (`spectral_convolution.rs:241-254`). On an extended axis that means m ≤ n − 1
  and m ≤ n.
- **Evenness is approximate.** The low and high corners hold independent complex weights
  (`spectral_convolution.rs:223-235`). An even input has a real spectrum with
  X[L − k] = X[k], but the layer's output is in general not even (§6.1, "evenness of the
  output only approximate"). The crop keeps the half that lies on the nodes.
- **The version lives in two places.** `CONVENTION_VERSION = 2` is in
  `docs/CONVENTIONS.md:3` and §9, and in code at `src/lib.rs:76`.
  `convention_version_matches_conventions_md` (`src/lib.rs:83-90`) fails if the two
  differ. Sidecars record the version (`pde/poisson/dataset.rs:280`), and the loaders do
  not check it (`data/loaders/poisson/mod.rs:132`, CONVENTIONS §9).
- **Data.** `read_uniform_split` applies T_cu (`uniform.rs:151-171`), and that is the
  only CGL → uniform step. `load_poisson_cheb_field` reads raw CGL fields with the same
  validation (`cheb.rs:26-46`). `PoissonConfig::s()` defaults to n − 1
  (`poisson/mod.rs:104-106`). The MLflow `group` tag reads `s()`
  (`tracking/builder.rs:126,139`), and so does `poisson_matrix`.
- **Errors.** `poisson_errors` takes `&Transfers` and computes errors 2–5 per sample
  (`metrics/poisson.rs:77-140`). `evaluate` uses the transfers only for n and s (`:341`).
  It skips the error 2 cross-check when the sidecar's s is not the run's (`:367`).
- **Examples.** `train_poisson` sets modes = min(12, s/2) (`examples/train/poisson.rs:183`),
  reads s from its third argument (`:173`) and defaults `PADDING` to 4 (`:106`), so
  baseline runs pass p = 0. `poisson_plot` and `poisson_resolution` assert s = n − 1
  (`poisson_plot.rs:93-99`, `poisson_resolution.rs:62-69`).

## Goal
`train_poisson -- 0 129 theta` trains on the CGL fields themselves. Inputs and targets
stay on the n × n nodes, with no T_cu. Inside the model, every spatial axis is extended
evenly to 2(n − 1) points around the spectral layers, so the FFT computes a DCT-I.
`predict_poisson` and `poisson_resolution` evaluate such a run. With the option off,
every path is bit for bit what it is today, and saved runs still load. CONVENTIONS is
amended to version 3 in the same PR, ahead of the code that relies on it (§9).

## Design (P8–P17 as chosen below)
1. **Config.** `FNOConfig::even_extension: Option<bool>`. `None` and `Some(false)` mean
   no extension. Old configs load it as `None`. `FNO` gets a `bool` field next to
   `padding`.
2. **Forward.** In the padding slot, after `fc0`, each spatial axis of length s_a is
   replaced by L_a = 2(s_a − 1) values, whole-sample symmetric:
   `cat([v, flip(v[1..s_a−1])])`. The endpoints θ = 0 and π are not repeated: v_ext[j] =
   v[j] for j < s_a, and v_ext[j] = v[L_a − j] for s_a ≤ j < L_a. After the last
   spectral layer the existing `crop_spatial` keeps the leading s_a. Extending after
   `fc0` equals extending the input and grid channels, because `fc0` is pointwise.
3. **Panics.** `init` panics if `even_extension` is set together with `padding > 0`.
   `forward` panics if some s_a < 3.
4. **Unchanged.** The grid (θ/π, as above) and `SpectralConv`. Modes are
   min(12, n − 1): D2's s/2 cap becomes L/2 = n − 1.
5. **Data.** `PoissonConfig::cgl: Option<bool>` with `s = Some(n)`. The Poisson loaders
   skip T_cu when it is set and normalise as today (CONVENTIONS §10).
6. **Errors.** Predictions are already on the nodes, so R = P = I. Error 2 is 0, error
   3 is the equal-weight training metric on the nodes, and errors 4 and 5 use
   Clenshaw–Curtis weights on û = the prediction.
7. **Entry points.** `theta` in `train_poisson`'s s slot. `predict_poisson` and
   `poisson_resolution` branch on `cgl`.

## Deliverables
1. **CONVENTIONS diff, version 3.** It is proposed in the PR description, and
   `docs/CONVENTIONS.md` and `src/lib.rs:76` are bumped together in the same PR:

   | Section | Change |
   | --- | --- |
   | Header, §9 | `CONVENTION_VERSION = 3`. History: version 3 adds the even extension (the θ-map). Version 2 configs, checkpoints and datasets stay valid: the field is optional and off by default |
   | Summary | One clause: on CGL data, an even extension may replace padding (§3) |
   | §1 | The flow becomes `pad or extend (§3)` … `crop`. `even_extension` joins the list of fields added after release (`spectral_init`, `padding`) |
   | §2 | The formula is unchanged. Added: on CGL nodes (§12) it is θ/π, with θ_j = πj/(n − 1). It is built before the extension and extended with the data (a triangle wave in θ) |
   | §3 | New part, "Even extension": the definition in item 2 above, the slot, exclusion with p > 0, s_a ≥ 3, not a parameter, not in the weights |
   | §4 | The FFT of the extended real field is the DCT-I of the node values, X[k] = v_0 + (−1)^k v_{n−1} + 2 Σ_{j=1}^{n−2} v_j cos(πjk/(n − 1)), real and even in k. L is a power of two for n = 2^k + 1 |
   | §5 | The limits apply to L_a: m_a ≤ s_a − 1 (a < D) and m_D ≤ s_a. Index k on an extended axis is Chebyshev degree k, since cos kθ = (−1)^k T_k(x) with x = −cos θ. The corners keep independent weights, so the output is only approximately even |
   | §7 | New identities: extension off is bit-identical; crop ∘ extend = identity, bit for bit; the DCT-I identity of T_q (Tests) |
   | §12 | Chebyshev data reach the model on a uniform grid (option 1) or, with the θ-map, on the CGL nodes, with no R and no P. The learning grid is then the n nodes, and the transfer-only error is 0 |
   | §11 (no bump) | Note: on CGL nodes `LpLoss::rel` uses equal weights, which is not the L² norm (P13) |

2. **Model** (`models/fno/mod.rs`). The field, a private `extend_even` next to
   `pad_spatial`, and the branch in `forward`. Rustdoc gives tensor shapes and panics,
   citing CONVENTIONS §3. `SpectralConv` is unchanged.
3. **Data** (`data/loaders/poisson/`). `PoissonConfig::cgl` (`Option<bool>`; `None` is
   uniform, as for every saved config). With `cgl`, `load_poisson_uniform{,_test}` skip
   T_cu and return `[N, n, n, 1]` inputs and `[N, n, n]` targets. `s` must be `None` or
   `Some(n)`, otherwise `LoadError::Invalid`. The doc comments say so.
4. **Evaluation** (`metrics/poisson.rs`). Add `poisson_errors_cgl(pred, u)` for
   predictions on the nodes. It shares the per-sample body with `poisson_errors`, and its
   error 2 is 0 for every sample. `evaluate` takes `(n, s)` in place of `&Transfers`,
   which it only reads for those two values. `PoissonErrorsRecord` is unchanged:
   `s = n`, and the run's `data_cfg.json` says which grid.
5. **Entry points.**
   - `train_poisson`: `theta` in the s slot sets `cgl`, `even_extension`,
     `s = Some(n)` and modes min(12, n − 1). It panics unless padding is 0. The run is
     named `poisson_fno_n{n}_theta_p0…`.
   - `predict_poisson` and `poisson_resolution`: call `poisson_errors_cgl` when `cgl`
     is set. `poisson_resolution` builds each n's config with `cgl` and `s = Some(n)`.
   - `poisson_plot` stays as it is: its s = n − 1 assert already refuses θ-map runs.
6. **Docs.** `CLAUDE.md` (Commands, Layout); the module docs of the touched files; this
   phase's README, Results row "θ-map model and convention diff".

## Tests
Every existing test passes unchanged. That includes `padding_zero_is_bitwise_identical`,
`checkpoint_round_trip_matches_old_forward` and the loader and metrics tests.
- **Option off** (`models/fno/tests/extension.rs`, new). `None` and `Some(false)` give
  bit-identical output to the pre-change forward, in 1D, 2D and 3D, as
  `check_padding_off_is_bitwise` does. The extension adds no parameters. A config
  without the field loads as `None`.
- **Round trip.** `crop(extend(v))` equals v bit for bit. `extend(v)` is even: v_ext[j]
  = v_ext[L − j] for 1 ≤ j < L. Lengths are 2(s_a − 1) on every axis, for odd and even
  s_a.
- **DCT-I identity.** v_j = T_q(x_j) on n = 17 nodes, rfft of the extension. Bin q
  holds (−1)^q (n − 1) for 0 < q < n − 1, and (−1)^q L for q ∈ {0, n − 1}. The other
  bins are 0, and the imaginary part is 0. Tolerance 1e-4 absolute in f32 (CONVENTIONS
  §8).
- **Forward against a reference.** Copy `check_padded_forward`: the θ-map forward
  matches a reference built in the test from an explicit extension. It differs by at
  least 1e-3 from the wrong conventions: half-sample symmetric (endpoints repeated,
  length 2s_a), zero padding to L, and a grid built on the extended axis. Then the
  gradients, as `padded_gradients_match_reference` does.
- **Panics.** `even_extension` together with p > 0 panics in `init`; s_a < 3 panics in
  `forward`; modes above n − 1 hit `check_modes_fit`. Each test checks its message.
- **Loader.** With `cgl` set, the loaders return shapes `[N, n, n, 1]` and `[N, n, n]`,
  equal to `normalize_split_pair` on the raw fields. The test split loaded with saved
  normalisers matches the refit one bit for bit. `s = Some(n − 1)` with `cgl` is
  `Invalid`. An old `data_cfg.json` without `cgl` loads as uniform.
- **Metrics.** For `poisson_errors_cgl`: pred = u gives errors 2–4 equal to 0; one
  perturbed node gives the hand-computed equal-weight and CC values. `evaluate` with
  s = n reports the error 2 cross-check as skipped.
- **Smoke.** `train_poisson_theta_smoke` in `training/trainer.rs` uses the n = 9
  fixture of `train_poisson_smoke` (L = 16) with `cgl` and the extension. Two epochs on
  flex, finite metrics, and the run reopens and predicts.
- **GPU parity** (`tests/gpu_parity.rs`). Add a θ-map case to `replay_2d_poisson`
  (input 65², L = 128), forward and backward, against Flex. Run it with
  `--features metal`, and say whether it ran and what it gave.

## Acceptance (sanity only; the comparison is T6)
- Time the first epochs on stage 2 (n = 65, L = 128) and state the timing before the
  run. As an estimate, not measured: the spectral grid is 128², as for the baseline at
  s = 128, which took 2.45 s/epoch on cuda (Phase 5 T5).
- Then train a short run, at least to the first checkpoint (25 epochs): flex, or Metal
  if the parity case passes. `test_l2` must fall. Run `predict_poisson` on it: the
  cross-checks pass (error 3 against `metrics.csv`, error 2 skipped) and error 2 is 0.
  Report errors 3, 4 and 5 as a sanity check only.
- `poisson_resolution` on that run at n = 129 and 257 (the K = 32 sets) runs end to end.
  Report the numbers but draw no conclusion (T6).
- Load a Phase 5 baseline run and re-run `predict_poisson` on it: its `errors.json` is
  unchanged.
- Fill in the README Results row (T5).

## Checks
`cargo fmt`, then the CI set in `CLAUDE.md` and `cargo clippy --all-targets -- -D warnings`.
The change is backend-sensitive (new tensor ops in `forward`), so also run
`cargo test --features metal --test gpu_parity` and the extension tests with
`--features metal`, and report whether they ran.

## Do not
- Change the behaviour of the baseline path: option off, uniform data, `SpectralConv`,
  `grid_cl`, the losses, the trainer and `train_components`'s signature.
- Edit CONVENTIONS without proposing the diff in the PR description and bumping
  `CONVENTION_VERSION` in both places.
- Train comparison runs, run seeds or write the T6 tables. That is T6.
- Change `poisson_matrix`, `poisson_plot` or the MLflow exporter, or depend on MLflow.
- Regenerate datasets, or make the loaders check a sidecar's version.

## Decisions (chosen by the author, 2026-11-09)
Numbered after Phase 9's planning decisions P1–P7 (Q8–Q17 in the draft written for
Phase 8). Each is recorded in design §12,
decision 30 (amendment of 2026-11-09, P8–P17). **All ten were chosen as recommended by
the author (AHA-HH), 2026-11-09**; the reasoning is kept below.

- **P8, where the extension lives.** Options: (a) an `FNOConfig` option inside the
  existing model; (b) a wrapper model around `FNO`; (c) a data transform that extends the
  fields in the loader. **Chosen: (a)** (author, 2026-11-09). It needs one field, one helper and one
  branch in the padding slot, and the off path is untouched, so it stays bit-identical.
  (b) duplicates `forward`, because the extension must sit between `fc0` and `fc1`.
  (c) is wrong: the model would build its grid ramp over L points, which is not even,
  and the crop would leak into the loss and predict.
- **P9, extend once or around each layer.** (a) Once, around the block of spectral
  layers, in padding's slot. The pointwise layers and ReLU run on L² points and keep
  evenness exactly; the odd part from the complex weights accumulates over the layers.
  (b) Around each `SpectralConv`: extend, transform, crop. Every layer is then an exact
  operator on node values, the pointwise layers run on n² points, and there are 4×
  more `flip`/`cat` ops (the Metal risk). **Chosen: (a)** (author, 2026-11-09). It is the smallest
  change, it is how §3 places padding, and it is the variant §6.1's "evenness only
  approximate" describes. (b) can be a T6 arm if (a) disappoints.
- **P10, grid channels.** (a) θ/π, which is `grid_cl` unchanged on the n-node input;
  extended, it is a triangle wave. (b) The physical ξ = (x + 1)/2 = (1 − cos θ)/2:
  smooth on the circle, but it needs a new grid function and a §2 formula change.
  **Chosen: (a)** (author, 2026-11-09). It needs no code, and §6.1 says "grid channels in θ".
- **P11, modes.** (a) min(12, n − 1): the baseline's count and its parameter count
  (2,368,001 at width 32), but Chebyshev degree < 12. (b) Matched bandwidth: T_k has k
  zeros on [−1, 1] and the baseline's FFT index q has 2q, so about 24 modes, with about
  4× the spectral parameters. An estimate, not measured. **Chosen: (a)** (author, 2026-11-09)
  as the default. It changes one thing, the basis, so T6 compares like with like. (b)
  is a natural T6 arm, mirroring the dropped mode-control arm of D2.
- **P12, data path and run record.** (a) `PoissonConfig::cgl`, handled inside
  `load_poisson_uniform{,_test}`. (b) New `load_poisson_cgl{,_test}`, chosen at each call
  site. **Chosen: (a)** (author, 2026-11-09). It adds no new public function and no new loader pair
  (S3), and the examples need no branch at load time. The cost is a misleading name
  ("uniform"), which is left for the S3 cleanup. Either way `s = Some(n)`, so the MLflow
  `group` tag reads `p0_s129` and is not mistaken for a uniform cell.
- **P13, training loss on the nodes.** (a) `LpLoss::rel` with equal weights, unchanged.
  Error 3 is then the training metric, as today. (b) A CC-weighted relative loss: the
  true L² norm, but a new loss and a §11 change. Equal weights on CGL nodes weight the
  boundary bands, where the nodes cluster and u = 0, more heavily than L² does.
  **Chosen: (a)** for T5 (author, 2026-11-09). One change at a time, and error 4 already reports the CC
  norm. Raise (b) at T6 if errors 3 and 4 diverge.
- **P14, command line.** (a) `theta` in the s slot: `train_poisson -- 0 129 theta [seed]
  [N subset_seed]`. (b) A `--theta` flag, which shifts the positional arguments. (c) An
  environment variable. **Chosen: (a)** (author, 2026-11-09). The s slot is where the learning grid is
  named, and no other argument moves.
- **P15, scope of evaluation.** `predict_poisson` and `poisson_resolution` in T5;
  `poisson_plot` and `poisson_matrix` not. **Chosen as stated** (author, 2026-11-09). T6 needs errors
  2–5 and the resolution study, and `poisson_resolution` needs only the branch of
  `predict_poisson`. The plot and the matrix table are presentation; T6 can add them.
- **P16, the review leftovers S1, S3, R1** (Phase 8 README, "Open items"). (a) Clear them first.
  (b) Work around them. **Chosen: (b)** (author, 2026-11-09). T5 adds no run file (S1 is untouched), no
  argument to `train_components` (R1), and with P12 (a) no loader pair. The one S3 copy
  it touches, predict → decode, gets a branch on which errors function to call, not a
  new copy. Clearing them first would put three refactors in front of a model task,
  against "no drive-by refactors".
- **P17, acceptance backend.** Flex, or Metal once the parity case passes.
  **Chosen** (author, 2026-11-09): time on both; run on Metal if parity holds, otherwise on flex
  (CONVENTIONS §8 requires results to hold on flex).
