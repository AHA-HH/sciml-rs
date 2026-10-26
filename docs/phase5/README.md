# Phase 5: grid-decoupled operator learning

As of 2026-10-26; planned, not started (design §12, decision 30, proposed). Design:
`docs/design/2d-chebyshev-poisson-fno.md` (§10 Phase 5, §13). `docs/CONVENTIONS.md` wins on
any conflict.

Phase 5 asks how coarse the forcing representation and the FNO's learning grid can be
while a fine-grid Poisson solve provides the reference. Today the grids are coupled: the
129² Chebyshev solve feeds a 128² uniform FNO grid, s = n − 1. This phase decouples the
solver grid G_s, the learning grid G_l and the evaluation grid G_e (design §13.3). It
reports three errors:
- the coarse error on the learning grid;
- the fine-grid error after prolongation;
- the transfer-only error, the floor no model can beat.

It also fixes the spatial baseline that Phases 6 and 7 use: learning s, restriction and
prolongation, padding, modes and training settings.

It absorbs old Phase 6 T1–T2 (decision 24): the baseline's padding, and several seeds with
error 4 against ‖u‖. The model is not redesigned in this phase (design §13.7).

## Read first

- Design §3.3 (forcing family, K, the tail table), §5.3 (stages), §6.2 (transfers), §7
  (the five errors), §13 (the whole programme; §13.3 is this phase, §13.6 the
  confounders, §13.15 the open decisions), §12 decisions 5, 8, 16, 17, 23, 24 and 30.
- `docs/CONVENTIONS.md` §2 (grid channels), §3 (padding), §4 (FFT, Bluestein), §5 (mode
  limits, m ≤ s/2), §10 (normalisers), §12 (Chebyshev grids and transfers).
- `docs/phase4/README.md` Results (stage 3 dataset, the 129² run, the resolution study and
  the padding check).
- `CLAUDE.md`: Working agreement, Checks, Burn.

## Inputs from Phase 4 and 4s

| Item | Value | Source |
| --- | --- | --- |
| Stage 3 dataset | n = 129, K = 64; 1000 train / 200 test; `datasets/poisson/poisson_n129_K64_{train,test}.{npz,json}` (main checkout) | Phase 4 T3 |
| Its label check (exact sine series), max | train 9.17e-13, test 8.97e-13 | Phase 4 T3 |
| Error 2 at s = n − 1 = 128, mean / max (test) | 2.53e-6 / 9.50e-6 | Phase 4 T3 |
| Stage 4 dataset (reference check) | n = 257, K = 64, the same seeds; label check max 1.6e-12 | Phase 4 T4 |
| The coupled 129² baseline | `runs/poisson_fno_n129_p4_1792037830`: cuda, p = 4, seed 42, test relative L² 1.079e-2, about 112 s/epoch | Phase 4 T3 |
| Padding check (65² models, mean error 3 at 65 / 129 / 257) | p = 0 1.138e-2 / 1.378e-2 / 1.608e-2 (1.4×); p = 4 1.069e-2 / 4.666e-2 / 7.546e-2 (7.1×) | decision 23 |
| Padding cost (cuda, 65²) | 1.7 s/epoch at p = 0, 32.8 at p = 4 (Bluestein) | decision 24 |
| Seeds | every Phase 4 result comes from one seed (42) | decision 24 |
| Code in place | `chebyshev::transfer` (`cheb_to_uniform`, `uniform_to_cheb`, `apply`, any s); `PoissonConfig::s()` = n − 1; `RunDir`; `poisson_errors`; `poisson_resolution` | Phase 4s |

## Planning decisions

Pending. Each is decided when its task's brief is approved and recorded in design §12
under decision 30. The recommendations are the design's (§13.15).

- **D1, restriction R:** filtered Chebyshev restriction, chosen by measured transfer-only
  error in T1 against cell average and unfiltered T_cu. P stays Floater–Hormann d = 2
  unless T1 shows it dominates at small s.
- **D2, modes:** main matrix m = min(12, s/2); control arm m = s/4. Modes [12, 12] panic at
  s = 16 (CONVENTIONS §5).
- **D3, padding:** p = 0. A fixed physical fraction only as its own version-bumping
  CONVENTIONS §3 proposal.
- **D4, seeds:** 3 per configuration; 5 if the spread between seeds is comparable to the
  differences being compared.
- **D8, evaluation grids:** 129 CGL and each learning grid; 257 only in T1's reference
  check.

D5–D7 belong to Phases 6 and 7.

## Tasks

Rows are proposals; one brief per task (`docs/phase5/T<k>-<name>.md`) is written before
the task starts. Task numbers are execution order (design §10). One task per branch and
PR: `phase5/T<k>-<name>`.

| Task | Delivers | Depends on |
| --- | --- | --- |
| T1 | Reference check: stage 4 u restricted to the 129 nodes against stage 3 u. The transfer-only error ‖P R u_f − u_f‖ for each candidate R at s ∈ {16, 32, 64, 128}, and the filtered-out energy of f at each s. Fixes D1 | – |
| T2 | The chosen R and P as reusable functions in `chebyshev::transfer`, with oracle tests (exact on resolved modes, no aliasing of an unresolved mode) | T1 |
| T3 | Resolution-aware loader: learning s in `PoissonConfig`, independent of n; s = n − 1 stays the default and today's runs reproduce | T2 |
| T4 | Run metadata and costs (design §13.9): solver n, learning s, R, P, padding, modes, seeds, per-epoch time, `costs.json`, parameter count | T3 |
| T5 | The training matrix of design §13.8: main arm, mode control, aliasing control; cost of each arm timed and stated before it runs | T4 |
| T6 | The three errors (and errors 2–5) on every run; error 4 against ‖u‖ over the test set (decision 22); comparison table and accuracy/cost plots | T5 |
| T7 | The Phase 7 baseline (s, R, P, padding, modes, settings), justified from T6 | T6 |

## Module layout

Proposals; the briefs fix them.

- `chebyshev/transfer.rs`: the filtered restriction beside `cheb_to_uniform` (ungated,
  ndarray).
- `data/loaders/poisson/`: `PoissonConfig` gains the learning size and restriction
  method; `uniform.rs` uses them.
- `metrics/poisson.rs`: the coarse, fine-grid and transfer-only errors beside
  `poisson_errors`.
- `runs/`: the metadata and `costs.json` of design §13.9; `RunDir` stays the only owner of
  the run-file names.
- `examples/train/poisson.rs`, `examples/predict/poisson.rs`,
  `examples/predict/poisson_resolution.rs`: a learning-size argument. The task that adds
  a command adds it to the Commands section of `CLAUDE.md`.

## Results

Filled in as tasks merge. Phase 5 is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| Reference check: stage 4 restricted to 129 against stage 3, max relative CC-L² | | T1 |
| Transfer-only error per R and s (mean / max), filtered-out energy of f per s; D1 chosen | | T1 |
| R and P tests; wall time of the new tests | | T2 |
| Loader at s = n − 1 reproduces today's tensors | | T3 |
| Metadata and `costs.json` written by a short run | | T4 |
| Main arm: coarse / fine-grid / transfer-only error per s, mean ± spread over seeds | | T5, T6 |
| Main arm costs per s: time per epoch, total training time, inference time, peak GPU memory, host-to-device bytes, parameter count, backend | | T5, T6 |
| Mode control and aliasing control against the main arm | | T5, T6 |
| Error 4 against ‖u‖ over the test set | | T6 |
| Dataset generation time (solver cost, reported separately) | | T1 (sidecars) |
| Phase 7 baseline and its justification | | T7 |

## Exit checklist

- [ ] T1–T7 merged.
- [ ] The solver and learning grids are independently configurable.
- [ ] The same forcing examples are represented at several learning resolutions.
- [ ] Restriction and prolongation errors are measured separately.
- [ ] Coarse-grid and fine-grid FNO errors are both reported.
- [ ] Accuracy/cost trade-offs are reproducible across seeds.
- [ ] The baseline resolution for Phase 7 is justified.
- [ ] D1–D4 and D8 recorded in design §12 under decision 30.
- [ ] Every row of Results filled in.
- [ ] The "Current phase" bullet in `CLAUDE.md` is updated for what comes next.
