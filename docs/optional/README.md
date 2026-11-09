# Optional phase: the θ-map

As of 2026-11-09: **optional, not started.** T1's brief is approved and its decisions
P8–P17 were chosen by the author on 2026-11-09. Design:
`docs/design/2d-chebyshev-poisson-fno.md` (§6.1 option 2, §10 "Optional phase: the
θ-map", §13.18, decision 30 and its amendments of 2026-11-09). `docs/CONVENTIONS.md` wins
on any conflict.

The θ-map trains the FNO on the Chebyshev data directly. With x = −cos θ the CGL nodes
are evenly spaced in θ, and an even extension of each field makes it periodic, so the
unchanged FFT layers compute a DCT-I: mode k is Chebyshev degree k. The fields are not
interpolated to a uniform grid and back (error 2 is 0), and no padding is needed. The
question is whether this beats the Phase 5 baseline, which interpolates.

**Why optional and unnumbered.** It is a research question beside the main line (old
Phase 6 T3–T4, then Phase 8 T2–T3, then Phase 9 T5–T6, all until 2026-11-09). It is not
systems work, so it left Phase 9, and no phase waits for it: Phase 10 starts without it
(G8, amended 2026-11-09). It can run at any time after Phase 5, on the author's Mac.
Decided by the author (AHA-HH), 2026-11-09 (decision 30).

## Read first

- Design §6.1 (the option 2 row), §13.6 (confounders), §13.18; decisions 2, 10, 23, 24
  and 30.
- `docs/phase5/README.md` and `docs/phase5/results.md` (the baseline);
  `docs/phase8/README.md` ("Open items": the review leftovers S1, S3, R1 that P16 works
  around).
- `CLAUDE.md`: Working agreement, Checks, Burn, Commands.

## Inputs from earlier phases

| Item | Value | Source |
| --- | --- | --- |
| Baseline run | stage 3 (n = 129, K = 64), s = 64, p = 0, modes [12, 12], width 32, 4 layers, batch 20, 500 epochs, Adam 1e-3 / wd 1e-4, cosine | Phase 5 T7 |
| Baseline cost (cuda, A100) | 1.69 s/epoch, about 851 s per run; 2.45 s/epoch at s = 128 (the θ-map's spectral grid at n = 65 is 128²) | Phase 5 T5 |
| Ungated Chebyshev module | `chebyshev/transfer`, `chebyshev/clencurt` without features | Phase 4s T16 |

## Decisions

P8–P17, chosen by the author on 2026-11-09, all as recommended, and recorded in design
§12, decision 30 (they keep the numbers they had in Phase 9). Reasoning in
[T1-theta-map.md](T1-theta-map.md) ("Decisions"): the even extension as an `FNOConfig`
option (P8), once around the spectral block (P9), grid channels θ/π (P10), 12 modes with
~24 a possible T2 arm (P11), `PoissonConfig::cgl` in the existing loaders (P12), the
unweighted relative L² loss (P13), `theta` in the s argument (P14), `predict_poisson` and
`poisson_resolution` in scope (P15), the review leftovers worked around (P16), Metal for
acceptance if its parity case passes, else flex (P17).

## Tasks

| Task | Brief | Delivers | Depends on |
| --- | --- | --- | --- |
| T1 | [T1-theta-map.md](T1-theta-map.md) | The θ-map model (option 2 of §6) and its convention diff (`CONVENTION_VERSION` 2 → 3) | Phase 5 T7, Phase 4s T16 |
| T2 | written when T1 merges | θ-map against the Phase 5 baseline at 65², 129² and across resolutions, same seeds and test sets | T1 |

Task numbers are execution order (design §10). One task per branch and PR:
`optional/T<k>-<name>`. Each task is worked from its brief alone, plus the files the
brief lists under "Read first".

## Module layout

- `models/fno`, `data/loaders/poisson`, `metrics/poisson`, the train and predict
  examples, and CONVENTIONS with a version bump (T1): as set by T1's brief.

## Results

Filled in as tasks merge. The phase is done when every row has a value.

| Item | Value | Source |
| --- | --- | --- |
| θ-map model and convention diff | | T1 |
| θ-map against the Phase 5 baseline | | T2 |

## Exit checklist

- [ ] T1–T2 merged, or a task dropped with the reason recorded in design §12.
- [x] P8–P17 chosen by the author and recorded in design §12 (2026-11-09).
- [ ] Every row of Results filled in.
