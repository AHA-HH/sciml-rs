# T6: evaluation on the learning grid and after prolongation

Phase 5, task 6. Branch `phase5/T6-evaluate`. Depends on T5.

## Read first
- Design §13.3 (the three errors, the floor), §7 (errors 1–5), §12 decision 22 (error 4
  against ‖u‖).
- `src/neural_operators/metrics/poisson.rs` (`poisson_errors`, `check_error3`,
  `evaluate`), `examples/predict/poisson.rs`.
- The T1 report and the T5 run table.

## Goal
Report, for every T5 run, the coarse error, the fine-grid error and its transfer-only
floor, so model error and lost information are separated.

## Deliverables
1. **`poisson_errors` takes the run's `Transfers`** (T2) instead of building
   `Transfers::new(n)`; the `s + 1 == n` assert becomes a check against the transfers'
   (n, s). With them, error 2 is the transfer-only error P R u_f − u_f, error 3 the
   coarse error on G_l against R u_f, error 4 the fine-grid error, error 5 on û_f. No new
   error functions. `errors.json` keeps its keys; the run's s is added.
2. **`predict_poisson`** evaluates every T5 run on flex: the five errors, the
   cross-checks, the boundary verdict, and the warm inference time on the test split
   into `costs.json` (T4).
3. **Error 4 against ‖u‖** over the test set (decision 22), for the s = 128 main-arm runs
   and the padding control: a scatter plot and its correlation, to test whether high
   error 4 comes from small ‖u‖.
4. **Comparison table** `docs/phase5/results.md`: one row per (arm, s) with mean and
   spread (min–max) over seeds of each error, and the costs. It is Markdown because
   `.gitignore` covers `*.csv` and `*.png`. A small example (or a mode of
   `predict_poisson`) builds it from the run directories, so it is reproducible.
5. **Plots** (gnuplot, skipped with a warning if missing): fine-grid error and its floor
   against s; time per epoch against s; error against total training time. They are
   attached to the PR, not committed (`*.png` is ignored); the table is the record.
6. **D4 escalation:** where the spread over seeds at some s is comparable to the gap to
   a neighbouring s, train seeds 45 and 46 for those cells (as in T5) and say so.

## Tests
- `poisson_errors` with `Transfers::new(n)` gives today's numbers bit for bit (existing
  tests).
- At s ≠ n − 1, a synthetic prediction equal to R u gives error 3 = 0 and error 4 equal to
  error 2 (the floor identity of §13.3).

## Acceptance
- Every T5 run evaluated; the T6 Results rows filled in.
- Error 2 at s = 128 with T_cu matches Phase 4 T3 (2.53e-6 / 9.50e-6 on the test split).

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, and the
backend each evaluation ran on.

## Do not
- Choose the baseline (T7).
- Drop or rename existing `errors.json` keys.
