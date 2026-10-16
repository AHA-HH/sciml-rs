# T19: typed sidecar, Poisson evaluation in the library, scoped allows

Phase 4s, task 19. Branch `phase4s/T19-sidecar-and-eval`. Depends on T8 and T10.
- Can run in parallel with T17 and T18, with small overlaps in the Poisson predict examples,
  so rebase.
- T22 comes after.

> **Refresh at start:**
> - Re-read `examples/predict/poisson.rs` and `poisson_resolution.rs` after T10/T13.
> - Check T8's shared `ErrorStats` and `TRANSFER_DEGREE` on `main`.
> - Check that T14 left the sidecar writer's JSON shape unchanged.

## Read first
- Review §2 (the "Evaluation policy" edge), §5 S5 ("Blanket allow"), §6 T4-5, T6-2, §7
  step D3.
- `pde/poisson/dataset.rs::write_split` (:291-322): it writes the sidecar untyped, with
  `serde_json::json!`.
- `examples/predict/poisson.rs`:
  - The local `#[allow(clippy::redundant_field_names)]` on `mod records` :47-49.
  - `SidecarError2 { max, mean, d, s }` :55-61, `Sidecar { n, count, error2 }` :64-69 and
    `PoissonErrorsRecord` :72-110, all `#[derive(Config)]`.
  - Evaluation policy :156-262: the worst sample (NaN-aware `position`), the sidecar
    cross-check (`ERROR2_TOL = 1e-12`, only if `count == n_test`), `check_error3`, the
    boundary verdict (`BOUNDARY_THRESHOLD = 0.10`), building the record, `errors.json`,
    and exiting 1 on a failed cross-check.
- `metrics/poisson.rs` (`poisson_errors`, `check_error3`, `ErrorStats`).
- `src/lib.rs:48`: a crate-wide `#[allow(clippy::redundant_field_names)]`, caused by Burn's
  `#[derive(Config)]` expansion (burn-derive `config/analyzer_struct.rs:196`).

## Goal
- The sidecar's two halves, the untyped writer and the typed reader, are kept in sync by
  hand, and the evaluation policy lives in an example.
- Make the format a single type, and the policy a tested library function.
- Remove the blanket allow, which CLAUDE.md forbids.

## Deliverables
1. **Test first:** check in a small sidecar fixture, copied from an existing
   `datasets/poisson/*.json` (or written by the current `write_split` in a test), and parse
   it with the new type.
2. **`SidecarRecord`**, typed and `#[derive(Config)]`, in the ungated
   `data::loaders::poisson` (or T22's catalogue module, if T22 has merged).
   - The generator serialises it and the example loads it.
   - Its JSON must stay byte-compatible: the same keys, nesting and number formatting
     (compare the fixture's bytes, or at least its parsed `serde_json::Value`).
   - Note: `serde_json` is behind the `chebyshev` feature. Using Burn's `Config` derive,
     which brings its own serialisation, keeps the record ungated. Check this.
3. **`metrics::poisson::evaluate(...) -> PoissonErrorsRecord`:**
   - It moves the worst sample, the error-2 cross-check, the `check_error3` cross-check and
     the boundary verdict out of the example.
   - The constants `ERROR2_TOL` and `BOUNDARY_THRESHOLD` move with it, documented.
   - `PoissonErrorsRecord` moves here and implements `Display`, which gives the printout
     `predict_poisson` produces today.
   - "Exit 1 on a failed cross-check" stays in the example, driven by a field or method on
     the record.
4. **Callers:** `predict_poisson` calls `evaluate`. `poisson_resolution` reuses it where it
   applies: at the trained n it runs `check_error3`; at other n there is no sidecar
   cross-check unless one exists.
5. **Scoped allows:**
   - Delete `lib.rs:48`.
   - Add `#[allow(clippy::redundant_field_names)]` on each `#[derive(Config)]` type that
     needs it, with a one-line reason citing the burn-derive expansion.
   - The 10 files with the derive: `metrics/poisson.rs`, `layers/spectral_convolution.rs`,
     `training/trainer.rs`, `models/fno.rs`, `data/transforms/normalizers.rs`,
     `data/loaders/{base_dataset,poisson,darcy,burgers}.rs`, `examples/predict/poisson.rs`.
     Expect fewer after earlier tasks.
   - Delete the example's local allow along with its records.

## Tests
- The sidecar fixture parse, plus a round trip `SidecarRecord` → JSON → fixture-equal.
- `evaluate` unit tests on synthetic arrays:
  - the worst sample is chosen correctly, including the NaN case
  - the boundary verdict on either side of the threshold
  - a cross-check mismatch is flagged
- The existing `metrics::poisson` tests pass.

## Acceptance
- `predict_poisson` on `runs/poisson_fno_p4_1791922918` prints the same text, with the
  same five errors and verdict, and writes an `errors.json` equal to the previous one
  (diff the files). Fill the Results row.
- `cargo clippy --all-targets -- -D warnings` passes, and also with `--features chebyshev`
  (or the renamed feature).
- `git grep -n 'allow(clippy::redundant_field_names)'` shows only per-type allows.

## Checks to run and report
The six from the README **plus the chebyshev pair**, because `dataset.rs`'s writer changes.
Give the line delta.

## Do not
- Change the sidecar JSON or the `errors.json` format.
- Split `loaders/poisson.rs` (T22).
