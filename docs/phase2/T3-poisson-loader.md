# T3: Poisson loader

Phase 2, task 3. Branch `phase2/T3-poisson-loader`. Depends on T1 (T_cu) and T2 (the
dataset format).

## Read first
- Design §5.2 (loader), §5.3 (stages), §6.2 (Chebyshev → uniform), §11 (the loader row),
  §12 decisions 2 and 13, and the "further choices" under §12's questions (transfers at
  load time, a separate Poisson loader).
- `docs/CONVENTIONS.md` §10 (normalisation, the `y_test` asymmetry) and §12.
- `src/neural_operators/data/loaders/darcy.rs` and its tests, the model to follow;
  `loaders/base_dataset.rs` (`DatasetConfig`, `HasBaseConfig`);
  `data/io/readers/npz.rs` (`NpzFileReader`); `data/io/errors.rs` (`LoadError`);
  `data/transforms/normalizers.rs` (`UnitGaussianNormalizer`).
- T1's `chebyshev::transfer` and T2's dataset layout (`datasets/README.md`, the
  sidecar).

## Goal
Turn T2's Chebyshev-grid files into the same `OperatorDataset`s the Darcy pipeline
produces, on the uniform grid of size s = n − 1, ready for an unchanged FNO.

## Deliverables
New ungated module `src/neural_operators/data/loaders/poisson.rs`, registered in
`loaders/mod.rs`:
1. **`PoissonConfig`** with `base: DatasetConfig` and `n`, the Chebyshev size.
   `s() = n − 1`. It implements `HasBaseConfig`.
2. **`load_poisson_uniform<T: HostFloat>(train_path, test_path, &PoissonConfig)`**,
   returning `(train, test, PoissonNormalizers)`, shaped like `load_darcy_uniform`:
   - Reads `f`, `u`, `x` and `y` from each `.npz` with `NpzFileReader`.
   - Validates, with each failure a `LoadError::Invalid`:
     - `f` and `u` are `[N, n, n]`;
     - `x` and `y` equal T1's closed-form CGL nodes of size n to 1e-14;
     - the file has at least the requested number of samples;
     - n_train ≥ 2.
   - Builds T_cu(n, n − 1) once (T1) and applies it to every f and u sample in f64
     (`apply`), before normalisation.
   - Fits `UnitGaussianNormalizer`s on the transferred train f and u, encodes x_train,
     y_train and x_test, and leaves y_test raw (CONVENTIONS §10), exactly as Darcy does.
   - Inputs `[N, s, s, 1]` (the normalised forcing only; the model appends the grid) and
     targets `[N, s, s]`, stored as `T` and rounded once at the end.
3. **`PoissonNormalizers { x, y }`**, named fields as in Darcy.
4. The module doc: the asymmetry, the transfer and its error (Phase 0 T3, T2's sidecar),
   and that the loader does not read the sidecar. Shapes and errors on every public
   item. A Layout line in `CLAUDE.md`.

If Darcy's private helpers (normalising, stacking into `OperatorDataset`) would otherwise
be duplicated, a shared helper may be moved into `loaders/` with no change in behaviour.
Say so in the PR, and make sure Darcy's tests still pass unchanged.

## Tests (`#[cfg(test)] mod tests`, ungated)
Write small synthetic `.npz` files (n = 9 or 17, a few samples) into a fresh directory
under `std::env::temp_dir()` with `ndarray-npy`; the tests need no dataset and no RLST.
- `shapes_match_uniform_grid`: inputs `[N, s, s, 1]` and targets `[N, s, s]` with
  s = n − 1, for train and test.
- `transfer_is_applied`: with f and u set to a polynomial of degree < n in each variable,
  the denormalised train targets and the raw y_test equal the polynomial on the uniform
  grid to 1e-12 relative, with `T` = f64.
- `normalizer_asymmetry_matches_darcy`: x_test is encoded with the train normaliser and
  y_test is raw, mirroring Darcy's tests.
- Error cases, each a `LoadError` and never a panic or a NaN: a missing file, nodes not
  CGL (uniform `x`), the wrong shape, too few samples, n_train < 2.

## Acceptance
- All tests pass under `cargo test`, which needs no `chebyshev` feature, and under
  `cargo test --features chebyshev`, on macOS and in both CI jobs.
- The stage 2 dataset from T2 loads locally. Record the load time (release) and the
  shapes in the Results row.
- The T3 Results rows in `docs/phase2/README.md` are filled in, and the Phase 2 exit
  checklist is ready to tick, apart from the CLAUDE.md pointer.

## Checks to run and report
All of `CLAUDE.md`'s CI checks and `cargo clippy --all-targets -- -D warnings`, then the
same clippy and test commands with `--features chebyshev`. Give the exact test filters and
check the "running N tests" lines. The loader runs on the host, but its datasets reach a
backend; run its tests once with `--features metal` and say whether that was done.

## Do not
- Change the Darcy loader's behaviour or its tests.
- Add training or evaluation code, or read the raw Chebyshev u for the §7 errors (Phase 3
  T2).
- Store or cache transferred fields on disk.
- Depend on the `chebyshev` feature or on RLST.
