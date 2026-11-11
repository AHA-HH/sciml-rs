# T4: unused data generality removed

Phase 4s, task 4. Branch `phase4s/T4-dead-data-generality`. Depends on T1.
- Can run in parallel with T2, T3 and T5–T8.
- **Shared files:**
  - with T8: `data/loaders/poisson.rs`, `pde/poisson/dataset.rs` (a test import),
    `examples/predict/{poisson,poisson_plot}.rs`
  - with T3: the train example imports

  These collide only in `use` lines, so whichever merges second rebases.
- Do not run it beside T14: both edit `pde/poisson/dataset.rs`.

## Read first
- Review §5 S2 ("Normalizer", "File reading", "Dataset config access" rows), §6 T2-6, T2-7,
  T2-14, §7 step A3.
- `docs/CONVENTIONS.md` §10. It names `GaussianNormalizer` and `RangeNormalizer`, and lists
  their tests under "Verification".
- `data/transforms/normalizers.rs`, `data/io/traits.rs`, `data/io/readers/{npy,npz,mat}.rs`,
  `data/loaders/base_dataset.rs`.

## Goal
Delete abstractions that have one implementor or no callers. Fix two latent `npz` bugs on the
way.

## Deliverables
1. **Normalizers (`normalizers.rs`):**
   - Delete `GaussianNormalizer` (:120) and `RangeNormalizer` (:159, `with_range` :168),
     with their tests. Neither has a caller outside its own tests.
   - Delete the `Normalizer` trait (:12-21). `fit` / `encode` / `decode` become inherent
     methods on `UnitGaussianNormalizer`.
   - Rename `mean_ref` / `std_ref` / `eps_val` (:64-72) to `mean` / `std` / `eps`. Their
     only caller is `FlatDecoder::new` (:234-235).
   - Fix the doc at :60, which says "above" for methods that are below.
   - Remove the trait imports in `predict/{darcy_plot:18, poisson:28, poisson_plot:31,
     poisson_resolution:25}` and in the loaders `darcy.rs:21` and `poisson.rs:34`.
2. **Readers:**
   - Delete `data/io/readers/npy.rs` (`NpyFileReader` has no callers) and the
     `FieldReader` trait (`data/io/traits.rs`, and `traits.rs` itself if nothing else is in
     it).
   - `MatFileReader` and `NpzFileReader` get inherent `read_field` methods.
   - Migrate the callers: loaders `burgers.rs:12`, `darcy.rs:19`, `poisson.rs:32`, and the
     test in `pde/poisson/dataset.rs:329` (chebyshev-gated).
   - Fix the docs that mention Npy (`traits.rs:2`, `mat.rs:13`).
3. **`NpzFileReader` (npz.rs:26-51):**
   - Today the file is re-opened on every field read: 4 times per Poisson split, 3 times per
     Chebyshev read. Open it once (hold the `NpzReader`, or read all needed fields in one
     call).
   - The f64→f32 fallback turns any `by_name` error into `FieldNotFound`, so a field that is
     present with the wrong dtype is reported as missing. Report "present, wrong dtype" as a
     parse error, and keep `FieldNotFound` for genuinely missing fields.
4. **Base config (`loaders/base_dataset.rs`):**
   - Delete `BaseDatasetConfig` (:20), `HasBaseConfig` (:26) and the blanket impl (:30).
     Keep `DatasetConfig` (:13).
   - Callers read `cfg.base.n_train` / `cfg.base.n_test` directly:
     `predict/{burgers:20, darcy:22, poisson:25, poisson_plot:28, poisson_resolution:22,
     burgers_resolutions:22, burgers_superresolution:31}`.
5. **CONVENTIONS §10 prose:**
   - Remove the `GaussianNormalizer` / `RangeNormalizer` bullets and their test names from
     "Verification".
   - §10 is outside the version-bumping set (§9). Note the edit in the PR description.

## Tests
- **Deliberate deletions** (list them in the PR): `gaussian_accepts_one_sample_with_several_values`,
  `range_constant_points_encode_to_midpoint_and_round_trip`, the other Gaussian/Range tests
  and the npy tests.
- **Add:**
  - `npz_wrong_dtype_is_parse_error`: write a field as i32 with `NpzWriter` and check the
    error kind.
  - `npz_reads_several_fields_from_one_open`: behaviour test; the values match the
    per-field path.
- The remaining normalizer and loader tests pass unchanged, apart from renamed accessors.

## Acceptance
- Every loader returns the same arrays. The existing loader tests are the evidence.
- `git grep -nE 'Normalizer\b|FieldReader|HasBaseConfig|BaseDatasetConfig|NpyFileReader'`
  finds no stale uses. `UnitGaussianNormalizer` and `NormalizerRecord` remain.

## Checks to run and report
The six from the README, **plus the chebyshev pair**: this task edits a test in
`pde/poisson/dataset.rs`. Give the line delta and the before/after test counts.

## Do not
- Merge the Darcy/Poisson loader tails (T9), or move `FlatDecoder` to `training/` (optional,
  not in this phase).
- Change the `.npz` / `.mat` formats or any loader output.
