# T22: `loaders/poisson.rs` split; dataset catalogue

Phase 4s, task 22. Branch `phase4s/T22-poisson-loader-split`. Depends on T9 and T19. Can
run in parallel with T23. Shares `loaders/burgers.rs` and `darcy.rs` with it, so rebase.

> **Refresh at start:** re-read `loaders/poisson.rs` on `main`. It has been changed by T4,
> T8, T9, T14, T18 and T19. Also re-read the generator (`pde/poisson/dataset.rs`,
> `examples/generate/poisson.rs`).

## Read first
- Review §2 (the generator → loader "direction" edge), §6 T4-6, T5-3, T6-1, §7 Wave E3.
- `data/loaders/poisson.rs` at `6eb3212`, 748 lines:
  - `PoissonSet` :68-84, `STAGES` :90-94, `stage(n) -> Result<PoissonSet, LoadError>`
    :100-107
  - `load_poisson_cheb` :191-197 (an identity wrapper)
  - `load_poisson_cheb_field(path, field: &str, n, count)` :213-240 (checks "f" | "u" at
    runtime)
- **Who uses the catalogue:**
  - `PoissonSet`: the generator's library code `pde/poisson/dataset.rs:33, 251`
    (`split_stem`) and `predict/poisson_resolution.rs:23, 68`.
  - `STAGES`: one test, `dataset.rs:471`.
  - `stage()`: `predict/{poisson:130, poisson_plot:97, poisson_resolution:50}` and
    `train/poisson.rs:69`.
  - `examples/generate/poisson.rs` does not import the catalogue. It hard-codes its own
    `DATASETS` table (:28-47).
- The hard-coded `.mat` paths in the Burgers and Darcy train and predict examples (grep
  `.mat"`).

## Goal
One file mixes three concerns (dataset catalogue, Chebyshev reader, uniform loader), uses a
string-typed field selector, and has the gated generator reaching into an ungated loader for
experiment configuration. Separate them, type the selector, and make the catalogue the
lowest layer.

## Deliverables
1. **Split** into `data/loaders/poisson/{mod.rs, cheb.rs, uniform.rs}`, each under about
   250 production lines.
2. **A catalogue module**, e.g. `data/catalogue.rs` or `datasets.rs`, holding `PoissonSet`,
   `STAGES` and `stage()`.
   - `stage()` returns `Option<PoissonSet>` (or its own small error), since it loads
     nothing and shouldn't return `LoadError`.
   - `pde/poisson/dataset.rs` imports it from here, not from the loader.
   - The generator example's `DATASETS` table is built from the catalogue (stages plus the
     3e/4e sets), so file names come from one place. Keep the generated sets exactly the
     same, and compare the list of stems before and after.
3. **`enum PoissonField { F, U }`:**
   - It replaces `field: &str`.
   - `load_poisson_cheb_field` takes it; delete `load_poisson_cheb` and update its callers
     to pass `PoissonField::U`.
   - The test `cheb_field_loader_rejects_unknown_field` becomes impossible at compile time.
     Delete it as a deliberate removal.
4. **`BurgersFiles` / `DarcyFiles`** in the catalogue replace the hard-coded `.mat` paths in
   the examples, mirroring `PoissonSet`.
5. CLAUDE.md "Layout": update the `data/loaders/poisson` paragraph and its pointers.
6. **Optional, from the T8 review** (`docs/code-quality/thermo-nuclear-review-2026-10-16-phase4s-T8.md`,
   finding 2.1): add `Transfers::to_uniform_all(&self, f: ArrayView3<f64>) -> Array3<f64>`
   in `chebyshev::transfer`. It replaces the loader's `transfer_samples` (the uniform
   loader calls it on `field.slice(s![..count, .., ..])`) and the loop in the
   `metrics/poisson.rs` test helper `to_uniform`. Same numerics: it is the per-sample
   `to_uniform` loop, written once.

## Tests
- Every loader test passes, moved with its code.
- **Add** `generator_stems_unchanged`: the catalogue-derived generator list equals the
  hard-coded list it replaces.
- **Deliberate removal:** the unknown-field test.

## Acceptance
- `train_poisson`, `predict_poisson`, `poisson_plot` and `poisson_resolution` find their
  files. Running `predict_poisson` on the reference run gives the same errors.
- `cargo run --example generate_poisson --features <feature>` in dry-run form (if one
  exists), or a test, shows the same set of output files.

## Checks to run and report
The six from the README **plus the chebyshev pair** (or the renamed feature), because the
generator's imports change. Give the line delta.

## Do not
- Change file names on disk, the `.npz` keys or the stage table.
- Change the loading numerics.
