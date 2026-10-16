# Thermo-nuclear code quality review: `phase4s/T4`

| | |
|---|---|
| Repo | sciml-rs |
| Branch | `phase4s/T4` (Phase 4s T4, unused data generality removed) |
| Base | `79f9c62` (merge-base with `main`) |
| Date | 2026-10-16 |
| Diff | 22 files, +225 / −494 (net −269), uncommitted working tree |

## Verdict: APPROVE

The branch is almost all deletion, and the deletions are correct. It removes three traits
that each had a single implementor (`Normalizer`, `FieldReader`, `HasBaseConfig` +
`BaseDatasetConfig` + a blanket impl), two normalizers with no callers, and the Npy reader.
Nothing new replaces them. Callers now use inherent methods or read the `config.base`
field directly. This is the code-judo the brief asked for: whole concepts are gone, not
moved.

The only new logic is in `NpzFileReader`, and it gets simpler. It keeps one open handle,
checks a field exists by name before reading it, and keeps the f32 fallback to one match
arm. That fixes both latent bugs without adding branches anywhere else.

No file crosses 1k lines, and nothing adds special-case branching to a shared flow. The
remaining issues (below) are pre-existing or belong to tasks already scheduled (T9, T22,
T24). One small, worthwhile tightening sits in the same diff: `read_samples` should
return `Array3`.

## Presumptive blockers

None met.
- **Incidental complexity kept despite a visible code-judo move:** no. The biggest move
  (finding 2.1) is mostly pre-existing structure, and the plan already gives it to T22.
- **Files over 1k lines:** none.
- **Ad-hoc branching:** none. The only new conditional is the dtype match in
  `npz.rs:60-68`, and it replaces an error-swallowing fallback.
- **Scattered feature checks, new wrappers, casts or helper duplication:** none.

## File sizes

| File | Before → after |
|---|---|
| `src/neural_operators/data/loaders/poisson.rs` | 748 → 742 |
| `src/neural_operators/data/transforms/normalizers.rs` | 507 → 318 |
| `src/neural_operators/pde/poisson/dataset.rs` | 476 → 476 |
| `docs/CONVENTIONS.md` | 334 → 329 |
| `src/neural_operators/data/loaders/darcy.rs` | 312 → 306 |
| `examples/predict/poisson.rs` | 271 → 268 |
| `src/neural_operators/data/io/readers/mat.rs` | 226 → 223 |
| `src/neural_operators/data/loaders/burgers.rs` | 217 → 208 |
| `examples/predict/poisson_plot.rs` | 197 → 194 |
| `src/neural_operators/data/io/readers/npz.rs` | 76 → 146 (+2 tests, +docs) |
| `examples/predict/poisson_resolution.rs` | 122 → 118 |
| `examples/predict/burgers_superresolution.rs` | 100 → 100 |
| `examples/predict/darcy_plot.rs` | 97 → 97 |
| `examples/predict/burgers_resolutions.rs` | 85 → 85 |
| `examples/predict/darcy.rs` | 80 → 77 |
| `examples/predict/burgers.rs` | 71 → 68 |
| `src/neural_operators/data/io/errors.rs` | 66 → 66 |
| `src/neural_operators/data/io/readers/npy.rs` | 65 → deleted |
| `src/neural_operators/data/loaders/base_dataset.rs` | 37 → 15 |
| `src/neural_operators/data/io/traits.rs` | 15 → deleted |
| `src/neural_operators/data/io/mod.rs` | 8 → 6 |
| `src/neural_operators/data/io/readers/mod.rs` | 7 → 6 |

No threshold is crossed. `npz.rs` grows only through tests and documentation; the
production code stays about the same size.

## Findings

### 1. Structural regressions
None. Every edit at a call site is an import removal or a method call replaced by a field
read.

### 2. Missed simplifications (code-judo)

**2.1 The Poisson loader passes an open file plus its validation context through free
functions** (`loaders/poisson.rs:158-166`, `:228-233`, `:238`, `:270`)

Now that `NpzFileReader` is a stateful open handle, the loader threads
`(&mut reader, split, n)` through `check_nodes` and `read_samples` on every call.
`load_split` and `load_poisson_cheb_field` both repeat the same three steps: open, check
nodes, read samples.

The real invariant is that "a field is read only after `x`/`y` are validated", but the
code only enforces it by call order. Encode it in a type instead:

```rust
/// One Poisson split file whose `x`, `y` have been checked against the CGL nodes.
struct CglFile { reader: NpzFileReader, label: String, n: usize }
impl CglFile {
    fn open(path: &Path, label: &str, n: usize, nodes: &Array1<f64>) -> Result<Self, LoadError>;
    fn samples(&mut self, name: &str, count: usize) -> Result<Array3<f64>, LoadError>;
}
```

Then:
- `load_split` becomes `let mut file = CglFile::open(..)?; (transfer(file.samples("f", c)?), ...)`.
- `load_poisson_cheb_field` becomes `CglFile::open(..)?.samples(field, count)`.
- The `split`/`n` parameters disappear from both helpers.

This is a decomposition of `poisson.rs`, and T22 already splits that file into
`poisson/{set, cheb, uniform}.rs`. T22 should take this with it rather than redoing it
here. It doesn't block this branch, because the brief's "Do not" keeps loader
restructuring out of T4.

**2.2 `read_samples` checks rank 3 but returns `ArrayD`** (`loaders/poisson.rs:270-291`). *Applied in this PR.*

It validates `ndim() == 3`, and then both consumers undo the type erasure with
`.into_dimensionality::<Ix3>().expect(..)` (`:231-232` and `:298-300`). That's two panics
guarding an invariant the callee has already established.

Fix: return `Array3<f64>` from `read_samples`, doing the conversion once and mapping its
error to the existing `Invalid` message. Both `expect`s go, and `transfer_samples` takes
`ArrayView3`. It's a small, local change to a function this diff already touches, and the
best "make the boundary explicit" fix available here. It's optional for this PR and
natural for T22.

### 3. Spaghetti or branching growth
None. The dtype fallback in `npz.rs:60-68` is a single match on the one error variant
that means "wrong element type". It replaces `Err(_) => try again` and is narrower and
easier to read.

### 4. Boundary and type contracts

**4.1 `ReaderError::FileNotFound`'s doc no longer matches either reader** (*applied in this PR*)
(`io/errors.rs:10`, `readers/npz.rs:32`, `readers/mat.rs:22`)

The variant's doc says "does not exist or could not be opened". Both readers send a
failed `File::open` to `ParseError`, and the new `NpzFileReader::new` doc says so
explicitly. Pick one meaning:
- Preferred: narrow the variant doc to "does not exist", which matches both readers.
- Alternatively: map `File::open` errors to `FileNotFound` in both readers.

This is a one-line doc fix and fits this branch, since it already rewrote the io docs.

**4.2 The error for "neither f64 nor f32" reports only the f32 attempt**
(`readers/npz.rs:62-66`)

For an `i32` field, the user sees the f32 attempt's `WrongDescriptor`, so the message
reads as though f32 was the expected type. A message that states the contract, such as
`field 'a' in <path>: expected f64 or f32, got <descr>`, makes the new error useful
rather than merely correctly classified. Low effort; optional.

**4.3 `read_field` takes `&mut self` on `NpzFileReader` but `&self` on `MatFileReader`**

This asymmetry is acceptable and honest. `by_name` really mutates the zip cursor, and no
trait is left that needs the signatures to match. A `RefCell` to fake `&self` would be
the wrong fix. No action needed; recorded so nobody "fixes" it later.

### 5. File size and decomposition
`poisson.rs` (742 lines) is the largest touched file. It isn't growing, and T22 splits
it. No action here.

### 6. Modularity
- **`FlatDecoder` sits in `data/transforms` but serves only training**
  (`normalizers.rs:116`). The brief rules a move out of scope ("optional, not in this
  phase"); noted for completeness.
- **Darcy and Poisson both carry the same `n_train < 2` guard and normalise-split
  tail** (`darcy.rs:78`, `poisson.rs:141`). This is T9's job; this diff correctly leaves
  it alone.

### 7. Legibility

**7.1 Burgers repeats `config.base.n_train` / `config.base.n_test` nine times**
(`loaders/burgers.rs:83-131`)

This is the mechanical result of deleting the accessor trait. Bind them once at the top
of `burgers_from_fields` (`let DatasetConfig { n_train, n_test } = config.base.clone();`,
or a pair of `let`s) and the body reads as it did before. Darcy and Poisson use the field
only two to four times, so they're fine as they are. T9 rewrites the Burgers reshapes
anyway, so do this there if not here.

**7.2 Two public `UnitGaussianNormalizer` methods have no docs**
(`normalizers.rs:88`, `:97`)

`to_record` and `from_record` are undocumented, and `from_record` can panic (`expect`)
without saying so. That breaks CLAUDE.md's "document every public item; say the panics".
It's pre-existing, and the file is now small enough to fix in two lines. It's also
covered by T24.

## Next steps

Each step preserves behaviour, can land on its own, and is ordered by leverage.

1. **Optional, in this PR: tighten the boundaries and docs.** Covers 2.2, 4.1, 4.2, 7.1
   and 7.2. 2.2 and 4.1 are applied; 4.2, 7.1 and 7.2 remain.
   - `read_samples` returns `Array3`.
   - Narrow the `FileNotFound` doc.
   - Make the dtype error say "expected f64 or f32".
   - Bind `n_train`/`n_test` once in Burgers.
   - Document `to_record`/`from_record`.

   Verify with `cargo test` and `cargo test --features chebyshev`; the loader tests must
   pass unchanged. Also run `cargo clippy --all-targets -- -D warnings` and
   `cargo doc --no-deps`.
2. **T9: shared loader tail** (already planned; depends on T4). Merge the
   `n_train < 2` guard and the normalise-split tail; Burgers' reshapes go there. Verify
   with T9's "test added first" Darcy checksum.
3. **T22: Poisson loader split** (already planned; depends on T9 and T19). Introduce the
   `CglFile` type from 2.1 as part of the `poisson/cheb.rs` split, so the
   validate-then-read order is enforced by the type and both public readers become two
   lines each. Verify with the existing `load_poisson_*` tests, unchanged, plus
   `predict_poisson` on an existing run printing the same errors.
4. **T24: docs pass.** Pick up any public items left undocumented after step 1.
