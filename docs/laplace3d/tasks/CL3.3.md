# CL3.3 3D loader

Started 2026-11-26 on `laplace3d/cl3-3`. Roadmap: "**CL3.3 3D loader.** The Poisson loader
for 3D splits, T_cu at load time." with the L1 gate's C3 and C4 and the L2 gate's Q2 folded
in. *Needs:* CL1.2, CL3.2, D5 (all met). Proposal: §10; §3.2 (storage 'ijk'), §6 (the
Poisson loader gets 3D shapes; new fields on new 3D types), §9 (error 2 per s in 3D; D2).

## Goal
Load the 3D Poisson splits of D5 as Darcy-shaped datasets on the uniform grid of any s,
with T_cu applied at load time, and measure error 2 of the n = 65 test split at s = 32 and
64 against the 2D floor at the same s.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
- Roadmap: "also error 2 of the n = 65 test split at s = 32 and 64, against the 2D floor at
  the same s (proposal §9; L1 gate C3)."
- First commit, no behaviour change (L1 gate C4): "`chebyshev/tensor.rs` for `apply_nd`,
  `as_matrix`, `kron`, the rank-2 view and one per-axis shape check; `transfer.rs`'s tests
  to `transfer/tests.rs`; "3D D1, D2" in the citations at `transfer.rs:189`,
  `clencurt.rs:130/147/166`, `metrics/poisson/mod.rs:160`; one doc sentence per `_nd`
  contract (any rank in `chebyshev`, 2D or 3D in `metrics`); checked by the D2 tests and
  the datagen suite (CL2.1 uses `apply_nd`)." The same commit fixes three docs in
  `pde/poisson` (L2 gate Q2): `IMAG_TOL`'s "D_xx or D_yy" to "an interior D² block"
  (`collocation.rs:25–28`), the module doc (`mod.rs:1`, 2D only), and the 115-column doc
  line (`collocation.rs:146`).
- Proposal §6: "the Poisson loader gets 3D shapes. New fields go on new 3D types, not on
  `PoissonConfig` […]". §9: "Transfers: error 2 (the floor) per s in 3D, against the 2D
  floor at the same s."

## Read first
`data/loaders/poisson/{mod,uniform,cheb}.rs` (the 2D loader, its validation helpers and
fixtures), `data/loaders/common.rs` (rank-generic normalisation), `chebyshev/transfer.rs`
(`Transfers::to_uniform_all_nd`, which gets its first production caller here),
`metrics/poisson/mod.rs` (`poisson_errors_nd`, error 2), `pde/poisson/dataset3d.rs` (the
file layout: `f`, `u` `[N, n, n, n]`, `x`, `y`, `z`), L1 gate findings 2–5.

## Changes
Commit 1 (no behaviour change):
- `chebyshev/tensor.rs` (new, private): `apply_nd` with its `apply_dyn`, `as_matrix`, `kron`,
  `rank2`, and `check_per_axis`, the one per-axis check (`apply_nd`'s and
  `check_weights`' panic texts kept verbatim, so their `should_panic` tests stay
  unedited). `apply_nd` re-exported from `chebyshev` as now.
- `chebyshev/transfer.rs` → `transfer.rs` plus `transfer/tests.rs` (tests moved unchanged).
- Citations "3D D1, D2"; one `_nd` sentence in `chebyshev/mod.rs` and in
  `metrics/poisson/mod.rs`.
- `pde/poisson`: the three doc fixes.

Then the loader, in `data/loaders/poisson/` only (no new config type: a 3D split needs
nothing `PoissonConfig` lacks, n, s, the counts and the optional subset; the rank comes from
the function, and a 3D run's saved config type is CL4.1's to settle):
- `mod.rs`: `check_nodes` takes the node fields (`AXES_2D`, `AXES_3D`) and `read_samples`
  the spatial rank, keeping the 2D messages byte for byte.
- `uniform.rs`: one private split reader for both ranks (`read_uniform_split`, with the
  node fields), through `Transfers::to_uniform_all_nd` (rank 3 equals `to_uniform_all` bit
  for bit), and `Rows::take` generic over the rank (a slice or a select of axis 0, no
  arithmetic). The 2D and 3D public loaders wrap `load_uniform` and `load_uniform_test`,
  which take the node fields (code review F1, F7).
- `uniform3d.rs` (new): `load_poisson3d_uniform` (inputs `[N, s, s, s, 1]`, targets
  `[N, s, s, s]`), `load_poisson3d_uniform_test` (with a run's normalisers), and the shared
  normalisation (CONVENTIONS §10).
- `cheb.rs`: `load_poisson3d_cheb_field`, `[count, n, n, n]`, untransferred; both public
  readers wrap one private `read_cheb_field` (code review F1).
- CLAUDE.md "Layout": the 3D loader and `chebyshev/tensor.rs`.

Conventions: none changed.

## Tests that prove it
- 3D shapes at s = n − 1 and another s; targets against the closed form of a polynomial of
  degree < n per axis on the uniform grid (T_cu exact, ≤ 1e-12); inputs exactly
  `norms.x.encode(to_uniform_all_nd(f))`, targets raw (CONVENTIONS §10), bit for bit;
  the test-only loader equal to the full loader's test split bit for bit; a subset equal
  to the selected rows of the whole split; `load_poisson3d_cheb_field` equal to the written
  `u` bit for bit; invalid files (wrong `z`, a 2D file, too few samples) give
  `LoadError::Invalid` with the shape in the message, and a missing `z` a
  `LoadError::Reader`, as a missing `x` or `y` does in 2D.
- 2D (D2): the 2D loader's whole suite unedited, including its exact error messages and
  the bit-for-bit `normalization_matches_darcy` against `Transfers::to_uniform_all`; the
  chebyshev D2 tests (`*_rank_two_is_bit_identical*`) and the datagen suite after the move.
- Error 2 (`#[ignore]`d, needs `datasets/poisson`): the n = 65 test splits, 2D and 3D,
  loaded at s = 32 and 64; `poisson_errors_nd` with the loader's targets as the prediction
  against the Chebyshev reader's u gives error 2 per sample; at s = 64 it must equal the
  sidecar's `error2` mean and max (recorded by the generator by the same transfers) to 1e-12
  relative. Prints the table for `docs/laplace3d/results/cl3-3.md`.

## Checks
`/verify` (fmt, the CI clippy jobs, `cargo clippy --all-targets`, `cargo test`, `cargo test
--features datagen`, `cargo doc`), and the ignored error-2 test in release:
`cargo test --release --lib <full path> -- --exact --ignored --nocapture`. No metal: host
ndarray only.

## VM runs
None (the n = 65 test splits are on the Mac).

## Constraints and risks
- Constraint 1 (D2): no arithmetic changed in a 2D path; the 2D split reader now calls
  `to_uniform_all_nd`, which is `to_uniform_all` bit for bit on 2D samples (CL1.2's D2
  test), and the private helpers change only in rank genericity. The 2D suite checks
  their outputs (`normalization_matches_darcy` against `to_uniform_all`, bit for bit) and
  messages exactly. The C4 move is a move: the D2 tests and the datagen suite
  (A3 uses `apply_nd`) check it.
- Memory: the 3D n = 65 train split is 4.4 GB (two fields of 2.2 GB); the measurement reads
  only the test splits (879 MB).

## Pending runs
None.

## Open questions
None.

## Notes
- `collocation.rs:146`, the L2 gate's "115-column doc line": 115 bytes but 87 characters
  (⊗, ⊕ and ⁻¹ are multi-byte), under the 100-column width; the gate measured bytes. Not
  edited; L2 gate Q2's item closed by this note.
