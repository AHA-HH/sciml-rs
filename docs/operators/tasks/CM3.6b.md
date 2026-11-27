# CM3.6b Tensor GaLore projector, complex tensors

Started 2026-11-27 on `operators/cm3-6b` (the author asked for it that day). Roadmap:
"CM3.6b Complex tensors. 2.0.0's complex Tucker (complex SVD, phase sign rule) for the
spectral weights. *Done when:* against complex fixtures. *Needs:* CM3.6a." Proposal: §10;
§1.4, §2, §5, §6.4. D1, D2, D3 (as amended 2026-11-25), D8.

## Goal

CM3.6a's `TensorGaLoreProjector` for complex tensors (2.0.0's target, the spectral
weights): tensorly's Tucker HOOI with complex SVDs, conjugate-transpose mode products and
tensorly's complex sign rule, matching 2.0.0 on complex fixtures within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- "against complex fixtures": the ranks, the factors, `project`, `project_back` and the
  second `project` against 2.0.0 on `cfloat` gradients within D3 (≤ 1e-5, f32, CPU).
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical. No FNO
  code path is touched (`optim/galore.rs` only).

## 2.0.0's complex behaviour, reproduced

- `get_projection_tensor` casts a complex gradient other than `cfloat` to `cfloat`, so a
  `cdouble` gradient gets `cfloat` factors and `project` then fails in `multi_mode_dot`
  (dtype mismatch; checked). 2.0.0 supports complex64 only. Here a complex f64 gradient keeps
  f64 factors; the f64 fixture is tensorly's `tucker` on the `cdouble` gradient with
  2.0.0's `transform` / `inverse_transform` (2.0.0 minus the cast).
- tensorly's mode products with `transpose=True` use the conjugate transpose: `project` is
  G ×ₖ Fₖᴴ, `project_back` scale · (Y ×ₖ Fₖ); the HOOI's sweeps likewise; norms are
  √Σ|x|².
- The sign rule: `svd_flip` with `tl.sign` = `torch.sgn` multiplies column j by
  sgn(u_ij) = u_ij / |u_ij|, i the entry of largest modulus (the first on a tie). This does
  not normalise the phase: it doubles it, so the result depends on the phase the SVD
  returned. torch's CPU SVD (LAPACK `gesdd`) fixes that phase structurally: for an m × n
  matrix with m ≥ n, the first row of V is real; with m < n, the first row of U is real
  (the bidiagonalisation's reflectors leave that row alone), up to a sign that the rule
  then cancels (sgn(−u) · (−u) = sgn(u) · u). Reproduced: from any complex SVD, rotate
  column j so that v_j[0] = (Aᴴ u_j)₀ / σ_j (m ≥ n) or u_j[0] (m < n) is real, then apply
  the rule. Checked in torch against `torch.linalg.svd` + `svd_flip` on wide, tall, square,
  QR-path and LQ-path shapes: equal to 1e-7 (complex64) and 1e-15 (complex128).
- The complex SVD: Burn has none. The real embedding E = [[B, −C], [C, B]] of M = B + iC
  has each singular value of M twice; Burn's `svd` sorts them, and column 2j of E's U,
  [x; y], gives a unit left singular vector x + iy of M (phase arbitrary, then fixed as
  above). Degenerate singular values of M leave the factors non-unique, in tensorly too.

## Changes

- `src/neural_operators/optim/galore.rs`: `ComplexTensor<D>` (`re`, `im`); the projector and
  `tucker_factors` generic over the gradient's field through a sealed trait
  `GaLoreTensor<D>` (implemented for `Tensor<D>`, factors `Tensor<2>`, and for
  `ComplexTensor<D>`, factors `ComplexTensor<2>`), so the HOOI and the projector exist once;
  `TensorGaLoreConfig::init` generic in the same way (`init::<D, Tensor<D>>` or inferred).
  The real path's arithmetic is unchanged (CM3.6a's parity tests still pass, same
  tolerances). Unit tests in `optim/galore/tests.rs`.
- `tools/parity/generate_galore.py`: complex cases, fixtures
  `tests/fixtures/parity/galore_complex_{f32,f64}.npz` (complex arrays as `view_as_real`,
  as the other fixtures); the real fixtures unchanged (`--check`). README row.
- `tests/parity_galore_common/`: the complex run; `tests/parity_galore.rs` and
  `tests/parity_galore_f64.rs`: the complex cases.
- CLAUDE.md Layout line, feature matrix row, roadmap tick, progress entry.
- Conventions: none.

## Tests that prove it

- Parity (fixtures, D3): the three CM3.6a cases with complex gradients (`matrix` 12 × 7,
  rank 3; `tensor4` 6 × 5 × 4 × 3, ranks [3, 2, 3, 2], a spectral weight's shape;
  `fraction` 8 × 6 × 5, 0.5), each structured (a complex Tucker tensor plus noise): ranks,
  factors, `project(grad, 0)`, `project_back`, `project(grad2, gap)`. f32 against 2.0.0
  (1e-5); f64 against tensorly with 2.0.0's transforms (1e-12).
- Control: factors with the phase left as the embedding's SVD gives it (no LAPACK rotation)
  fail parity (shows the rotation is what matches); another gradient's factors fail.
- Unit: the complex SVD (U Σ Vᴴ reconstructs M; Uᴴ U = I); the phase rule on a hand-built
  matrix (wide and tall); orthonormal complex factors; full ranks make
  `project_back ∘ project` the identity; an exact complex Tucker tensor is reproduced; a
  complex tensor with zero imaginary part gives the real path's factors.

## Checks

`/verify` (fmt, clippy default/`--examples`/`--all-targets`/datagen, `cargo test`,
`cargo test --features datagen`, `cargo doc --no-deps`); `uv run python generate_galore.py
--check`; `generate.py` / `generate_losses.py --check` unchanged. Constraint 1: only
`optim/galore.rs` changes under `src/`; nothing in `models/` or `training/` imports it.
`CARGO_TARGET_DIR=~/sciml-rs-target/cm3-6b`.

## VM runs

None.

## Constraints and risks

- Constraint 1: holds by construction.
- The reproduction rests on `gesdd`'s structural phase; a different torch LAPACK
  (e.g. MKL on Linux) follows the same bidiagonalisation, and the f32 fixture records the
  platform in its sidecar.
- The embedding doubles each SVD's size (4× the entries); the spectral weights' unfoldings
  are small enough for the host SVD, as for CM3.6a.

## Pending runs

None.

## Open questions

None yet.
