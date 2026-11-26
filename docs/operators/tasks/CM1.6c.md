# CM1.6c Fourier and non-uniform derivatives

Started 2026-11-26 on `operators/cm1-6c`. Roadmap: "CM1.6c Fourier and non-uniform
derivatives. `FourierDiff` without Fourier continuation (that option with CM2.4),
`non_uniform_fd`. *Done when:* each against fixtures. *Needs:* CM0.5." Proposal: §10; §1.4,
§2, §5, §6.4. D1, D2, D3 (as amended 2026-11-25), D8.

## Goal

2.0.0's spectral grid derivatives (`FourierDiff`, periodic) and its point-cloud first
derivatives (`non_uniform_fd`) in `losses/`, each matching 2.0.0 on committed fixtures
within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "each against fixtures".
- D3 (amended 2026-11-25): relative max error max |a − e| / max |e| ≤ 1e-5 in f32, ≤ 1e-12
  in f64, on the CPU; f32 parity suffices while Flex's f64 `rfft` uses f32 twiddles, each
  f64 test that goes through an FFT `#[ignore]`d with its measured error recorded.
  `non_uniform_fd` has no FFT: its f64 test runs.
- Proposal §5, output parity; §1.4, constraint 1: every existing FNO output bit for bit
  identical.

## Read first

- 2.0.0 `neuralop/losses/differentiation.py`: `FourierDiff`, `get_non_uniform_fd_weights`,
  `non_uniform_fd` (in `tools/parity/.venv`).
- `src/neural_operators/losses/differentiation.rs` (`FiniteDiff`, `GridDiff`, `MatrixDiff`).
- `tests/parity_losses_common/`, `tests/parity_common/`, `tools/parity/generate_losses.py`.

## Changes

- `losses/differentiation.rs`:
  - `FourierDiff<const D: usize>` with the public field `l: [f64; D]` (domain lengths; 2.0.0's
    default 2π is `[TAU; D]`). `partial(u, axis, order)` for any order ≥ 1 by `rfft` along
    the axis, the multiplier (i k)ᵖ with k = 2π m / L, m = 0..n/2, and `irfft` of length n.
    2.0.0 uses `rfft` in 1D and `fftn`, then `.real`, in 2D and 3D; for a derivative along one
    axis both equal the per-axis `rfft`/`irfft` (the other axes' transforms cancel, and the
    real part is the Hermitian projection), with one bin to fix: at the Nyquist bin of an
    even n an odd order contributes nothing to the real part, so its multiplier is set to 0
    (this also makes the result independent of how `irfft` treats that bin's imaginary
    part). `GridDiff` implemented (orders 1 and 2 through `partial`).
  - `dx`/`dy`/`dz`, `gradient`, `divergence`, `curl_2d`, `curl_3d` move from `FiniteDiff`'s
    inherent methods to default methods of `GridDiff`, so `FourierDiff` (and `MatrixDiff`)
    share them; their bodies are unchanged. `FiniteDiff` keeps `partial`, `partial_sum`,
    `laplacian`. Callers bring `GridDiff` into scope.
  - Not ported (matrix row): Fourier continuation (`use_fc`, CM2.4), `low_pass_filter_ratio`
    (its 2D/3D form zeroes a full-spectrum slice whose index ranges mix the axes), the mixed
    `derivative(order=(1, 1))` and `compute_multiple_derivatives` (per-axis `partial`s
    compose to the same result except at the double-Nyquist bin of two even axes),
    `partial(direction="x")` strings.
- `losses/non_uniform_fd.rs` (new): `NonUniformFd` (fields `num_neighbors`,
  `derivative_indices`, `radius`, `regularize`; `Default` at 2.0.0's defaults 5, `[0]`, none,
  false). `stencils(points)` (`points`: `Array2<f64>` (N, d)) gives `Stencils { indices (N, k),
  weights (N, j, k) }`, as 2.0.0's `get_non_uniform_fd_weights`, on the host in f64:
  k = min(max(num_neighbors, 3), N) nearest points (the point itself first, ties to the lower
  index), the radius mask keeping at least 3, and per point and derivative the minimum-norm
  solution of A w = b (A (d+1) × k: a row of ones and the coordinate offsets; b the unit vector
  of the derivative), as `torch.linalg.lstsq` (gelsy) gives for a full-row-rank A, by a QR of
  Aᵀ (modified Gram–Schmidt, reorthogonalised); with `regularize`, 2.0.0's
  (AᵀA + 1e-6 I) w = Aᵀ b by Cholesky. `Stencils::apply(values)`: `values` a Burn `Tensor<1>`
  [N], result `[j, N]`, gradients flowing to `values` (gather and a weighted sum);
  `derivatives(points, values)` is 2.0.0's `non_uniform_fd`, the two in one. Rank-deficient A
  (k < d + 1, or collinear neighbours) panics: 2.0.0 returns gelsy's pseudo-inverse solution
  there; `shortcut:` note.
- `losses/differentiation.rs` also: `partial_sum` and `laplacian` become `GridDiff` defaults
  (code-quality F1; `FiniteDiff`'s copies removed, `sobolev.rs` imports `GridDiff`).
- `losses/mod.rs`: the new module; doc line.
- `tools/parity/generate_losses.py`: fixtures `derivatives_{f32,f64}` (D8: well under
  1 MB): `FourierDiff` with L = (2, 3, 1.5) on seeded fields over grids 32 (1D), 16 × 15 (2D,
  one even and one odd axis), 8 × 6 × 5 (3D): orders 1, 2, 3 per axis, Laplacian, gradient,
  divergence, curl; and `non_uniform_fd` on seeded points: 2D (N = 40, derivatives x and y,
  default k), with `radius=0.2`, with `regularize_lstsq=True`; 3D (N = 30, derivatives x
  and z, `num_neighbors=7`). README rows.
- Conventions: none changed.

## Tests that prove it

- Parity, `tests/parity_losses.rs` (f32, 1e-5) and `tests/parity_losses_f64.rs` (f64, 1e-12;
  the Fourier part `#[ignore]`d with its error recorded, the non-uniform part run), through a
  new `run_derivatives` in `tests/parity_losses_common/`.
- The ridge case (`regularize_lstsq=True`): 2.0.0's system (AᵀA + 1e-6 I) has condition
  ≈ 5e6, and its own f32 result is 4.0e-2 from its f64 result on the same inputs, so the f32
  test checks against the f64 reference within D3 (measured 7.3e-8), and the f64 test within
  cond · ε ≈ 1e-9 (measured 6.4e-11). Added during implementation; a question for sign-off.
- Negative control: the 2D Fourier dx with the axes swapped fails (> 0.1).
- Unit tests (`differentiation.rs`): `FourierDiff` exact (f32 rounding) on a trigonometric
  polynomial below Nyquist, orders 1–3, odd and even n; the Nyquist mode of an even grid has
  zero odd derivatives and (−k²) for order 2; panics (axis ≥ D, order 0). (`non_uniform_fd.rs`):
  exact on an affine function (weights reproduce the gradient); the minimum-norm weights
  against the closed form Aᵀ(AAᵀ)⁻¹b; gradients flow to `values`; panics on k < d + 1.
- Existing tests: unchanged except `use GridDiff` where the moved methods are called.

## Checks

`/verify` (fmt, clippy ×4, `cargo test`, `cargo test --features datagen`, `cargo doc`, the
bit-identity dumps against `main`), `uv run python generate_losses.py --check`.

## VM runs

None.

## Constraints and risks

- Constraint 1: no file outside `losses/` (and the tests, fixtures and generator) changes;
  `FiniteDiff`'s arithmetic is unchanged (methods moved, not edited), and nothing on the FNO
  path uses `losses/differentiation`. The bit-identity dumps confirm it.
- Lanes: `training/` and `models/fno` untouched (CO1.2, CM1.4 run beside this).
- Risk: Flex's f32 Bluestein FFT on the non-power-of-two axes (15, 6, 5) against torch's;
  D3's 1e-5 is relative to the tensor's maximum, so it should hold.

## Pending runs

None.

## Open questions

1. The ridge case's bounds above (f32 against 2.0.0's f64 reference; f64 within 1e-9, not
   D3's 1e-12), as `lp.rel`'s 1e-8 in CM1.6a.
