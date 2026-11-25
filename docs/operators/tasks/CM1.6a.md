# CM1.6a Finite differences and the data losses

Started 2026-11-25 on `operators/cm1-6a`. Roadmap: "CM1.6a Finite differences and the data
losses. `FiniteDiff` (first and second derivatives, periodic or one-sided boundaries,
1D–3D; gradient, Laplacian, divergence, curl), `H1Loss`, `HdivLoss`,
`PointwiseQuantileLoss`, `MSELoss`; the relative `LpLoss` against 2.0.0's. *Done when:* each
against 2.0.0 fixtures within D3 (f32; f64 too, no FFT involved). *Needs:* CM0.5." Split
from CM1.6 on this branch (2026-11-25). Proposal: §10; §1.2, §1.4, §2, §5, §6.4. D1, D2,
D3 (as amended 2026-11-25), D8.

## Goal

2.0.0's grid finite differences and its four data losses (H1, Hdiv, pointwise quantile,
MSE) in `losses/`, each matching 2.0.0's output on committed fixtures within D3.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)

- Roadmap: "each against 2.0.0 fixtures within D3 (f32; f64 too, no FFT involved)".
- D3 (amended 2026-11-25): relative max error max |a − e| / max |e| ≤ 1e-5 in f32,
  ≤ 1e-12 in f64, on the CPU. These ops use no FFT, so the f64 tests run (not `#[ignore]`d).
- Proposal §5, output parity: the same inputs give the same outputs within D3.
- Proposal §1.4, constraint 1: every existing FNO output bit for bit identical.

## Read first

- 2.0.0: `neuralop/losses/data_losses.py` (`LpLoss`, `H1Loss`, `HdivLoss`,
  `PointwiseQuantileLoss`, `MSELoss`), `neuralop/losses/differentiation.py`
  (`FiniteDiff`, `central_diff_{1,2,3}d`), in `tools/parity/.venv` after `uv sync --locked`.
- `src/neural_operators/losses/data_losses.rs` (`LpLoss`, `Reduction`), CONVENTIONS §11.
- `tests/parity_common/mod.rs` (`Fixture`, `assert_parity`, `Precision::d3`).
- `tools/parity/generate.py`, `README.md` (fixture format, `--check`).

## Changes

- `src/neural_operators/losses/differentiation.rs` (new): `FiniteDiff<const D: usize>`
  with `h: [f64; D]` and `periodic: [bool; D]`; `partial(u, axis, order)` for order 1 or 2
  on the spatial axis `axis` (0 = x, the first of the last D axes of `u`); `dx`, `dy`, `dz`;
  `partial_sum` and `laplacian`; `gradient` (components stacked at axis −D−1);
  `divergence` (components at axis −D−1); `curl_2d` (scalar) and `curl_3d` (vector); the
  public fields `h` and `periodic`, no constructor. One rank-generic stencil along an axis
  replaces 2.0.0's eighteen per-dimension methods: periodic by `roll_dim`, otherwise
  central inside and 2.0.0's one-sided third-order (first derivative) and second-order
  four-point (second derivative) formulas at the ends. D is any positive number for the
  axis operators (2.0.0 stops at 3); curl is D = 2 or 3. The deprecated
  `central_diff_{1,2,3}d` are `FiniteDiff::partial` over each axis and get no function of
  their own.
- `src/neural_operators/losses/sobolev.rs` (new): `SobolevLoss<const D: usize>` with
  `norm: SobolevNorm::{H1, Hdiv}`, `measure: [f64; D]`, `periodic: [bool; D]`, `reduction`,
  `eps`; constructors `h1(reduction)` and `hdiv(reduction)` at 2.0.0's defaults (measure 1,
  periodic, eps 1e-8); `rel(x, y)` and `abs(x, y)` on `[.., s_1, .., s_D]`, reduced over
  every leading axis (batch and channel, as 2.0.0's `reduce_all`). As 2.0.0: quadrature
  and finite-difference step h_a = measure_a / s_a; rel = √Σ / (√Σ_y + eps). The root has
  the zero subgradient at an exact match, as `LpLoss` (CONVENTIONS §11).
- `src/neural_operators/losses/pointwise.rs` (new): `PointwiseQuantileLoss` (`alpha`,
  `reduction`, `eps` 1e-7) and `mse_loss` (mean over every non-batch axis, summed over the
  batch).
- `src/neural_operators/losses/mod.rs`: the three modules.
- `tools/parity/generate_losses.py` (new, beside `generate.py`, which CM1.1 edits):
  fixtures `tests/fixtures/parity/losses{1,2,3}d_{f32,f64}.{npz,json}`; `--check` as
  `generate.py`'s. README section.
- Not ported (each in the matrix row): the per-call `quadrature` argument (it equals
  `measure = quadrature · s`), `take_root=False`, `MSELoss`'s `dim`; `gradient` in 1D keeps
  its component axis (2.0.0's `FiniteDiff.gradient` drops it in 1D only).
- `LpLoss` is not edited (constraint 1). Its `rel` is compared against 2.0.0's `LpLoss.rel`,
  which adds eps = 1e-8 to the denominator: inside D3 in f32, but in f64 a gap of about
  1e-8 / ‖y‖ (measured 6e-10 to 1.9e-9), so `lp.rel` is checked against 1e-8 and the gap
  recorded. Its `abs` keeps CONVENTIONS §11's h_a = 1/(s_a − 1); the row stays **partial**.
- Conventions: none changed. The new losses follow 2.0.0's quadrature (measure / s), not
  §11's `LpLoss` weight; whether §11 should record them is a question in the PR.

## Tests that prove it

- Parity (`tests/parity_losses.rs` in f32, 1e-5; `tests/parity_losses_f64.rs` in f64,
  1e-12, its own binary because Flex's float dtype is set process-wide; the shared runner
  `tests/parity_losses_common/`), for D = 1, 2, 3, on
  2.0.0's outputs for one seeded pair `x`, `y` of shape (2, 2, grid), grids 32, 16×12,
  8×6×5 (non-square so an axis mix-up fails), and a vector field `v` (2, D, grid):
  - `FiniteDiff` with h = (0.1, 0.2, 0.3): first and second derivative on every axis,
    periodic and one-sided; Laplacian, gradient, divergence; curl for D = 2, 3.
  - `H1Loss` and `HdivLoss`, `rel` and `abs`, measure (1, 2, 0.5): all periodic, none
    periodic, and (D ≥ 2) mixed; reduction `sum`, and `mean` for H1 `rel`.
  - `PointwiseQuantileLoss(alpha=0.1)`, sum and mean; `MSELoss`.
  - `LpLoss(d, p=2).rel` against the crate's `LpLoss::rel` on the flattened pair.
- Unit tests (closed forms, `differentiation.rs`): the periodic first and second
  derivatives of sin on a periodic grid within O(h²) bounds; the one-sided boundary rows
  exact on a cubic (first) and quadratic (second) polynomial; panics for a non-periodic
  axis under 4 points, an axis ≥ D, order 3, and curl in 1D.
- Unit tests (`sobolev.rs`, `pointwise.rs`): zero loss and zero (not NaN) gradient at an
  exact match; H1 `abs` of a known pair by hand.
- A wrong axis order fails parity (the 2D test swaps dx and dy as a negative control).

## Checks

`/verify` (fmt, clippy default / `--examples` / `--all-targets` / `--features datagen
--all-targets`, `cargo test`, `cargo test --features datagen`, `cargo doc --no-deps`; the
bit-identity dumps against `main`), and `uv run python generate_losses.py --check` in
`tools/parity/`.

## VM runs

None.

## Constraints and risks

- Constraint 1: no existing source file changes except `losses/mod.rs` (three `pub mod`
  lines); `LpLoss`, the trainer and the FNO are untouched, so FNO outputs cannot change.
  `/verify`'s bit-identity check confirms it.
- `training/` is not touched (the optimisers lane, CO1.1, owns it now).
- Coordination with CM1.1: own fixture names (`losses*`), own generator file, own test
  helper (`tests/parity_losses_common/`, `parity_common/` unchanged); README and
  `progress.md` conflicts resolved by keeping both.
- Risk: 2.0.0's one-sided formulas need four points per non-periodic axis; the fixtures'
  smallest axis is 5.

## Pending runs

None.

## Open questions

1. CONVENTIONS §11 covers only `LpLoss`. Should it record the 2.0.0 losses' quadrature
   (h_a = measure_a / s_a, also the finite-difference step) and `rel`'s eps? §11 lies
   outside §1–§6, so no `CONVENTION_VERSION` bump. Proposed: yes, as a short paragraph in
   a follow-up docs PR.
