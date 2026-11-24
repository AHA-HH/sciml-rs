# Optimisers CO3.1: Burn linear algebra for SOAP

Spike for `docs/optimisers/tasks/CO3.1.md`. Burn `=0.23.0-pre.2` with `linalg`; f32; release
builds; flex and metal on the author's M2 Pro Mac (2026-11-24); cuda on the VM: pending.

## What is measured (`src/bin/linalg_check.rs`)

For n ∈ {12, 32, 128, 144}, matrices built in f64 and rounded to f32 (seeded):

- **qr** (`burn::linalg::qr`, Householder in tensor ops, on the device) of SOAP's refresh
  input A = P Q₀, P symmetric PSD with eigenvalues 10^(−6 i/(n−1)) (κ = 1e6), Q₀ orthogonal.
  *backward* ‖A − QR‖_F / ‖A‖_F and *orth* max|QᵀQ − I| (both in f64); *fwd* max|Q − Q_ref|
  against an f64 host Householder QR of the same f32 matrix, columns sign-fixed by
  diag R > 0. The same three errors of an f32 host Householder QR are the baseline any f32
  QR meets: with κ = 1e6, Q's trailing columns are only determined to about κ·ε/gap, so
  *fwd* ≈ 1e-3 is the problem's, not Burn's.
- **svd** (`burn::linalg::svd`, 30 sweeps; runs on the host through `into_data`) of P (the
  first basis): σ error max|σᵢ − λᵢ| / λ₁ against a cyclic Jacobi eigensolver in f64 on the
  same f32 matrix; residual ‖P U − U diag(σ)‖_F / ‖P‖_F; orth max|UᵀU − I|.
- **time**: median of 20 calls after 3 warm-up calls, each read back with `into_data` (so
  it includes the upload of the input and the download of the outputs); *host qr* is the
  spike's own unblocked f32 Householder on the CPU, the "else host" option of D9.

## Results

flex (`cargo run -p linalg-spike --release --bin linalg_check`):

| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 2.3e-7 | 3.2e-7 | 2.3e-3 | 3.8e-7 / 3.1e-7 / 3.4e-4 | 0.62 ms | 0.01 ms | 1.2e-7 | 2.5e-7 | 1.0e-6 | 0.02 ms |
| 32 | 2.1e-7 | 3.5e-7 | 2.2e-3 | 3.9e-7 / 7.7e-7 / 1.9e-3 | 1.18 ms | 0.07 ms | 1.7e-7 | 5.9e-7 | 7.7e-7 | 0.10 ms |
| 128 | 4.1e-7 | 5.4e-7 | 2.9e-3 | 8.2e-7 / 1.1e-6 / 1.8e-3 | 4.26 ms | 3.19 ms | 6.7e-7 | 1.0e-6 | 1.8e-6 | 1.52 ms |
| 144 | 4.1e-7 | 5.3e-7 | 2.1e-3 | 9.2e-7 / 1.4e-6 / 2.4e-3 | 5.03 ms | 4.56 ms | 5.5e-7 | 9.0e-7 | 1.6e-6 | 2.10 ms |

metal (`--features metal`):

| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1.6e-7 | 4.2e-7 | 1.4e-3 | 3.8e-7 / 3.1e-7 / 3.4e-4 | 7.36 ms | 0.00 ms | 1.2e-7 | 2.5e-7 | 1.0e-6 | 0.13 ms |
| 32 | 1.7e-7 | 3.5e-7 | 7.7e-4 | 3.9e-7 / 7.7e-7 / 1.9e-3 | 20.04 ms | 0.05 ms | 1.7e-7 | 5.9e-7 | 7.7e-7 | 0.16 ms |
| 128 | 2.6e-7 | 4.1e-7 | 1.1e-3 | 8.2e-7 / 1.1e-6 / 1.8e-3 | 89.13 ms | 3.33 ms | 6.7e-7 | 1.0e-6 | 1.8e-6 | 1.70 ms |
| 144 | 2.1e-7 | 5.2e-7 | 1.2e-3 | 9.2e-7 / 1.4e-6 / 2.4e-3 | 92.60 ms | 4.84 ms | 5.5e-7 | 9.0e-7 | 1.6e-6 | 2.33 ms |

cuda (`--no-default-features --features cuda`, the VM): pending.

Reading:
- **Correct everywhere measured.** qr is backward stable and orthogonal to f32 level on both
  backends; its forward error is within 7× of the f32 baseline at n = 12 and equal or
  better above. svd's singular values are within 7e-7 · σ₁ of the f64 eigenvalues, with
  orthonormal U.
- **Device qr is slow on metal**: 7–93 ms, 20–30× the host at n ≥ 128 and far more below
  (many small kernels and a sync per Householder step). On flex Burn's qr is already host
  code and costs about what the plain host loop does at n ≥ 128.
- **svd** is host code on every backend: ≤ 2.4 ms at n = 144, once per tensor.

## Constraint 1 (`src/bin/fingerprint.rs`)

FNV-1a of the bits of a seeded `FNO<4>` forward, two Adam epochs of `train_fno` and a
three-epoch hybrid Gauss–Newton run (CG ≤ 5) on tiny 2D datasets, flex:

| build | forward | + Adam | + hybrid GN (final) |
|---|---|---|---|
| this branch, `linalg` on | `681d0cb622163b3d` | `22251275ef53a81e` | `84e63f28a21024db` |
| `origin/main` 4d681d0, `linalg` off | `681d0cb622163b3d` | `22251275ef53a81e` | `84e63f28a21024db` |

The `origin/main` row: a scratch worktree of `origin/main`, this spike's `Cargo.toml` and
`fingerprint.rs` copied in (not `linalg_check.rs`, which needs the feature) and
`spikes/linalg` added to its workspace members; then
`cargo run -p linalg-spike --release --bin fingerprint` in both checkouts.

## D9, proposed

The first basis by `svd` on the host (as recommended); the refreshes by **host** QR, not
device `qr`, at least on metal: Burn's device qr is correct but 20–30× slower there than a
host QR of the downloaded factor. The cuda row decides whether cuda keeps device `qr`;
unless it is within about 2× of host there, use the host path on every backend (one code
path).
