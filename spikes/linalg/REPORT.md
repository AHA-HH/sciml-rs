# Optimisers CO3.1: Burn linear algebra for SOAP

Spike for `docs/optimisers/tasks/CO3.1.md`. Burn `=0.23.0-pre.2` with `linalg`; f32; release
builds; flex and metal on the author's M2 Pro Mac (2026-11-24); cuda on the VM (one A100,
2026-11-25).

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
- **time**: median of 20 calls after 3 warm-up calls, each read back with `into_data`.
  *qr time* includes uploading A (in SOAP the factor is already on the device, so this
  slightly overstates it) and downloading Q and R; *host qr* is the "else host" option of
  D9 with the same round trip: A downloaded from the device, the spike's own unblocked f32
  Householder on the CPU, Q and R uploaded and read back.

## Results

flex (`cargo run -p linalg-spike --release --bin linalg_check`):

| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 2.3e-7 | 3.2e-7 | 2.3e-3 | 3.8e-7 / 3.1e-7 / 3.4e-4 | 0.49 ms | 0.01 ms | 1.2e-7 | 2.5e-7 | 1.0e-6 | 0.02 ms |
| 32 | 2.1e-7 | 3.5e-7 | 2.2e-3 | 3.9e-7 / 7.7e-7 / 1.9e-3 | 0.60 ms | 0.05 ms | 1.7e-7 | 5.9e-7 | 7.7e-7 | 0.06 ms |
| 128 | 4.1e-7 | 5.4e-7 | 2.9e-3 | 8.2e-7 / 1.1e-6 / 1.8e-3 | 4.10 ms | 3.29 ms | 6.7e-7 | 1.0e-6 | 1.8e-6 | 1.56 ms |
| 144 | 4.1e-7 | 5.3e-7 | 2.1e-3 | 9.2e-7 / 1.4e-6 / 2.4e-3 | 5.11 ms | 4.56 ms | 5.5e-7 | 9.0e-7 | 1.6e-6 | 2.08 ms |

metal (`--features metal`):

| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1.6e-7 | 4.2e-7 | 1.4e-3 | 3.8e-7 / 3.1e-7 / 3.4e-4 | 7.28 ms | 0.08 ms | 1.2e-7 | 2.5e-7 | 1.0e-6 | 0.10 ms |
| 32 | 1.7e-7 | 3.5e-7 | 7.7e-4 | 3.9e-7 / 7.7e-7 / 1.9e-3 | 19.49 ms | 0.13 ms | 1.7e-7 | 5.9e-7 | 7.7e-7 | 0.15 ms |
| 128 | 2.6e-7 | 4.1e-7 | 1.1e-3 | 8.2e-7 / 1.1e-6 / 1.8e-3 | 76.63 ms | 3.30 ms | 6.7e-7 | 1.0e-6 | 1.8e-6 | 1.65 ms |
| 144 | 2.1e-7 | 5.2e-7 | 1.2e-3 | 9.2e-7 / 1.4e-6 / 2.4e-3 | 84.83 ms | 4.80 ms | 5.5e-7 | 9.0e-7 | 1.6e-6 | 2.23 ms |

cuda (`--no-default-features --features cuda`; the VM, one A100, 2026-11-25, commit
080c317, run by the coordinating session at the author's go):

| n | qr backward | qr orth | qr fwd vs f64 | f32 host baseline (bwd / orth / fwd) | qr time | host qr time | svd σ err / σ₁ | svd residual | svd orth | svd time |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 1.6e-7 | 4.6e-7 | 7.9e-4 | 3.8e-7 / 3.1e-7 / 3.4e-4 | 78.69 ms | 0.83 ms | 1.2e-7 | 2.5e-7 | 1.0e-6 | 1.08 ms |
| 32 | 1.6e-7 | 2.0e-7 | 1.5e-3 | 3.9e-7 / 7.7e-7 / 1.9e-3 | 220.77 ms | 0.91 ms | 1.7e-7 | 5.9e-7 | 7.7e-7 | 0.29 ms |
| 128 | 1.9e-7 | 3.5e-7 | 1.0e-3 | 8.2e-7 / 1.1e-6 / 1.8e-3 | 1144.69 ms | 5.62 ms | 6.7e-7 | 1.0e-6 | 1.8e-6 | 3.47 ms |
| 144 | 2.1e-7 | 3.7e-7 | 1.3e-3 | 9.2e-7 / 1.4e-6 / 2.4e-3 | 1224.20 ms | 8.80 ms | 5.5e-7 | 9.0e-7 | 1.6e-6 | 4.64 ms |

Reading:
- **Correct everywhere measured.** qr is backward stable and orthogonal to f32 level on both
  backends (all three); its forward error is within 7× of the f32 baseline at n = 12 and within 1.6×
  above (better than the baseline on metal). svd's singular values are within 7e-7 · σ₁ of the f64 eigenvalues, with
  orthonormal U.
- **Device qr is slow on metal**: 7–85 ms, 18–23× the host round trip at n ≥ 128 and 60–150× below
  (lazy tensor ops, about 20 small kernels per column, n times). On flex Burn's qr runs on the CPU
  and costs 1.1–1.2× the plain host loop at n ≥ 128.
- **Device qr is slower still on cuda**: 79 ms to 1.2 s, 95–240× the host round trip at
  every size.
- **svd** is host code on every backend: ≤ 4.7 ms at n = 144, once per tensor.

## Constraint 1 (`src/bin/fingerprint.rs`)

FNV-1a of the bits of a seeded `FNO<4>` forward, two Adam epochs of `train_fno` and a
three-epoch hybrid Gauss–Newton run (CG ≤ 5) on tiny 2D datasets, flex:

| build | forward | + Adam | + hybrid GN (final) |
|---|---|---|---|
| this branch, `linalg` on | `681d0cb622163b3d` | `22251275ef53a81e` | `84e63f28a21024db` |
| `origin/main` 4d681d0, `linalg` off | `681d0cb622163b3d` | `22251275ef53a81e` | `84e63f28a21024db` |

Enabling the feature changes nothing the crate compiles against otherwise, on any backend
(checked in the Burn 0.23.0-pre.2 sources): `linalg = ["dep:burn-linalg", "burn-nn/linalg"]`;
the `burn-core` features `burn-linalg` turns on (`default`, `fusion`, `extension`, `std`)
are already on through `burn-signal`; `burn-nn/linalg` only adds `Initializer::Orthogonal`
and the `cosine_embedding` and `triplet_margin` losses, none used here. The new compiled
code is `burn-linalg` itself (and `gemm` under flex). The fingerprint checks this on flex.

The `origin/main` row: a scratch worktree of `origin/main`, this spike's `Cargo.toml` and
`fingerprint.rs` copied in (not `linalg_check.rs`, which needs the feature) and
`spikes/linalg` added to its workspace members; then
`cargo run -p linalg-spike --release --bin fingerprint` in both checkouts.

## D9, proposed

The first basis by `svd` on the host (as recommended); the refreshes by **host** QR on
every backend, not device `qr` (one code path). Burn's device qr is correct everywhere but
18–23× slower than a host QR of the downloaded factor on metal (n ≥ 128) and 95–240× on
cuda, round trip included; on flex the two cost the same.
