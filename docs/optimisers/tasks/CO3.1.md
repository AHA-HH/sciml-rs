# CO3.1 Burn linear algebra for SOAP

Started 2026-11-24 on `optimisers/co3-1`. Roadmap: "CO3.1 Burn linear algebra for SOAP.
Enable `linalg`; `qr` and `svd` at 12, 32, 128 (and 144) against an f64 host reference on
flex, metal and cuda: error and time." Proposal: §10; §1.4, §2 (Burn 0.23), §5.4 (D9).

## Goal
Enable Burn's `linalg` feature without changing any existing output, and measure whether
`linalg::qr` (device) and `linalg::svd` (host) are accurate and fast enough at SOAP's
factor sizes to settle D9.

## Acceptance (roadmap *Done when*, verbatim, and the proposal's matching text)
"the table; constraint 1 shown with the feature on; D9 proposed."
Proposal §5.4: "the first basis by `svd` on the host (sizes ≤ 128, once per tensor), the
refreshes by `qr` on the device if O3's first task shows it correct and fast on cuda, metal
and flex, else on the host. Precision f32, as the paper's optimiser; O3's first task checks
the refresh against an f64 host reference at our sizes." §1.4: "Enabling Burn's `linalg`
feature (Section 5.4) must not change any existing output either; O3's first task checks it."

## Read first
- `burn-linalg-0.23.0-pre.2/src/functions/{qr,svd}.rs`, `svd_host.rs` (svd runs on the host
  through `into_data`, f32 stays f32).
- `burn-0.23.0-pre.2/Cargo.toml`: `linalg = ["dep:burn-linalg", "burn-nn/linalg"]`;
  `burn-nn`'s `linalg` only adds `Initializer::Orthogonal` and two losses
  (`cosine_embedding`, `triplet_margin`), none used here.
- `spikes/metal_diag/` (the workspace-member spike pattern); `training/trainer.rs` tests
  and `training/fixtures.rs` (the tiny datasets the fingerprint reuses, copied).

## Changes
- `Cargo.toml`: `linalg` added to the workspace `burn` features (the pin unchanged);
  `spikes/linalg` added to the workspace members (not a default member). `Cargo.lock`:
  the spike package; `burn-linalg` is already in the lock.
- `spikes/linalg/` (package `linalg-spike`, never built by CI), with `REPORT.md`:
  - `linalg_check`: for n ∈ {12, 32, 128, 144}, f32 on the default device:
    - **qr** of SOAP's refresh input A = P Q₀ (P symmetric PSD with eigenvalues
      10^(−6 i/(n−1)), Q₀ orthogonal, both built in f64 and rounded to f32): backward error
      ‖A − QR‖_F / ‖A‖_F, orthogonality max|QᵀQ − I|, and the forward error of Q against an
      f64 host Householder QR of the same f32 matrix (columns sign-fixed by diag R > 0),
      beside the same errors of an f32 host Householder QR (the baseline any f32 QR meets);
    - **svd** of P itself (the first basis): max|σᵢ − λᵢ| / λ₁ against a cyclic Jacobi
      eigensolver in f64 on the same f32 matrix (rounding P to f32 alone moves its
      eigenvalues by about 6e-8, so the eigenvalues it was built from are not the
      reference), the residual ‖P U − U diag(σ)‖_F / ‖P‖_F, and max|UᵀU − I|
      (amended during the task, after the code-quality review);
    - time: median of 20 calls after 3 warm-up calls, each synchronised by reading the
      outputs back (`into_data`), release build; the host QR arm timed with the same
      round trip (A downloaded, Q and R uploaded and read back).
  - `fingerprint`: constraint 1. Seeded (`device.seed`), flex: an `FNO<4>` forward on a
    fixed input, two Adam epochs of `train_fno` and a three-epoch hybrid Gauss–Newton run
    on tiny 2D datasets; prints an FNV-1a hash of the output's, the final weights' and the
    metrics' bits.
- No change under `src/` (no `training/`), no convention change, no new crate.

## Tests that prove it
- Constraint 1: the whole existing suite with `linalg` on (`cargo test`, `cargo test
  --features datagen`), every test unedited; and `fingerprint`'s hashes equal on this
  branch (feature on) and on `origin/main` (feature off; the spike copied into a scratch
  worktree of `origin/main`, `linalg_check` removed there as it needs the feature).
- The table: its errors against the f64 host reference; "correct" means qr's errors within
  a small factor (10×) of the f32 host Householder baseline and svd's singular values and
  vectors at f32 level for the spectrum's conditioning. Timing is reported, never asserted.

## Checks
`/verify` (fmt, the three clippy lines, `cargo clippy --all-targets`, `cargo test`,
`cargo doc --no-deps`; `datagen` clippy and tests since the feature reaches every build;
`cargo tree -e normal` for new dependencies), and:
- `cargo clippy -p linalg-spike --all-targets -- -D warnings` (flex and `--features metal`)
- `cargo run -p linalg-spike --release --bin linalg_check` (flex) and `--features metal`
- `cargo run -p linalg-spike --release --bin fingerprint` here and on `origin/main`

## VM runs
`linalg_check` on cuda (`--no-default-features --features cuda`), about 5 min including the
build; decides whether D9's device `qr` holds on cuda. At the author's go.

## Constraints and risks
- Constraint 1 (§1.4): the feature reaches every build of the crate; shown as above.
- Burn's `qr` is Householder written in tensor ops: many small kernels at n = 128, so it
  may be correct but slow on GPU; that is what D9's "else host" covers.
- No effect on the 2D or 3D objectives' outputs beyond the shared constraint 1 check.

## Pending runs
- none. cuda `linalg_check` ran on the VM 2026-11-25 (A100, commit 080c317, exit 0), at the
  author's go, by the coordinating session; table in the report.

## Open questions
- none. D9 signed off 2026-11-25: host `svd` first basis, host QR refreshes, every backend.
