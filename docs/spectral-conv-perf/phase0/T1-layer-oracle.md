# spectral-conv-perf phase 0 / T1 - f64 oracle of `SpectralConv::forward` (C0.2)

Can start at once. It is independent of T2.

Read first:
- `docs/CONVENTIONS.md` §1, §4 and §5;
- `docs/design/spectral-conv-perf.md` §2.1 (the operator, including the `irfft` formula),
  §5.1, §8.1 and §8.2 (the oracle note);
- the phase README's "Design decisions" (E-oracle, E-control, deterministic data);
- `src/neural_operators/layers/spectral_convolution.rs`:
  - `corner_ranges` (`:216-227`), `check_modes_fit` (`:234-247`), `forward` (`:325-376`);
  - `mod tests` (`:389` on), in particular `forward_2d_matches_pytorch_reference`
    (`:496-529`), which shows how weights are set;
- `src/neural_operators/losses/data_losses.rs:397` (`TensorData::new` upload).

This task delivers an independent f64 reference for the layer. It is a plain-Rust
separable DFT that does not use Burn's FFTs, compared against today's
`SpectralConv::forward`. Phase 1 and phase 2 reuse it unchanged as their oracle. T3
copies the errors it prints into the baseline report.

Do:
- `src/neural_operators/layers/spectral_convolution.rs`: add, at the end of the file,
  exactly

  ```rust
  #[cfg(test)]
  #[path = "spectral_convolution_oracle.rs"]
  mod oracle;
  ```

  This is the only change to that file.
- `src/neural_operators/layers/spectral_convolution_oracle.rs` (new). It starts with a
  `//!` comment saying what it is and citing "CONVENTIONS §4, §5" and "design
  spectral-conv-perf §2.1". Contents:

  ```rust
  use super::*;              // SpectralConv, Param, Tensor, Device
  use burn::tensor::TensorData;  // not imported by the parent (`:18-24`)

  /// Fixed-seed host generator (e.g. SplitMix64 or an LCG), uniform in [-1, 1).
  struct HostRng(u64);
  impl HostRng { fn next_f32(&mut self) -> f32; }

  /// Row-major f64 array with its shape.
  struct Array { data: Vec<f64>, shape: Vec<usize> }

  /// Layer under test with deterministic weights. Returns the layer, plus every corner's
  /// (re, im) weights as the f32 values actually uploaded, cast to f64,
  /// each shaped [I, O, modes..].
  fn layer_with_weights<const R: usize>(
      device: &Device, i: usize, o: usize, modes: &[usize], rng: &mut HostRng,
  ) -> (SpectralConv<R>, Vec<(Array, Array)>);

  /// What the oracle computes; the controls switch one rule off.
  #[derive(Clone, Copy)]
  struct Variant { swap_corners: bool, nyquist_weight_two: bool, keep_dc_imag: bool }

  /// Design §2.1 in f64:
  /// - forward DFT per axis, unnormalised, in CONVENTIONS §4's axis order;
  /// - keep the CONVENTIONS §5 corners, and mix each with `out[b,o,k] = Σ_i x[b,i,k] w[i,o,k]`;
  /// - inverse complex DFT (1/n) on axes D−1..1;
  /// - then `irfft` on the last axis, which ignores Im at DC and, for even n, at Nyquist.
  /// x: [B, I, s_1..s_D] -> y: [B, O, s_1..s_D].
  fn oracle_forward(x: &Array, weights: &[(Array, Array)], modes: &[usize], v: Variant) -> Array;

  /// ‖a − b‖∞ / ‖b‖∞.
  fn rel_inf(a: &[f64], b: &[f64]) -> f64;
  ```

  - Implement `oracle_forward` **separably**, with plain `f64` loops and
    `f64::cos`/`f64::sin`:
    - forward along each spatial axis, then the per-frequency channel mix, then the
      inverse along each axis;
    - compute every phase as `2π · ((k·x) mod n) / n` with integer `k·x mod n`;
    - no `burn::tensor::signal`, no `rustfft`, no new dependencies.

    The full spectrum may be formed and then masked to the retained corners. Correctness
    matters here, not speed.
  - **Corner indexing must be computed independently** of `SpectralConv::corner_ranges`.
    Corner `mask`, bit j for non-last axis j (0-based), selects the high block
    `s_j − m_j .. s_j` when set (CONVENTIONS §5). Do not call `corner_ranges`.
  - **Variants (controls).**
    - `swap_corners` exchanges the weights of corners 0 and 1 (D ≥ 2).
    - `nyquist_weight_two` uses weight 2 instead of 1 for the retained Nyquist bin in the
      last-axis inverse.
    - `keep_dc_imag` adds the DC imaginary part's contribution in the last-axis inverse:
      `−Im Y_0 · sin(0)` is 0, so add `Im Y_0 / n` instead. This models a backend that
      does not ignore it.
  - Inputs: `x` from `HostRng`, uploaded with
    `Tensor::<R>::from_data(TensorData::new(vec, shape), &device)`.
  - Weights: from `HostRng`, scaled by 1/√(I·O). `layer_with_weights` necessarily calls
    `SpectralConv::new`, which draws `Tensor::random` (`:190-199`) before the weights
    are overwritten. That is allowed; the ban below is on using random draws as test
    data. Set them via
    `conv.weights_re[c] = Param::from_tensor(..)`, as at `spectral_convolution.rs:502-506`.
  - Read the layer output with `into_data().try_to_vec::<f32>()`.
  - Device: `Device::default()` (no autodiff).
  - Each test prints `case, rel_inf` with `println!` so that T3 can collect the values
    with `--nocapture`.
  - Shared parameters: B = 2, I = 3, O = 2 (I ≠ O), seed fixed per test.

Tests that define done (each is a `#[test]` in the oracle module):
- **`oracle_1d`.** `SpectralConv<3>`, against the oracle with every `Variant` flag false,
  using E-oracle. Cases:

  | s | modes | why | tolerance |
  |---|---|---|---|
  | 16 | [9] | retained Nyquist (9 = 16/2 + 1), power of two | 1e-5 |
  | 15 | [8] | odd, at the last-axis limit, Bluestein | 1e-5 |
  | 12 | [7] | even non-power-of-two with retained Nyquist (Bluestein `irfft`, `hermitian_extend`) | 1e-5 |
  | 94 | [16] | Darcy padded extent, even non-power-of-two (Bluestein) | 1e-5 |
  | 256 | [16] | Burgers example | 3e-5 |
  | 1024 | [16] | Burgers at r = 8 | 3e-5 |
- **`oracle_2d`.** `SpectralConv<4>`, all flags false, E-oracle 1e-5. Cases:
  - (94, 94), modes [12, 12];
  - (16, 16), modes [8, 9]: non-last axis at its limit 16/2, retained Nyquist on the last axis;
  - (15, 17), modes [7, 9]: both odd, both at their limits.
- **`oracle_3d`.** `SpectralConv<5>`, (8, 6, 10), modes [2, 3, 4], all flags false,
  E-oracle 1e-5.
- **`oracle_controls`.** Shows that E-oracle can fail. Each mutation must give
  E-control ≥ 1e-3 relative to the unmutated oracle's ‖y‖∞:
  - `swap_corners` on 2D (16, 16) [8, 9];
  - `nyquist_weight_two` on 1D s = 16 [9], 1D s = 12 [7] and 2D (16, 16) [8, 9];
  - `keep_dc_imag` on 1D s = 94 [16].
- **`oracle_reproduces_pinned_reference`.** Feed the oracle the input and weights of
  `forward_2d_matches_pytorch_reference` (`:496-529`): 4 × 4 input 1..16, modes [1, 1],
  weights 0.1 + 0.2i and 0.3 + 0.4i, given to the oracle as their f32 values cast to f64
  (`0.1f32 as f64`, and so on). It must reproduce that test's expected values to
  ‖a − b‖∞ ≤ 1e-4 absolute, the tolerance the existing test uses for these PyTorch-pinned
  values (‖b‖∞ = 2.25). This checks the oracle against a published reference, not only
  against the layer.

Must pass:
1. `cargo fmt` after every edit.
2. CI, exactly as `CLAUDE.md` lists it:
   - `cargo fmt -- --check`
   - `cargo clippy --no-deps -- -D warnings`
   - `cargo clippy --no-deps --examples -- -D warnings`
   - `cargo test`
   - `cargo doc --no-deps`
3. `cargo clippy --all-targets -- -D warnings`. The oracle is test code, so this is the
   run that lints it. No `#[allow]`s, except a narrowly scoped
   `#[allow(clippy::too_many_arguments)]` if one is genuinely needed, with a reason.
4. Targeted:
   ```
   cargo test --lib neural_operators::layers::spectral_convolution::oracle:: -- --nocapture
   ```
   Check that "running N tests" says 5 (CLAUDE.md gotcha). With `--lib`, only the library
   test binary prints a count.
5. `cargo test --lib --features metal neural_operators::layers::spectral_convolution::oracle:: -- --nocapture`.
   Report in the PR whether it ran and what it printed. If no Apple GPU is available,
   say so; do not report it as passed.
   - The 2D (94, 94) and 3D (8, 6, 10) cases are **expected to fail on Metal** (phase
     README "Known issue", design §9.1 R8).
   - Report their values. Do not `#[ignore]` them, and do not change tolerances for
     Metal. CI is flex only.
   - Make each test check every case before panicking, so that every value is printed
     even when one fails.
6. Paste the printed `case, rel_inf` lines from both runs into the PR description.

If a case fails E-oracle on flex, **do not loosen the tolerance and do not change
`SpectralConv`.**
- First rule out an oracle bug: the controls and the pinned-reference test must pass.
- Then report the observed value in the PR. Move that case out of its table into its
  own `#[ignore]`d `#[test]` with a comment citing risk R4. The step 4 test count rises
  above 5 accordingly; say so in the PR.
- T3 proposes the §8.2 revision for the user's sign-off.

Note: the oracle hard-codes "`irfft` ignores Im at DC and Nyquist", and `keep_dc_imag`
tests that behaviour at layer level, for the sizes tested. The `irfft`-level tests and
the CONVENTIONS §4 record belong to pytorch-parity T2 and T4. Mention in the PR that this
oracle agrees with them.

Do not:
- Change anything in `spectral_convolution.rs` except the three-line `mod oracle;`
  include, and touch no other library code (`utils/fft.rs`, `models/fno.rs`).
- Add the `irfft` DC/Nyquist tests (pytorch-parity T2 owns them), or any dependency or
  fixture file.
- Edit `docs/CONVENTIONS.md` or the design document. T3 records the results.
- Use `Tensor::random`, `device.seed` or `SpectralInit` draws for the test data.
