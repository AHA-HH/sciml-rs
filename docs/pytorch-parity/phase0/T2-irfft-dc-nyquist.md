# pytorch-parity phase 0 / T2 - `irfft` DC and Nyquist imaginary parts (C0.2)

Can start at once. T4 writes the CONVENTIONS §4 line that cites this task's test, and
both briefs fix the test name, so the two tasks can merge in either order.

Read first:
- `docs/CONVENTIONS.md` §4, in particular the "Not yet fixed by a test" bullet;
- `docs/design/pytorch-parity.md` §2.1 ("main #702");
- `src/neural_operators/layers/spectral_convolution.rs`: `ifft_ctensor` (`:305-315`)
  and `mod tests` (`:390` on), especially `fft_round_trip_*` and the autodiff test at
  `:591-620`;
- the Burn sources this test pins, all under
  `~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/`:
  - `burn-signal/src/functions/fft.rs:118-160`: `irfft` and its Bluestein branch;
  - `burn-flex/src/ops/fft.rs:1277-1278`;
  - `burn-signal/src/backends/autodiff.rs:95-127, :169-194`.

This task delivers tests that fix how `signal::irfft` treats the imaginary parts of the
DC bin (every n) and of the Nyquist bin (even n): it ignores them, as NumPy and torch do. It ignores them exactly for power-of-two n,
and to f32 round-off on the Bluestein path. The review measured value differences up to
8.9e-8 on flex.
That closes CONVENTIONS §4's open point. It also confirms that neuraloperator main #702
(Hermitian enforcement before the inverse transform) needs no extra step on equal grids
(design §2.1). Later phases rely on this when they compare against PyTorch values.

Do:
- `src/neural_operators/layers/spectral_convolution.rs`, `mod tests` only: add two
  tests and a small helper. Use `signal::irfft(re, im, 0, Some(n))` on `Tensor<1>`, with
  the device from `Device::default()`. In `mod tests`, cite this as "CONVENTIONS §4".

  ```rust
  /// Pasted from NumPy 2.0.1: np.fft.irfft(re + 1j*im, n=n), float64, rounded to 9 d.p.
  #[test]
  fn irfft_ignores_dc_nyquist_imag() { /* see cases below */ }

  #[test]
  fn irfft_grad_is_zero_for_dc_nyquist_imag() { /* autodiff, see below */ }
  ```

  - **Inputs**, as half-spectra:
    - n = 8 (power of two, flex native kernel) and n = 9 (odd, Bluestein, no Nyquist)
      share the 5-bin spectrum: re = `[1.0, 0.5, -0.25, 0.75, 2.0]`,
      im = `[0.3, -0.4, 0.6, 0.2, -0.7]`.
    - n = 12 (even, not a power of two, Bluestein) uses the 7-bin spectrum:
      re = `[1.0, 0.5, -0.25, 0.75, 0.1, -0.6, 2.0]`,
      im = `[0.3, -0.4, 0.6, 0.2, 0.9, -0.5, -0.7]`.
  - **Expected `irfft` output**, from NumPy 2.0.1, float64:
    - n = 8: `[0.625000000, -0.283838835, 0.587500000, 0.104549513, 0.000000000, -0.266161165, 0.287500000, -0.054549513]`
    - n = 12: `[0.333333333, -0.128568360, 0.158034180, 0.158333333, 0.338098306, -0.013098306, 0.116666667, -0.529444342, 0.453568360, -0.208333333, 0.100299153, 0.221111008]`
    - n = 9: `[0.777777778, -0.373852367, 0.420177978, 0.354942930, -0.117589100, 0.033561960, -0.299387375, 0.459304747, -0.254936550]`

Tests that define done:
- **Values.** For each n ∈ {8, 12, 9}, `irfft` on the inputs above matches the pasted
  NumPy output, to `E-ref`: ‖a − b‖_∞ ≤ 1e-5 · ‖b‖_∞ + 1e-7.
- **DC and Nyquist are ignored.** For each n, zero `im[0]`, and for even n also
  `im[last]`. The output equals the unmodified output to `E-same`:
  ‖a − b‖_∞ ≤ 1e-6 · ‖b‖_∞ + 1e-7. NumPy gives a difference of exactly 0.0.
- **Vacuity control.**
  - For n = 9, the last bin (index 4) is an interior frequency. Zeroing `im[4]` must
    change the output by ‖Δ‖_∞ ≥ 1e-2 (`E-diff`; NumPy gives 0.1532).
  - For n = 8, zeroing an interior imaginary part, `im[2]`, must also change it by
    ≥ 1e-2.
  - Without these, a function that ignores every imaginary part would pass.
- **Gradients.**
  - Setup: on `Device::default().autodiff()`, with `im` set to `require_grad()`, take
    L = Σ_k irfft(re, im)[k] · r_k, where r_k = k + 1 (k = 0..n−1, a fixed weight
    vector).
  - **Check the whole `im.grad(&grads)` vector** against these NumPy values. They were
    computed by unit impulses, g_k = Σ_j r_j · ∂irfft/∂im_k, with NumPy 2.0.1 in
    float64. The tolerance is ‖a − b‖_∞ ≤ 1e-4 · ‖b‖_∞ + 1e-7 (design §8.2, gradients).

    | n | expected gradient |
    |---|---|
    | 8 | `[0, 2.414214, 1, 0.414214, 0]` |
    | 12 | `[0, 3.732051, 1.732051, 1, 0.577350, 0.267949, 0]` |
    | 9 | `[0, 2.747477, 1.191754, 0.577350, 0.176327]` |

  - **n = 8 (native power-of-two kernel): also exact zeros.** Check exactly 0.0
    (`E-zero`) at index 0 and at index 4. The `Irfft` backward pass and flex's `rfft`
    set these exactly (`burn-flex/src/ops/fft.rs:694-700`).
  - **n = 9 and n = 12 (Bluestein): zero only to round-off.** Gradients flow through
    ordinary ops (`burn-signal/src/functions/fft.rs:147-155`; `hermitian_extend` keeps
    `im[0]` and `im[n/2]`), so the zeros are zero only up to f32 round-off. The review
    measured up to 1.1e-7 on flex and metal. Use the bound above; do **not** assert
    exact zero there.
  - Control: the index-1 gradient is nonzero (|g| > 1e-3) for every n.
- **Run on metal.** Run both tests with `--features metal`. Record in the PR whether
  that run happened and what it gave. If no Apple GPU is available, say so; do not
  report it as passed.

Must pass:
1. `cargo fmt` (after every edit).
2. Then, as `CLAUDE.md`'s CI runs them:
   - `cargo fmt -- --check`
   - `cargo clippy --no-deps -- -D warnings`
   - `cargo clippy --no-deps --examples -- -D warnings`
   - `cargo test`
   - `cargo doc --no-deps`
3. Also `cargo clippy --all-targets -- -D warnings`.
4. The targeted run:

   ```
   cargo test --lib neural_operators::layers::spectral_convolution::tests::irfft_ -- --nocapture
   ```

   Check that the "running N tests" line says 2, as the `CLAUDE.md` gotcha warns.
   Without `--lib`, every test binary prints its own line, and the others say 0.

Do not:
- Change any non-test code in `spectral_convolution.rs`, `utils/fft.rs` or anywhere
  else.
- Edit `docs/CONVENTIONS.md`. T4 owns it, and its §4 line already names
  `irfft_ignores_dc_nyquist_imag`. Keep that exact test name.
- Add Python or fixture files.
