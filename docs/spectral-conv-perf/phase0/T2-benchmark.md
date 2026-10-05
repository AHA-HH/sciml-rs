# spectral-conv-perf phase 0 / T2 - `bench_spectral_conv` example (C0.1)

Can start at once. It is independent of T1.

Read first:
- `docs/design/spectral-conv-perf.md` §1 (definition of done, settings), §2.6 (cost
  model) and §8.3 (benchmark protocol);
- the phase README's "Design decisions" (timing protocol, benchmark settings);
- `examples/train/burgers.rs` and `examples/train/darcy.rs`: their `FNOConfig` literals,
  and the header comment style ("Run with: ...");
- `src/neural_operators/training/trainer.rs`:
  - one training step is `:172-202`: forward, `flatten_pair`, `LpLoss::rel`,
    `GradientsParams::from_grads`, `optim.step`;
  - Adam is built at `:274-277`;
- `src/neural_operators/losses/data_losses.rs:34` (`LpLoss::new`) and `:96` (`rel`);
- `Cargo.toml` `[[example]]` entries;
- fork `~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/burn-tensor/src/device.rs:707`
  (`Device::sync`).

This task delivers the benchmark that defines "faster" for this objective. It times
`SpectralConv` forward + backward alone, and a full FNO training step, at the design §1
settings. T3 records its baseline output. Phases 1–3 rerun it unchanged and compare.

Do:
- `examples/bench/spectral_conv.rs` (new). The `//!` header says what it measures and how
  to run it:
  `cargo run --release --example bench_spectral_conv [--features metal]`. Hyperparameters
  are `const` literals at the top (CLAUDE.md "Layout"):

  ```rust
  const BATCH: usize = 20;
  const WARMUP: usize = 5;
  const TIMED: usize = 20;
  // Layer cases: (I = O, spatial extents, modes)
  // 1D: (64, [256], [16]), (64, [1024], [16]);  2D: (32, [94, 94], [12, 12])
  // FNO step cases: burgers s = 256 and s = 1024 (FNOConfig of examples/train/burgers.rs),
  //                 darcy s = 85 (FNOConfig of examples/train/darcy.rs, padding Some(9))
  ```

  - Device: `Device::default().autodiff()`. Print `device: {device:?}` first, as the
    training examples do.
  - **Layer case:**
    - Build `SpectralConv::<R>::new(&device, c, c, &modes)`.
    - Use input `Tensor::random([BATCH, c, s..], Normal(0, 1))` with `require_grad()`.
    - One step is `conv.forward(x).powf_scalar(2.0).sum().backward()`.
    - Keep the returned `Gradients` alive until after the closing `device.sync()`, then
      drop them. Dropping them earlier could let a fusing backend skip work.
  - **FNO step case:**
    - Build `FNOConfig { .. }.init::<R>(&device)`, with the literals copied from the
      training example (record the source line in a comment).
    - Use random inputs `[BATCH, s.., data_channels]` and targets `[BATCH, s..]`.
    - Use `LpLoss::new(R - 2, 2, Reduction::Sum)`, as `trainer.rs:306` constructs it.
    - Use Adam as at `trainer.rs:274-277`:
      `AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1e-4))).init()`.
      The 1e-4 is from `examples/train/{burgers,darcy}.rs`. Use a fixed learning rate
      of 1e-3.
    - One step is forward, flatten to `[B, n_points]`, `rel`, backward,
      `GradientsParams::from_grads`, `optim.step`.
    - Thread the model through the steps, as `train_epoch` does.
  - **Timing.**
    - For each case, run `WARMUP` untimed steps.
    - Then, for `TIMED` steps: call `device.sync()`, record `Instant::now()`, run the
      step, call `device.sync()`, record the elapsed time.
    - `sync()` returns `Result`; `expect` it with a message.
  - **Output.**
    - One table on stdout with one row per case: `case | backend | median ms | IQR ms
      (q1–q3) | steps`.
    - The backend is labelled by build feature, with `cfg!(feature = "metal")` giving
      "Metal" and anything else "flex". The debug string on macOS reads
      `Cube(Wgpu(.. backend: Auto))`.
    - Compute the median and quartiles from the sorted samples, using the nearest-rank
      method. Say in a comment that with 20 samples these are x(5), x(10) and x(15), so
      the "median" is the lower middle value.
    - Write no files.
  - Optional CLI filter: `std::env::args().nth(1)` as a substring match on the case name,
    so `... -- darcy` runs only the Darcy cases. No CLI dependency.
- `Cargo.toml`: add, after the existing examples,

  ```toml
  [[example]]
  name = "bench_spectral_conv"
  path = "examples/bench/spectral_conv.rs"
  ```

- `README.md`: no change. T3 decides where the baseline is documented.

Tests that define done (this is an example; it has no unit tests, so these are checks):
- **Builds and lints:** CI's `cargo clippy --no-deps --examples -- -D warnings` passes.
- **Runs on flex:** `cargo run --release --example bench_spectral_conv` finishes. It
  prints 6 rows: 3 layer cases and 3 FNO-step cases. Every median is > 0, and q1 ≤ median ≤ q3.
  Paste the table into the PR.
- **Runs on Metal:** `cargo run --release --features metal --example bench_spectral_conv`.
  Paste the table, or say that no Apple GPU was available. Do not report it as passed if
  it did not run.
- **Sync is real, not decorative.** Run the 2D layer case on Metal, or on flex if Metal is
  unavailable, once more with the closing `device.sync()` removed (a temporary local
  edit, reverted before commit). Report both medians in the PR. On Metal the version
  without sync should be noticeably lower, which shows the sync matters. On flex they
  may be equal. This shows that the timings include queued GPU work.
- **Filter works:** `cargo run --release --example bench_spectral_conv -- darcy` prints
  only the Darcy row.

Must pass:
1. `cargo fmt` after every edit.
2. CI, exactly as `CLAUDE.md` lists it:
   - `cargo fmt -- --check`
   - `cargo clippy --no-deps -- -D warnings`
   - `cargo clippy --no-deps --examples -- -D warnings`
   - `cargo test`
   - `cargo doc --no-deps`
3. `cargo clippy --all-targets -- -D warnings`.
4. The runs above, with outputs in the PR.

Do not:
- Change any library code (`src/`), including `trainer.rs`. If a helper you need is
  private (for example `flatten_pair`), reimplement the few lines it takes in the
  example, with a comment citing the original.
- Add dependencies (no `criterion`), datasets, or files under `runs/`.
- Tune the settings away from the phase README's list. They are the design §1 gate.
- Record the baseline in docs. That is T3.
