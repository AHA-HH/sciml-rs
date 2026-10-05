# pytorch-parity phase 0 / T3 - Pin the default FNO's parameters (C0.3)

Can start at once. **Every Phase 1 task depends on this one.**

Read first:
- `docs/design/pytorch-parity.md` §2.3 ("Implementation note: module structure and
  checkpoints"; the last bullet, "`None` branches must make no RNG calls") and §5.4;
- `tests/spectral_init.rs`, the whole file: the `rng()` mutex, the `params()` and
  `bits()` helpers, and `default_spectral_init_is_bit_identical`;
- `src/neural_operators/models/fno.rs:54-110`: the `FNO` fields, in declaration order
  `fc0, conv, w, fc1, fc2, padding`, and `FNOConfig::init`;
- `src/neural_operators/layers/spectral_convolution.rs:162-207` (`new_with_init`, eager
  draws);
- under `~/.cargo/git/checkouts/burn-4aa1ec9707f89bee/ddfa9af/crates/`:
  `burn-nn/src/initializer.rs:100-125, :282-290` (lazy `Param::uninitialized`) and
  `burn-nn/src/modules/linear.rs:59-80`.

This task delivers a test binary that pins every parameter a default `FNOConfig` draws at
a fixed seed, bit for bit. Phases 1–2 add many `Option` fields to `FNOConfig`; this test
catches any of them that makes an RNG draw when it is `None`.

The existing tests cannot catch that:
- `default_spectral_init_is_bit_identical` replays draws for a bare `SpectralConv` only.
  At FNO level it compares `None` against `Some(LiUniform)`, and a stray draw shifts both
  sides equally.
- The seeded FNO tests in `fno.rs` compare against the model's own weights.

**How draws are ordered, which the test must respect.**
- `SpectralConv::new_with_init` draws its weights **eagerly**, during construction.
- `Linear` and `Conv1d` parameters are **lazy**: they are drawn on first access.
- So the weights depend on the order of construction *and* the order of first access.

The test fixes both:
- it builds a mirror module with the same fields, in the same order, constructed in the
  same order as `FNOConfig::init`;
- it reads every parameter of both models through the same helper, so first access
  happens in the same visit order.

Do:
- **New file `tests/fno_default_params.rs`.** It is its own binary, and so its own
  process, because Flex's RNG is process-wide.
  - Module doc comment: what is pinned and why, citing design §2.3. State the rule for
    later PRs: this test may only change when a PR deliberately changes the default
    model, and that PR must say so and bump `CONVENTION_VERSION`.
  - Copy the `rng()` mutex, `params()` and `bits()` helpers from `tests/spectral_init.rs`.
    Keep them local; do not share code between test binaries.
  - **`params()` reads the tensors in visit order**: field declaration order, then
    `Vec` index order. `ModuleSnapshot::collect` itself initialises lazy parameters
    while visiting: `collector.rs:139` calls `param.transform_for_save().val()`. The
    `sort_by` comes after. Write a comment saying that the order of first access matters
    for lazy parameters, and why.
  - **Always seed → build → `params()` immediately, before the next `seed`.** Read every
    model's parameters right after it is built, before re-seeding for the next one.
    `tests/spectral_init.rs` does this with `params(&model(..))`.
    - Why: if two models are built first and read afterwards, the first model's lazy
      draws happen after the second model's eager spectral draws, and the bits differ.
    - The review reproduced this: it gives "not equal" with today's unchanged code.
  - Define a mirror module in the test:

    ```rust
    #[derive(Module, Debug)]
    struct Mirror<const R: usize> {
        fc0: Linear,
        conv: Vec<SpectralConv<R>>,
        w: Vec<Conv1d>,
        fc1: Linear,
        fc2: Linear,
    }
    /// Builds the components in FNOConfig::init's order, with today's defaults:
    /// fc0 = Linear(data + D, H), conv = SpectralConv::new_with_init(H, H, modes,
    /// LiUniform) x L, w = Conv1d(H, H, 1) x L, fc1 = Linear(H, 128), fc2 = Linear(128, out).
    fn mirror<const R: usize>(device: &Device, data: usize, h: usize, modes: &[usize],
                              layers: usize, out: usize) -> Mirror<R>;
    ```

    - The field names must make `params()` names identical to `FNO`'s (`fc0.weight`,
      `conv.0.weights_re.0`, `w.1.bias`, …).
    - `padding` is not a parameter, so it is left out. If `#[derive(Module)]` on a test
      struct does not compile in this Burn fork, report the error. Fall back to
      collecting each component's `params()` in order with the `fc0.`/`conv.<i>.`/…
      prefixes added by hand, and say so in the PR.

Tests that define done (error measures from the phase README):
- **`default_fno_1d_draws_are_pinned`.** Hold `rng()`.
  - Model: `FNOConfig::new(vec![4], 1, 1).with_hidden_channels(6).with_n_layers(2)` with
    `init::<3>`, seed 2024.
  - Mirror: `mirror::<3>(.., data 1, h 6, modes [4], layers 2, out 1)`, after re-seeding
    with 2024.
  - Check: the name lists are equal, and every parameter is equal to `E-bits`.
- **`default_fno_2d_draws_are_pinned`.** The same, with `FNOConfig::new(vec![4, 3], 1, 1)`,
  `init::<4>` and modes [4, 3]. This case has two spectral corners.
- **`default_config_matches_explicit_defaults`.** `FNOConfig::new(..)` with no `with_*`
  calls, and the same config with `with_spectral_init(None)` and `with_padding(None)`,
  give `E-bits`-equal parameters at seed 7. This guards the meaning of `None` for the
  existing `Option` fields.
- **Vacuity controls**, each with ‖Δ‖_∞ ≥ 1e-2 (`E-diff`) on at least one parameter. They
  show the pin detects the bug it exists for.
  - **(a) Stray eager draw before the spectral weights.** A mirror built with one extra
    `Tensor::<2>::random([1, 1], Distribution::Default, &device)` right after `fc0`'s
    construction must differ from `FNO` on `conv.0.weights_re.0`.
  - **(b) Stray lazy draw.** A mirror built with an extra `Linear` constructed between
    `conv` and `w`, and read with `let _ = extra.weight.val();` during construction,
    must differ on `w.0.weight`.
    - Expected side effect: the read happens before *any* of the mirror's lazy draws, so
      it shifts every lazy parameter (`fc0.*` included) while the spectral weights stay
      equal. The review measured Δ 0.6–1.1 on the lazy parameters.
    - A lazy parameter that is constructed but never read makes no draw. That case is
      harmless, and is not a gap in the pin.
- **Run on metal.** Run the binary with `--features metal`. The test compares draws on
  the same device, so it should pass on any backend. Record in the PR whether that run
  happened.

Must pass:
1. `cargo fmt`.
2. Then, as `CLAUDE.md`'s CI runs them:
   - `cargo fmt -- --check`
   - `cargo clippy --no-deps -- -D warnings`
   - `cargo clippy --no-deps --examples -- -D warnings`
   - `cargo test`
   - `cargo doc --no-deps`
3. Also `cargo clippy --all-targets -- -D warnings`.
4. The targeted run: `cargo test --test fno_default_params`. Check that the "running 5
   tests" line says 5, not 0.

Do not:
- Change anything under `src/`. The test must pass against today's code unchanged; if it
  does not, report why instead of changing library code.
- Change `tests/spectral_init.rs`.
- Pin checksums or literal weight values. They differ by backend, and the mirror
  comparison does not.
- Edit `docs/CONVENTIONS.md` (T4 owns it).
