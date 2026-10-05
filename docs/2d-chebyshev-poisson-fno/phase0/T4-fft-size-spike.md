# 2d-chebyshev-poisson-fno phase 0 / T4 - FFT size and padding timing spike (C0.4)

Can start at once.

Read first:
- `docs/CONVENTIONS.md` §3 (padding) and §4 (FFT, any length via Bluestein).
- `docs/design/2d-chebyshev-poisson-fno.md` §6.2 and §6.5.
- `CLAUDE.md` (Burn, backends).
- `src/neural_operators/models/fno.rs`: `FNOConfig` (:18-52), `FNO::forward` (:249).
- `src/neural_operators/training/trainer.rs` (:249, :315).
- `examples/train/darcy.rs` (model and training settings).
- The pinned FFT code: `~/.cargo/git/checkouts/burn-*/ddfa9af/crates/burn-signal/src/functions/fft.rs`
  (:53-70, :118-160, :208-230).

The FFT accepts any length, but a padded length that is not a power of two goes through Bluestein's
algorithm. Bluestein pads internally to the next power of two ≥ 2n − 1, so it can be slower. This
spike measures the training-step cost of the candidate model grid sizes and paddings, so that s and
p are chosen from data (design §9.2 Q6). It changes no library code.

Do:
- Create `spikes/fft-sizes/`, a stand-alone binary crate:
  - `Cargo.toml` with an empty `[workspace]` table;
  - dependency `sciml-rs = { path = "../.." }`, plus `burn` exactly as the root `Cargo.toml` pins
    it, if it is needed directly;
  - features `flex` (default) and `metal` that forward to `sciml-rs/flex` and `sciml-rs/metal`.
- Add `/spikes/*/target/` to `.gitignore`. T3 adds the same line; keep one copy.
- `spikes/fft-sizes/src/main.rs` contains:
  - **Model.** For each s ∈ {55, 63, 64, 65} and p ∈ {0, 9, s}, build `FNOConfig` exactly as
    `examples/train/darcy.rs`:
    - modes [12, 12], hidden 32, 4 layers, 1 data channel, 1 output channel;
    - padding `Some(p)`, or `None` for p = 0.
  - **Data.** Random data `[1000, s, s, 1]` and targets `[1000, s, s]`, f32, seeded.
  - **Training.** Run the repository's own training path for 1 warm-up epoch and 3 timed epochs:
    batch 20, Adam, `LpLoss::rel`, as in `training/trainer.rs`. Calling `build_training_components`
    and `training_loop` with `epochs = 4` and timing the epochs is acceptable. So is a minimal loop
    with the same forward, loss, backward and optimiser step; record which one you used.
  - **Output.** Print a Markdown table with columns: s, p, padded length s + p, power of two (yes/no),
    median epoch seconds (of the 3 timed epochs), and seconds relative to (s = 64, p = 0).
- Run it with `--release` on `flex`. If on Apple hardware, also run with `--features metal`.
- **`spikes/fft-sizes/SPIKE_REPORT.md`** contains:
  - the tables;
  - the machine, OS, toolchain and backend;
  - a note on any configuration that failed. For example, modes 12 must satisfy CONVENTIONS §5
    limits on the padded extent; every candidate does;
  - a recommendation for (s, p):
    - Prefer p ≥ 8, which keeps the non-periodic padding margin of design §3.2.
    - Among those, take the fastest. If configurations are within 20 % of each other, prefer s = 64
      for data generation.
    - State the speed ratio of Bluestein to power-of-two that you measured.

Tests that define done:
- `padded_lengths`: a `#[test]` asserting that for every (s, p) the table reports s + p and the
  correct power-of-two flag.
  - The oracle is a hard-coded set, not `is_power_of_two`: exactly (55, 9) → 64, (64, 0) → 64 and
    (64, 64) → 128 are powers of two, as in design §6.2.
- `forward_shapes`: for (s, p) = (63, 9) and (64, 0), one forward pass on a `[2, s, s, 1]` input
  returns `[2, s, s, 1]`, and every value is finite. This is checked against the expected shape and
  `is_finite`, on flex.
- The timing table itself is the deliverable. It is a measurement, so no threshold applies, but
  every row must be present or have a stated reason why not.

Must pass:

```sh
cargo fmt --manifest-path spikes/fft-sizes/Cargo.toml -- --check
cargo clippy --manifest-path spikes/fft-sizes/Cargo.toml -- -D warnings
cargo test --manifest-path spikes/fft-sizes/Cargo.toml
cargo run --release --manifest-path spikes/fft-sizes/Cargo.toml
```

and, when on Apple hardware:
`cargo run --release --manifest-path spikes/fft-sizes/Cargo.toml --features metal`.
State whether it ran.

Also run the repository's full checks:

```sh
cargo fmt -- --check
cargo clippy --no-deps -- -D warnings
cargo clippy --no-deps --examples -- -D warnings
cargo test
cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
```

Do not:
- change `src/`, the root `Cargo.toml`, the FFT code or `CONVENTIONS §3/§4`. In particular, do not
  round padding to powers of two inside the model;
- add the spike to CI;
- use real Poisson data. None exists yet, and timing does not depend on it;
- edit the design document. The numbers go into it at the phase exit, after sign-off.
