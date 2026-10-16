# metal_diag: GPU vs Flex diagnostics

The bisection notebook behind the Burn bug reports in `upstream/burn/`: the fused-`cat`
broadcast-stride bug (`cfft-kernel/`) and the fused-`cat` replay at a new shape
(`fused-cat-reused-at-new-shape/`). Written in Phases 3 and 4 as `tests/metal_diag.rs`
and moved here in Phase 4s T7.

Each test runs an op, a layer or the FNO on Flex and on the GPU from the same host input,
and prints the relative errors, with `<-- BAD` (or `BAD`, `!`) on a mismatch. Apart from
`repro_fusion_cache_new_shape` (an intermittent, env-var-driven variant of the published
repro) the tests assert nothing, so a pass means only that they ran. Read the output.

The asserting regression tests (`repro_*` and `replay_2d_poisson`) stay in the root crate:
`cargo test --features metal --test gpu_parity`.

## Running

From the repository root:

```sh
cargo test -p metal-diag-spike --release -- --nocapture                      # Metal
cargo test -p metal-diag-spike --release -- --nocapture after_warmup_blocks  # one test
cargo test -p metal-diag-spike --release --no-default-features --features cuda -- --nocapture
```

Some tests read environment variables (`MIN`, `CFG`, `PARTS`, `SHAPES`, `WARMS`); see the
test bodies.

The package is a non-default member of the root workspace. It shares the root's
`Cargo.lock`, `target/` and Burn pin, and root `cargo test` and CI never build it.
