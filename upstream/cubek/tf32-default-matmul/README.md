# f32 matmuls default to TF32 on cuda (a 50× accuracy drop against Burn 0.22)

**Status:** not a bug: a documented default (`MatmulConfig::allow_tf32 = true` in
`cubek-std 0.4.0-pre.2`, `src/config.rs`). Worked around in sciml-rs by `cubek.toml` at
the repository root (PR #119). Worth raising upstream only as a request to state the
change in Burn's release notes, or to default to full f32.

## Summary

Burn 0.23's cuda matmuls round f32 inputs to TF32 (10 mantissa bits) when the GPU
supports it. Nothing in Burn's API says so; it is set in CubeK's config, read from
`cubek.toml` or the `[cubek]` section of `burn.toml`, walking up from the working
directory. Code that was f32-accurate on Burn 0.22 becomes about 1e-4-accurate.

## Measurement

`tests/gpu_parity.rs`, `replay_2d_poisson`: a 2D FNO (4 layers, width 32, 12 × 12 modes,
paddings 0, 4, 8, 16), forward and backward on the GPU against Flex, worst relative
difference; NVIDIA A100 80GB PCIe.

| | worst GPU/Flex difference |
|---|---|
| Burn fork `ddfa9af` (0.22.0-pre.4) | 3.6e-6 |
| Burn 0.23.0-pre.2, default | 1.6e-4 (every call 3e-5 to 1.6e-4) |
| Burn 0.23.0-pre.2, `allow_tf32 = false` | 1.4e-6 |

## Why it matters here

Gauss–Newton (Phase 10) computes J v by central finite differences in f32 and accepts
a step on the ratio ρ of two small differences of the loss. Both lose accuracy with
every digit the matmuls drop, so training under TF32 would silently change.

## Setting

```toml
# cubek.toml at the repository root
[matmul]
allow_tf32 = false
```

Runs must start from inside the repository for the file to be found. The speed cost of
full-f32 kernels was not measured.
