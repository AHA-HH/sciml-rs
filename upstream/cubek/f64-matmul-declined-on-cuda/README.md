# f64 matmul on cuda panics: every autotune candidate declines f64

**Status:** open; not filed upstream yet (2026-11-13). Found while moving sciml-rs to
Burn 0.23.0-pre.2 (PR #119).

## Summary

On a cuda device configured for f64, every `Tensor::matmul` panics: the matmul autotuner
has no candidate that accepts f64, so `cubecl-runtime` panics with "Can't execute the
autotune plan". For a plain 64 × 64 × 64 product all 400 candidates return `Err`. The
same matmul in f32 is correct. The Burn 0.22 line ran f64 matmuls on the same GPU, so
this is a regression for f64 users.

## Environment

| | |
|---|---|
| Burn | `0.23.0-pre.2` (crates.io), features `autodiff, signal, store, train` + `cuda` |
| CubeK | `cubek-matmul 0.4.0-pre.2` |
| CubeCL | `cubecl 0.12.0-pre.2` |
| GPU | NVIDIA A100 80GB PCIe, driver 610.57.04 |
| OS / Rust | Linux 5.14 (aarch64), rustc 1.98.1 |

## Minimal reproduction

```rust
use burn::prelude::*;
use burn::tensor::{FloatDType, TensorData};

let mut device = Device::cuda(0);
device.configure(FloatDType::F64).unwrap();
let a = Tensor::<2>::from_data(TensorData::new(vec![1.0f64; 64 * 64], [64, 64]), &device);
let c = a.clone().matmul(a).into_data(); // panics
```

Output (abridged):

```text
thread 'DSU-0-0' panicked at cubecl-runtime-0.12.0-pre.2/src/tune/schedule.rs:539:17:
Can't execute the autotune plan for key: MatmulTuneKey { elem: Float(F64), k: 64, n: 64, m: 64,
  batches: 1, lhs: OperandKey { kind: Float, layout: K, batching: Shared }, rhs: ... }
...
Error while reading data: ... An error happened during execution
```

The candidates decline with (counts for the 64 × 64 case):

- 192 × "a setting listed for a packed input reads one, and both inputs are float"
- 134 × "the microkernel matmul takes f32, f16, bf16 and i32 operands of one element"
- the rest `Rejected the problem` without a reason.

## Shapes tried

Each case ran in its own process against a host f64 product (a throwaway test,
`tests/zz_f64_matmul_repro.rs`, kept on the VM only):

| lhs × rhs | f64 | f32 |
|---|---|---|
| [64, 64] × [64, 64] | panics | |
| [256, 256] × [256, 256] | panics | |
| [2, 3] × [3, 16] | panics | |
| [2, 2, 3] × [2, 3, 16] | panics | |
| [1, 2, 3] × [2, 3, 16] (lhs batch broadcast) | panics | correct, max error 1.6e-7 |
| [2, 2, 3] × [1, 3, 16] (rhs batch broadcast) | panics | |
| [2, 4, 4, 3] × [1, 1, 3, 16] (a `Linear`) | panics | |
| [1, 64, 64] × [2, 64, 64] | panics | |

## Impact

f64 on cuda is unusable for any model with a matmul. In sciml-rs it blocks the f64
check of Phase 10's Gauss–Newton (`tests/gauss_newton_f64.rs`,
`f64_oracle_and_adjoint_on_cuda`): on the old fork it failed in the FFT
([../fft-f64-kernels-launched-as-f32/](../fft-f64-kernels-launched-as-f32/README.md),
since fixed), now it fails here. The process may also SIGSEGV afterwards
([../../cubecl/autotune-no-candidate-panics/](../../cubecl/autotune-no-candidate-panics/README.md)).

cubek #695 ("test(matmul): skip the f64 auto case on a device without f64") suggests an
f64 auto matmul is meant to work on a device with f64, as an A100 is.

## Workaround

None on 0.23.0-pre.2. f64 work on cuda needs the 0.22 line (Burn 0.22.1 has the f64
FFT fix but not the arbitrary-size Bluestein FFT, so power-of-two grids only).
