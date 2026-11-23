# Autotune with no usable candidate panics in a worker thread, then the process may segfault

**Status:** open; not filed upstream yet (2026-11-13). Secondary to
[../../cubek/f64-matmul-declined-on-cuda/](../../cubek/f64-matmul-declined-on-cuda/README.md):
fixing that removes this trigger, but not the failure mode.

## Summary

When every candidate of an autotune plan fails, `cubecl-runtime` panics
(`src/tune/schedule.rs:539`, "Can't execute the autotune plan for key …") on its device
thread (`DSU-0-0`) instead of returning an error. The caller then sees only a generic
read failure, and the test process ended with SIGSEGV.

## Environment

As in the f64 matmul report: Burn 0.23.0-pre.2, `cubecl 0.12.0-pre.2`,
`cubek-matmul 0.4.0-pre.2`, NVIDIA A100 80GB PCIe, driver 610.57.04, Linux 5.14 (aarch64).

## Reproduction

Any f64 matmul on cuda (see the f64 matmul report). From
`cargo test --release --no-default-features --features cuda --test gauss_newton_f64 -- --ignored`:

```text
thread 'DSU-0-0' panicked at cubecl-runtime-0.12.0-pre.2/src/tune/schedule.rs:539:17:
Can't execute the autotune plan for key: MatmulTuneKey { elem: Float(F64), k: 3, n: 16, m: 2, batches: 2, ... }
thread 'f64_oracle_and_adjoint_on_cuda' panicked at tests/gauss_newton_common/mod.rs:51:7:
Error while reading data: use `try_into_data` instead to catch the error at runtime: An error happened during execution
test f64_oracle_and_adjoint_on_cuda ... FAILED
process didn't exit successfully: ... (signal: 11, SIGSEGV: invalid memory reference)
```

## Expected

An error naming the operation and dtype ("no matmul kernel accepts F64 on this device"),
returned to the caller, and a clean exit.
