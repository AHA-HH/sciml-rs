# GPU VM environment (`sciml-gpu`)

The Spack environment for cuda runs on the 2× A100 VM (Phase 9 T2 onward). Copies of
the files on the VM; edit here and copy back.

**Why.** The driver (610.57.04) supports CUDA ≤ 13.3, but `/usr/local/cuda` is 13.5 and
the system `libnccl` is 2.32.3+cuda13.4, so NCCL failed at init. This environment builds
NCCL against the system CUDA 12.6 (used as an external), plus gnuplot for `plots.png`.

**Set up** (once; Spack at `dd20aee`, the develop branch on 2026-11-09):

```sh
git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git ~/spack
. ~/spack/share/spack/setup-env.sh
spack compiler find
spack env create sciml-gpu env/vm-a100/spack.yaml
spack env activate sciml-gpu && spack install -j "$(nproc)"
cp env/vm-a100/sciml-gpu.sh ~/
```

**Use**: `source ~/sciml-gpu.sh` before any `cargo` command. It puts CUDA 12.6 first, which
matters at build time too, because cudarc reads `nvcc --version` when it compiles.

- `spack.yaml`: NCCL for the A100 only (`cuda_arch=80`), without InfiniBand
  (`fabrics=none`: one node, PCIe); gnuplot with the `pngcairo` terminal `save_to_png`
  uses; the system's build tools as externals, which cut the build from 84 packages to 58.
- `sciml-gpu.sh`: activates the environment and puts the Spack NCCL and CUDA 12.6 ahead of
  the system copies in `PATH` and `LD_LIBRARY_PATH`.
