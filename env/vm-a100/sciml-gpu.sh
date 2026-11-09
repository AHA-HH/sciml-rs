# sciml-rs GPU runs (Phase 9 T2 onward). Use: source ~/sciml-gpu.sh
# Driver 610.57 supports CUDA <= 13.3, but /usr/local/cuda is 13.5 and the system
# libnccl is 2.32.3+cuda13.4, so pin CUDA 12.6 and the Spack-built NCCL ahead of them.
. ~/spack/share/spack/setup-env.sh
spack env activate sciml-gpu
export CUDA_PATH=/usr/local/cuda-12.6   # cudarc picks its bindings from this toolkit at build time
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$(spack location -i nccl)/lib:$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
