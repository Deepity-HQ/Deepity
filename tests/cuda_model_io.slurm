#!/bin/bash
#SBATCH --job-name=deepity-cuda-modelio
#SBATCH --account=pas0350
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --output=cuda-modelio-%j.out
#SBATCH --error=cuda-modelio-%j.err

# Builds Deepity with CUDA on a GPU node and runs the model-IO tests that touch the device.
# Submit with: sbatch tests/cuda_model_io.slurm      (or run the same lines inside `srun --pty` / salloc)
# Module names come from `module spider` on Ascend (there is no openblas module, so this builds with MKL).
cd ~/Jack_rose1775/Deepity || exit 1

module load gcc/13.2.0 cmake/3.25.2 cuda/13.2.1 intel-oneapi-mkl/2025.0.1
echo "MKLROOT=$MKLROOT"

export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0

# CUDA build (no --no-cuda). Add -D flags your cluster needs (oneDNN/OpenBLAS locations).
uv run python build.py Release MKL --jobs 8 || exit 1

echo "=== C++ tests (includes the CUDA ones) ==="
ctest --test-dir build/Release --output-on-failure

echo "=== GPU checkpoint test ==="
uv run python tests/pyGpuCheckpointVerify.py
