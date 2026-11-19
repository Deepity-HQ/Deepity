#!/bin/bash
#SBATCH --job-name=deepity-cuda-modelio
#SBATCH --account=pas0350
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=cuda-modelio-%j.out
#SBATCH --error=cuda-modelio-%j.err

# Builds Deepity with CUDA on a GPU node and runs the model-IO tests that touch the device.
# Submit with: sbatch tests/cuda_model_io.slurm      (or run the same lines inside `srun --pty` / salloc)
# Module names come from `module spider` on Ascend (there is no openblas module, so this builds with MKL).
cd ~/Jack_rose1775/Deepity || exit 1

# The cmake module (3.25.2) is too old: the project needs 3.26.5+. A user-level PyPI cmake covers it.
module load gcc/13.2.0 cuda/13.2.1 intel-oneapi-mkl/2025.0.1
export PATH="$HOME/.local/bin:$PATH"
uv tool install cmake   # no-op once installed

# The cluster's /apps/python has no development headers, so build in a venv on a uv-managed Python
# (kept apart from .venv, which other scripts use).
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-cuda"
uv python install 3.12
[ -d "$UV_PROJECT_ENVIRONMENT" ] || uv venv --python 3.12 --managed-python "$UV_PROJECT_ENVIRONMENT"
cmake --version | head -1
echo "MKLROOT=$MKLROOT"

export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0

# CUDA build (no --no-cuda). cuDNN is off: these tests are dense networks, and the Ascend CUDA module
# does not put cudnn.h where the build looks. Add -D flags your cluster needs (oneDNN locations).
uv run python build.py Release MKL --jobs 4 -DDEEPITY_USE_CUDNN=OFF || exit 1

echo "=== C++ tests (includes the CUDA ones) ==="
ctest --test-dir build/Release --output-on-failure

echo "=== GPU checkpoint test ==="
uv run python tests/pyGpuCheckpointVerify.py
