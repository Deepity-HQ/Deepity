#!/bin/bash
#SBATCH --job-name=deepity-mnist-dkppcn
#SBATCH --account=pas0350
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# Slurm batch script for running Deepity's mnist.py (DKPPCN) on an A100
# GPU on OSC's Ascend cluster. Matches everything confirmed working
# tonight: cuda/12.9.1 (12.4.1 hit a real, documented nvcc bug parsing
# GCC 11's amxtileintrin.h -- see include-patches/ or the earlier fix
# discussion if this resurfaces on a different node/module set), the
# project's .venv for python3.12, account pas0350.
#
# Submit with: sbatch run_mnist_gpu.slurm
# Check status: squeue -u $USER
# Watch output live: tail -f slurm-<jobid>.out

set -euo pipefail

module load cuda/12.9.1

cd "$SLURM_SUBMIT_DIR"

# Activate the project's venv (contains python3.12 + nanobind + the
# built pydeepity extension module).
source .venv/bin/activate

echo "=== Job started on $(hostname) at $(date) ==="
echo "=== GPU visible to this job: ==="
nvidia-smi --query-gpu=name,memory.total --format=csv

python mnist.py

echo "=== Job finished at $(date) ==="
