#!/bin/bash
#SBATCH --job-name=imagenet-pc
#SBATCH --account=pas0350
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=imagenet-pc-%j.out
#SBATCH --error=imagenet-pc-%j.err

# Navigate to the project directory
cd ~/Jack_rose1775/Deepity || exit 1

# Load CUDA
module load cuda/13.2.1

# Activate the project's virtual environment
source .venv/bin/activate

# Run the experiment
python imagenet_pc_vgg7_ce.py 5 128 7.961890843897934e-05 0.0223381085942421 ckpt_T12 12
python imagenet_pc_vgg7_ce.py 5 128 7.961890843897934e-05 0.0223381085942421 ckpt_T8  8
python imagenet_pc_vgg7_ce.py 5 128 7.961890843897934e-05 0.0223381085942421 ckpt_T6  6
python imagenet_pc_vgg7_ce.py 5 128 7.961890843897934e-05 0.0223381085942421 ckpt_T4  4