#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pn_tune
#SBATCH --output=output/pn_tune_npoints_random_512.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u pn_tune.py
