#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pt-01n40_tune_windows2
#SBATCH --output=output/pt_01n40_tune_windows2_3.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u tune_convolution.py --batch_size=1024 --grid_size=40 --cell_size=0.1 --grid_folder="grid_01" --tune_folder="windows" --tuning="window"
