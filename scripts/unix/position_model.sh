#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pt-01n40
#SBATCH --output=output/pt_01n40_1024_rerun.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u position_model.py --batch_size=1024 --grid_size=40 --epochs=100 --cell_size=0.1 --grid_folder="grid_01" --save="_1024_rerun" --lr=0.00002 --convolution=1
