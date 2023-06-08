#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pt-cs01_gs
#SBATCH --output=output/pt_cs01_gs102030_check.out
 
# Activate environmentuenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u tune_cell_grid.py --batch_size=1024 --cell_size=0.1 --cell_str="01" --grid_folder="grid_01"
