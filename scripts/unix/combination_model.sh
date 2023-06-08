#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pt-05n1
#SBATCH --output=output/pt_05n1_512_combination.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u combination_model.py --batch_size=512 --grid_size=1 --epochs=100 --cell_size=0.5 --grid_folder="grid_05" --save="_512_combination" --lr=0.0001
