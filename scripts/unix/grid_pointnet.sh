#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu36 
#SBATCH --time=96:00:00
#SBATCH --job-name=grid_felt3_512
#SBATCH --output=output/grid_felt3_512_part5.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda3-py38
conda activate torch

# Run the Python script that uses the GPU
python -u grid_pointnet.py --num_points=512 --random_sampling=1 --part=5
