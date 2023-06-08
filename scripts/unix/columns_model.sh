#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pt-felt3_256
#SBATCH --output=output/pt_felt3_felt4_256.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u columns_model.py --batch_size=256 --epochs=60 --save="_felt3_felt4_256"
