#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=gpuA100
#SBATCH --time=30:00:00
#SBATCH --job-name=pn_npoints_512
#SBATCH --output=output/pn_npoints_512_norm_bs64_es_15p.out
 
# Activate environment
uenv verbose cuda-11.0 cudnn-11.x-8.2.1

uenv miniconda3-py38

conda activate torch

# Run the Python script that uses the GPU
python -u pn_model.py --batch_size=64 --num_points=512 --random_sampling=1 --epochs=100 --save="_512_norm_bs64_es_15p" --normalize=1
