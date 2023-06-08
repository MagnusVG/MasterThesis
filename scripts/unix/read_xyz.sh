#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu20
#SBATCH --time=168:00:00
#SBATCH --job-name=read_xyz
#SBATCH --output=output/read_xyz
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda3-py38
conda activate torch
# Run the Python script that uses the GPU
python -u read_xyz.py --felt_akseptert="SVG_havn_akseptert.xyz" --felt_forkastet="SVG_havn_forkastet.xyz" --felt_output="ac_felt3.csv"
