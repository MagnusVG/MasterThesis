#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=cpu20 
#SBATCH --time=168:00:00
#SBATCH --job-name=prep_c1_felt3
#SBATCH --output=output/prep_c1_felt3_part10.out
 
# Activate environment
uenv verbose cuda-11.4 cudnn-11.4-8.2.4
uenv miniconda3-py38
conda activate torch
# Run the Python script that uses the GPU
python -u grid_dataset.py --cell_size=1 --folder="grid_1/parts/ac_felt3" --dataset="ac_felt3_clean.csv" --part=10
