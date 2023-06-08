#!/bin/bash
#SBATCH --gres=gpu:0
#SBATCH --partition=gpuA100 
#SBATCH --time=24:00:00
#SBATCH --job-name=setup
#SBATCH --output=setup.out
 
# Activate environment
uenv verbose cuda-11.1 cudnn-11.x-8.2.1
uenv miniconda3-py38
#conda create --name torch python=3.8
conda activate torch
#conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pandas numpy matplotlib scikit-learn
#conda install -n pytorch_gpu_env pytorch torchvision cuda90 -c pytorch
#conda install -n pytorch_gpu_env pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
#conda install -n pytorch_gpu_env pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
#conda install -n pytorch_gpu_env pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
