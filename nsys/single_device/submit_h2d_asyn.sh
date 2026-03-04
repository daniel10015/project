#!/bin/bash
#SBATCH -q debug 
#SBATCH -A grp_ashriva6
#SBATCH --job-name=nsys       
#SBATCH --output=nsys.out         
#SBATCH --error=ddp-%j.err          
#SBATCH -N 1                               
#SBATCH --gres=gpu:1               
#SBATCH --cpus-per-task=10          
#SBATCH --time=10:00

# 1. 환경 설정
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate cu13cupti
module load cuda-12.8.1-gcc-12.1.0

nsys profile -o main_nsys_data_wait_syn_H2D_asyn\
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --export=sqlite \
  python main_nsys_syn_h2d.py

