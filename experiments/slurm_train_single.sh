#!/bin/bash
#SBATCH -q public                  
#SBATCH --job-name=train-cifar
#SBATCH -N 1                    # number of nodes
#SBATCH --ntasks-per-node=1     # number of processes per node
#SBATCH -c 1			# cores per process
#SBATCH --gpus-per-node=1       # GPUs per node
#SBATCH --time=20
#SBATCH --output=train-single-%j.out

# Activate the environment
module load mamba/latest
source activate cu13cupti

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE = $WORLD_SIZE"

srun python train_cifar_single.py
