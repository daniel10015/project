#!/bin/bash
#SBATCH -q debug                  
#SBATCH --job-name=train-cifar
#SBATCH -N 2                    # number of nodes
#SBATCH --ntasks-per-node=2     # number of processes per node (should equal to num of GPUS)
#SBATCH -c 4			# cores per process
#SBATCH --gpus-per-node=2      # GPUs per node
#SBATCH --time=5

# Activate the environment
module load mamba/latest
source activate cu13cupti

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE = $WORLD_SIZE"

srun python train_cifar.py
