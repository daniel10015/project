#!/bin/bash
#SBATCH -q debug                    
#SBATCH --job-name=srun-test
#SBATCH -N 2                    # number of nodes
#SBATCH --ntasks-per-node=2     # number of processes per node
#SBATCH -c 1			# cores per process
#SBATCH --gpus-per-node=2       # GPUs per node
#SBATCH --time=5
#SBATCH --output=srun-test-%j.out

# Activate the environment
module load mamba/latest
source activate cu13cupti

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961

export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
echo "WORLD_SIZE = $WORLD_SIZE"

srun python srun_test.py
