#!/bin/bash
#SBATCH -q debug                    
#SBATCH --job-name=mp-test
#SBATCH -N 2                   # number of nodes
#SBATCH --ntasks-per-node=2     # number of processes (e.g. torchrun) per node
#SBATCH -c 1			# cores per process
#SBATCH --time=2
#SBATCH --output=mp-test-%j.out

# Activate the environment
module load mamba/latest
source activate py39cupti

# Derive MASTER_ADDR / MASTER_PORT from SLURM
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE = $WORLD_SIZE"


srun python ./srun_test.py

