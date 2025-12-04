#!/bin/bash
#SBATCH -q debug 
#SBATCH --job-name=ddp-cupti        
#SBATCH --output=ddp-%j.out         
#SBATCH --error=ddp-%j.err          
#SBATCH -N 2                        # number of nodes                   
#SBATCH --ntasks-per-node=2        
#SBATCH --gres=gpu:2              
#SBATCH --cpus-per-task=4        
#SBATCH --time=5  


# Activate the environment

module load mamba/latest
source activate cu13cupti


export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "Starting DDP on MASTER_ADDR: $MASTER_ADDR"



srun python main_ddp.py
