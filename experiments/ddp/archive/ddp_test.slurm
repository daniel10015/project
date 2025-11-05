#!/bin/bash
#SBATCH -q debug
#SBATCH --job-name=ddp-test      # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:2             # number of allocated gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load mamba/latest
source activate cu13cupti

srun python train_cifar_ddp.py --epochs=1
