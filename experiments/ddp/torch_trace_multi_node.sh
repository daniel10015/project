#!/bin/bash
#SBATCH -p htc -q debug
#SBATCH --job-name=torch-trace-multi-node
#SBATCH -N 2                    # number of nodes
#SBATCH --ntasks-per-node=1     # number of processes per node (should equal to num of GPUS)
#SBATCH -c 4			# cores per process
#SBATCH --mem=64GB 
#SBATCH --gres=gpu:a100:1       # GPUs per node
#SBATCH -C a100_80
#SBATCH --time=10
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_out/%j.err

# Activate the environment
module load cuda-12.8.1-gcc-12.1.0

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "WORLD_SIZE = $WORLD_SIZE"

srun /usr/bin/apptainer exec --nv \
        --env "MASTER_ADDR=$MASTER_ADDR" \
        --env "MASTER_PORT=$MASTER_PORT" \
        --env "WORLD_SIZE=$WORLD_SIZE" \
        /scratch/zyin36/cu13cupti_latest.sif  \
    python3 resnet_exp.py --get_torch_trace --trace_file_path=/scratch/zyin36/multi_node_trace
