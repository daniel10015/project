#!/bin/bash
#SBATCH -p htc -q debug
#SBATCH --job-name=nsys-single
#SBATCH -N 1
#SBATCH --ntasks-per-node=2     # number of processes per node (should equal to num of GPUS)
#SBATCH --mem=64GB 
#SBATCH --gres=gpu:a100:1       # GPUs per node
#SBATCH -C a100_80      # GPU type
#SBATCH --time=10
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_out/%j.err

module load cuda-12.8.1-gcc-12.1.0

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

echo "LEADER = $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE = $WORLD_SIZE"
PATH_TO_NSYS=$(dirname $(dirname $(which nsys)))
	
srun /usr/bin/apptainer exec --nv --bind=$PATH_TO_NSYS:/nsight-systems \
    --env "MASTER_ADDR=$MASTER_ADDR" \
    --env "MASTER_PORT=$MASTER_PORT" \
    --env "WORLD_SIZE=$WORLD_SIZE" \
    /scratch/zyin36/cu13cupti_latest.sif  \
        /nsight-systems/bin/nsys profile \
        --trace=cuda,nvtx \
        --force-overwrite=true \
        --output=./single_node_nsys/rank%q{SLURM_PROCID} \
        --export=sqlite \
    python3 resnet_exp.py --get_torch_trace=False # ../nsys/nvtx_basic.py
