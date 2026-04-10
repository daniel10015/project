#!/bin/bash
#SBATCH -p htc -q debug
#SBATCH --job-name=nsys-multi
#SBATCH -N 2                    # number of nodes
#SBATCH --ntasks-per-node=1     # number of processes per node (should equal to num of GPUS)
#SBATCH -c 4			# cores per process
#SBATCH --gpus-per-node=1      # GPUs per node
#SBATCH --time=5
#SBATCH --output=./slurm_out/%j.out
#SBATCH --error=./slurm_out/%j.err

module load cuda-12.8.1-gcc-12.1.0

echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# The following environment variables are needed for pytorch
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

OUT_NAME="nsys_multi"

echo "LEADER = $MASTER_ADDR:$MASTER_PORT"
echo "WORLD_SIZE = $WORLD_SIZE"
PATH_TO_NSYS=$(dirname $(dirname $(which nsys)))
	
srun /usr/bin/apptainer exec --nv --bind=$PATH_TO_NSYS:/nsight-systems \
    --env "MASTER_ADDR=$MASTER_ADDR" \
    --env "MASTER_PORT=$MASTER_PORT" \
    --env "WORLD_SIZE=$WORLD_SIZE" \
    /scratch/zyin36/cu13cupti_latest.sif    \
        /nsight-systems/bin/nsys profile \
            --trace=cuda,nvtx \
            --force-overwrite=true \
            --output=./${OUT_NAME}_rank%q{SLURM_PROCID} \
            --export=sqlite \
        python3 resnet_exp.py --get_torch_trace=False
