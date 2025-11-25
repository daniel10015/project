#!/bin/bash
#SBATCH -q debug 
#SBATCH --job-name=ddp-cupti        
#SBATCH --output=ddp-%j.out         
#SBATCH --error=ddp-%j.err          
#SBATCH -N 1                        # number of nodes                   
#SBATCH --ntasks-per-node=2        # 노드당 Slurm 태스크 개수 (torchrun은 단일 태스크로 실행됨)
#SBATCH --gres=gpu:2                # 노드당 GPU 2개 요청 (DDP를 위해 2개 사용)
#SBATCH --cpus-per-task=10          # 노드당 CPU 코어 요청 (데이터 로딩 num_workers=2에 충분)
#SBATCH --time=5  
#SBATCH --export=ALL                  



# Activate the environment

#module load mamba/latest
#source activate cu13cupti

#mamba install pytorch torchvision -c pytorch -c nvidia

PYTHON="$HOME/.conda/envs/cu13cupti/bin/python"

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

#export WORLD_SIZE= 2
echo "WORLD_SIZE"=$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR
echo "Starting DDP training on MASTER_ADDR: $MASTER_ADDR"

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr


srun $PYTHON main_ddp.py