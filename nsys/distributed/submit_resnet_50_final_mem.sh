#!/bin/bash
#SBATCH -q debug 
#SBATCH -A grp_ashriva6
#SBATCH --job-name=ddp-nsys         
#SBATCH --output=ddp-%j.out         
#SBATCH --error=ddp-%j.err     
#SBATCH -N 2                             
#SBATCH --ntasks-per-node=2        
#SBATCH --gres=gpu:2            
#SBATCH --cpus-per-task=10          
#SBATCH --time=10:00



module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate cu13cupti
module load cuda-12.8.1-gcc-12.1.0



export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export OMP_NUM_THREADS=1


# NCCL DEBUG setup

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET
export NCCL_DEBUG_FILE=/home/hson17/nsys/distributed/nccl_log/nccl_rank%q{SLURM_PROCID}.log

BATCH_SIZE=500
IMAGE_SIZE=224
MAX_BATCHES=50

OUT_NAME="profile_result_resnet_50_fin_bs${BATCH_SIZE}_img${IMAGE_SIZE}_mb${MAX_BATCHES}"

OUT_DIR="/home/hson17/nsys/distributed/sqlite"
LOG_DIR="/home/hson17/nsys/distributed/nccl_log"


mkdir -p ${LOG_DIR}


srun --export=ALL nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
   --output=${OUT_DIR}/${OUT_NAME}_rank%q{SLURM_PROCID} \
    python main_ddp_resnet_50_final_mem.py --batch_size ${BATCH_SIZE} --image_size ${IMAGE_SIZE} --max_batches ${MAX_BATCHES}




echo "Exporting to SQLite..."
for FILE in ${OUT_DIR}/${OUT_NAME}_rank*.nsys-rep; do   
    nsys export --type=sqlite --output="${FILE%.*}.sqlite" "$FILE"
done

# NCCL 로그에서 핵심 정보 추출
echo ""
echo "===== NCCL Ring 순서 ====="
grep "Ring" ${LOG_DIR}/nccl_rank*.log

echo ""
echo "===== 통신 알고리즘 ====="
grep "algorithm\|proto" ${LOG_DIR}/nccl_rank*.log

echo ""
echo "===== InfiniBand 사용 확인 ====="
grep "NET/IB\|NET/Socket" ${LOG_DIR}/nccl_rank*.log

echo ""
echo "===== 초기화 완료 시점 ====="
grep "Init Done\|Comm.*Done" ${LOG_DIR}/nccl_rank*.log


echo "Done. SQLite files are ready."
echo "NCCL logs: ${LOG_DIR}/nccl_rank*.log"