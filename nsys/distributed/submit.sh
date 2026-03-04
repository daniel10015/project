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


# 1. 환경 설정
module load mamba/latest
eval "$(conda shell.bash hook)"
conda activate cu13cupti
module load cuda-12.8.1-gcc-12.1.0


# 2. DDP 설정
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=16961
export OMP_NUM_THREADS=1


OUT_NAME="profile_result"


srun --export=ALL nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output=./${OUT_NAME}_%p \
    python main_ddp_nsys.py


# 4. JSON 변환 (작업이 끝난 후 실행)
# 각 랭크(GPU)별로 생성된 nsys-rep 파일을 json으로 변환합니다.
echo "Exporting to SQLite..."
for FILE in ${OUT_NAME}_*.nsys-rep; do
    nsys export --type=sqlite --output="${FILE%.*}.sqlite" "$FILE"
done

echo "Done. SQLite files are ready."
