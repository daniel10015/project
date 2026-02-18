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


OUT_NAME="profile_result_resnet_50"


srun --export=ALL nsys profile \
    --trace=cuda,nvtx \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    --output=./${OUT_NAME}_rank%p{SLURM_PROCID} \
    python main_ddp_resnet_50_nsys.py



echo "Exporting to SQLite..."
for FILE in ${OUT_NAME}_rank*.nsys-rep; do
    nsys export --type=sqlite --output="${FILE%.*}.sqlite" "$FILE"
done

echo "Done. SQLite files are ready."
