# Setup

**Please** change if there are any issues with the current setup on SOL.

Your environment name on Sol should be "cu13cupti".
If not, change the .sh script accordingly.

## Setup on SOL
0. Log into Sol (shell), then load up mamba:
    `module load mamba/latest`

1. Create an environment
    `mamba create -n cu13cupti -c conda-forge python=3`

2. Activate your environment 
    `source activate cu13cupti` 
    - NOTE: You should see `(cu13cupti)` to the left of your directory

3. Install dependencies
    - NOTE: Only use pip when an environment is active
   Use pip to install packages. <br>
   To install PyTorch: <br>
    `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130` <br>
    To install CUPTI Python: <br>
    `pip install cupti-python==13.0.0` <br>
4. Verify your setup
   Try running the CIFAR example: <br>
   `sbatch slurm_train_cifar.sh`
   It should output something like: <br>
   ```
   (cu13_testenv) [zyin36@sol-login02:~]$ sbatch slurm_train_cifar.sh
    sbatch: Default partition applied: htc
    sbatch: Default min_mem_per_node applied for GPU: 24000.0 MB
    Submitted batch job <job_id> 
   ```
   Go to your Sol Dashboard, under Jobs > Active jobs, you should see your job entry with the same job id.
   Once it's completed, check your output in Sol shell by running: <br>
   `cat slurm-<job_id>`
