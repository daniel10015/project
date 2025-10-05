import os

print(f'Hello from rank {os.environ["SLURM_PROCID"]} \
        (local rank {os.environ["SLURM_LOCALID"]} of Node {os.environ["SLURM_NODEID"]})')

