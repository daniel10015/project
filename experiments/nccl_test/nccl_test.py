"""
The test script for one process/task
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP

def run(world_size, rank, local_rank):
    x = torch.tensor([1.]).to(f'cuda:{local_rank}')
    # sum up the tensors from all processes
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expected = torch.tensor([float(world_size)], device=f'cuda:{local_rank}')
    # check
    if not torch.allclose(x, expected):
        print(f"[rank {rank} / local {local_rank}] ERROR: got {x.item()}, expected {expected.item()}")
        return False
    else:
        print(f"[rank {rank} / local rank {local_rank}] PASSED: {x.item()} == {expected.item()}")
        return True

if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])
    # MASTER_ADDR and MASTER_PORT are set by the slurm script
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    # the i-th task in each node uses the i-th gpu
    # (assuming 1 GPU per task)
    passed = run(world_size, rank, local_rank)
    # wait for other processes to finish
    dist.barrier()
    # clean up
    dist.destroy_process_group()
    if not passed:
        sys.exit(1) 

