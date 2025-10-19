"""
script for each process/task
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def train_epoch(epoch, sampler):
    sampler.set_epoch(epoch)
    loader = Dataloader(dataset, shuffle=False, sampler=sampler)

def run(num_epochs):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, \
                                            download=True, transform=transform)
    sampler = DistributedSampler()
    
    

if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])
    # MASTER_ADDR and MASTER_PORT are set by the slurm script
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    # the i-th task in each node uses the i-th gpu
    # (assuming 1 GPU per task)
    torch.cuda.set_device(local_rank)
    passed = run(world_size, rank, local_rank)
    # wait for other processes to finish
    dist.barrier()
    # clean up
    dist.destroy_process_group()
    if not passed:
        sys.exit(1) 

