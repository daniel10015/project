"""
script for each process/task
Reference: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""

import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim


def run(num_epochs, rank, local_rank):
    trainset = CIFAR10(root='./datasets', train=True, \
                        download=True, transform=ToTensor())
    sampler = DistributedSampler(trainset)
    
    device = torch.device('cuda', local_rank)
    model = models.resnet18(num_classes=10).to(device)
    print(f"(Rank {rank}, Local {local_rank}) {device}")
    ddp_model = DDP(model, device_ids=[device])
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        loader = DataLoader(trainset, batch_size=128, shuffle=False, sampler=sampler)

        for X, y in loader:
            y_hat = ddp_model(X)
            y = y.to(local_rank)
            loss_fn(y_hat, y).backward()
            optimizer.step()

    if rank==0:
        # print the accuracy on training set
        loader = DataLoader(trainset, batch_size=128, shuffle=False)
        total = 0
        total_correct = 0
        for X, y in loader:
            y = y.to(device)
            y_hat = torch.argmax(ddp_model(X), dim=1)    
            total_correct += (y_hat == y).sum()
            total += len(y)

        print("Trainset Accuracy: ", total_correct / total)



if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])

    # MASTER_ADDR and MASTER_PORT are set by the slurm script
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    ## the i-th task in each node uses the i-th gpu
    ## (assuming 1 GPU per task)
    # torch.cuda.set_device(local_rank)
    run(num_epochs=1, rank=rank, local_rank=local_rank)

    # wait for other processes to finish
    dist.barrier()
    # clean up
    dist.destroy_process_group()

