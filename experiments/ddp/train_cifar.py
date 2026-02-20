"""
script for each process/task
Reference: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from socket import gethostname
import torch.nn as nn
import torch.optim as optim
import os
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity, record_function


def train_epoch(model, local_rank, train_loader, optimizer, epoch):
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(local_rank), y.to(local_rank)
        optimizer.zero_grad() # clear grad
        y_hat = model(X)      
        loss = loss_fn(y_hat, y)
        loss.backward()      # backprop
        optimizer.step()


def test_model(model, device):
    testset = CIFAR10(root='./datasets', train=False, \
                        download=True, transform=ToTensor())
    loader = DataLoader(testset, batch_size=1024, shuffle=False, pin_memory=True)
    total = 0
    total_correct = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        y_hat = torch.argmax(model(X), dim=1)    
        total_correct += (y_hat == y).sum()
        total += len(y)

    print("Testset Accuracy: ", total_correct / total)


def run(num_epochs, rank, local_rank):
    trainset = CIFAR10(root='./datasets', train=True, \
                        download=True, transform=ToTensor())
    sampler = DistributedSampler(trainset)
    
    device = torch.device('cuda', local_rank)
    model = models.resnet18(num_classes=10).to(device)
    print(f"(Rank {rank}, Local {local_rank}) {device}")
    ddp_model = DDP(model, device_ids=[device])
    
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loader = DataLoader(trainset, batch_size=256, shuffle=False, sampler=sampler,
                        num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]) - 1,
                        pin_memory=True)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        train_epoch(ddp_model, local_rank, loader, optimizer, epoch)
        scheduler.step()

    if rank==0:
        # get test accuracy
        test_model(ddp_model.module, device)
    

if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])

    # MASTER_ADDR and MASTER_PORT are set by the slurm script
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    ## the i-th task in each node uses the i-th gplocal_ranku
    ## (assuming 1 GPU per task)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities) as prof:
        run(num_epochs=10, rank=rank, local_rank=local_rank)
    
    prof.export_chrome_trace(f"trace_{rank}.json")
    # clean up
    dist.destroy_process_group()

