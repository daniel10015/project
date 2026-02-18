import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import torch.cuda.nvtx as nvtx  
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR

# ==============================
# Model Definition (NVTX)
# ==============================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, do_1x1=False, block_name="block"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if do_1x1 else None
        self.block_name = block_name

    def forward(self, x):
        with nvtx.range(self.block_name):
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            
            identity = x
            if self.do1x1 is not None:
                identity = self.do1x1(x)
            
            out += identity
            out = self.relu(out)
        return out

class SmallResidualNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(3, stride=2)

        self.block1 = ResidualBlock(8, 16, stride=2, do_1x1=True, block_name="ResBlock1")
        self.block2 = ResidualBlock(16, 32, stride=2, do_1x1=True, block_name="ResBlock2")
        self.block3 = ResidualBlock(32, 64, stride=1, do_1x1=True, block_name="ResBlock3")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        with nvtx.range("Model_Forward"):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        return x

# ==============================
# Utils
# ==============================
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False, 
        sampler=train_sampler, num_workers=2, pin_memory=True
    )
    return train_loader, train_sampler

def train_one_epoch(model, loader, optimizer, loss_fn, local_rank, max_batches=10):
    model.train()

    iterator = iter(loader)

    for i in range(max_batches):
        try:
            nvtx.range_push(f"Batch_{i}")
            try:
                nvtx.range_push("data_wait")
                try:
                    data, label = next(iterator)
                except StopIteration:
                
                    iterator = iter(loader)
                    data, label = next(iterator)
            finally:
                nvtx.range_pop()
            # (2) H2D: Data transfer from Host(CPU) -> Device(GPU)
            nvtx.range_push("h2d")
            try:
                data = data.to(local_rank, non_blocking=True)
                label = label.to(local_rank, non_blocking=True)
            finally:
                nvtx.range_pop()

            # (3) GPU Compute: Total actual model computation
            nvtx.range_push("gpu_compute")
            try:
                # Zero Grad
                nvtx.range_push("zero_grad")
                try:
                    optimizer.zero_grad()
                finally:
                    nvtx.range_pop()

                # Forward
                nvtx.range_push("forward")
                try:
                    pred = model(data)
                finally:
                    nvtx.range_pop()

                # Loss Calculation
                nvtx.range_push("loss")
                try:
                    loss = loss_fn(pred, label)
                finally:
                    nvtx.range_pop()

                # Backward
                nvtx.range_push("backward")
                try:
                    loss.backward()
                finally:
                    nvtx.range_pop()

                # Optimizer Step
                nvtx.range_push("opt_step")
                try:
                    optimizer.step()
                finally:
                    nvtx.range_pop()

            finally:
                nvtx.range_pop()  # End of gpu_compute
            
            nvtx.range_push("nccl_sync")
            try:
                dist.barrier()
            finally:
                nvtx.range_pop()

        finally:
            nvtx.range_pop()      # End of Batch_i

def nvtx_comm_hook(state, bucket):
    # This labels the actual communication chunks in NVTX
    nvtx.range_push("NCCL_AllReduce")
    
    # Perform the actual AllReduce
    fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()
    
    def callback(fut):
        nvtx.range_pop() # Pop when communication is done
        return fut.value()[0]
    
    return fut.then(callback)

def main():
    # 1. DDP Setup
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ.get("SLURM_NTASKS", 4))

    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    # 2. Model & Data
    train_loader, train_sampler = get_dataloaders()
    model = SmallResidualNetwork().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    ddp_model.register_comm_hook(state=None, hook=nvtx_comm_hook)
    
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 3. Training
    if rank == 0:
        print("Starting training...")
    

    
    train_sampler.set_epoch(0)
    train_one_epoch(ddp_model, train_loader, optimizer, loss_fn, local_rank, max_batches=10)

    if rank == 0:
        print("Training Complete.")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()