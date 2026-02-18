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
from torchvision import models

# ==============================
# Model Definition (NVTX)
# ==============================


# ==============================
# ResNet-50 Wrapper with NVTX
# ==============================
def get_profiled_resnet50(num_classes=10):
    # Load a standard torchvision ResNet-50
    model = models.resnet50(num_classes=num_classes)

    # Patch the forward method to wrap selected modules with benchmark_ns
    def forward_patched(self, x):
        # Only profile during training (so evaluation stays clean and fast)
        
            # Normal forward path (no NVTX / no timing)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # Bind patched forward to the model instance
    model.forward = forward_patched.__get__(model, models.ResNet)
    
    return model
# ==============================
# Utils
# ==============================
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), #64 -> 224
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
    model = get_profiled_resnet50().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    ddp_model.register_comm_hook(state=None, hook=nvtx_comm_hook)
    
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # 3. Training
    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")
    
    
    train_sampler.set_epoch(0)
    train_one_epoch(ddp_model, train_loader, optimizer, loss_fn, local_rank, max_batches=20)

    if rank == 0:
        print("Training Complete.")
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()