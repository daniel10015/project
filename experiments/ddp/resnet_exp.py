import sys
# from profiler.profiler import *

import argparse
import getpass
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import torch.cuda.nvtx as nvtx  
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.models import resnet50
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.profiler import profile, ProfilerActivity, record_function

from memory_logger_ddp import MemoryLogger

torch.set_float32_matmul_precision('high')

def parse_args():
    parser = argparse.ArgumentParser(description="DDP ResNet50 Training")
    parser.add_argument("--get_torch_trace", action="store_true", \
                        help="Get execution trace from torch.profiler.")
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to run training for")
    parser.add_argument("--trace_file_path", type=str, default=f"/scratch/{getpass.getuser()}", \
                        help="The directory path for torch trace. Default: /scratch/<whoami>")
    parser.add_argument("--batch_size",  type=int, default=128, help="Batch size per GPU")
    return parser.parse_args()

args = parse_args()


class ProfileModel():
    def __init__(self, model, local_rank, global_rank): # metrics: tuple[str, ...]):
        self.model = model
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.step_count = 0
        # self.profiler = profiler(model.forward, metrics)

    
    def train_epoch(self, loader, optimizer, loss_fn, memlog, *args):
        self.model.train()
        iterator = iter(loader)
        num_batches = len(loader)
        for i in range(50):
            print(f"Rank {self.global_rank}: Step {self.step_count}")
            memlog.mark(i, "batch_start",model=self.model, optimizer=optimizer)
            nvtx.range_push(f"Batch_{self.step_count}")
            with nvtx.range("data_wait"):
                data, label = next(iterator)

            with nvtx.range("h2d"):
                data = data.to(self.local_rank, non_blocking=False)
                label = label.to(self.local_rank, non_blocking=False)

            with nvtx.range("gpu_compute"):
                with nvtx.range("zero_grad"):
                    optimizer.zero_grad()
                with nvtx.range("forward"):
                    pred = self.model(data)
                with nvtx.range("loss"):
                    loss = loss_fn(pred, label)
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("opt_step"):
                    optimizer.step()

            nvtx.range_pop() #  f"Batch_{i}"
            memlog.mark(i, "step_end",model=self.model, optimizer=optimizer)
            self.step_count += 1


    # Inference
    def inf_epoch(self, *args):
        pass
    
    
    def forward(self, *args):
        return self.model(*args)

    def __call__(self, *args):
        return self.forward(*args)


def get_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    train_data = CIFAR10(root='/scratch/zyin36/datasets', train=True, download=False, transform=transform)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, 
        sampler=train_sampler, num_workers=2, pin_memory=True
    )
    return train_loader, train_sampler


def get_ddp_train_objs(rank, local_lank, memlog):
    props = torch.cuda.get_device_properties(local_rank)
    free, total = torch.cuda.mem_get_info(local_rank)
    print(f"(Rank {rank}) {props.name} total={total/1e9:.1f}GB free={free/1e9:.1f}GB")

    train_loader, train_sampler = get_dataloader()
    model = resnet50(num_classes=10).to(local_rank)
    # model = torch.compile(model)
    ddp_model = DDP(model, device_ids=[local_rank])
    ddp_model.register_comm_hook(state=None, hook=memlog.make_comm_hook())
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    return train_loader, train_sampler, ddp_model, optimizer, scheduler


def train_epoch(model, local_rank, train_loader, optimizer):
    model.train()
    sum_loss = 0.
    loss_fn = nn.CrossEntropyLoss()
    for batch_idx, (X, y) in enumerate(train_loader):
        X, y = X.to(local_rank), y.to(local_rank)
        optimizer.zero_grad() # clear grad
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        loss.backward()      # backprop
        optimizer.step()
        sum_loss += loss.item()

    return sum_loss


def nsys_run(num_epochs, rank, local_rank, memlog):
    train_loader, train_sampler, ddp_model, optimizer, scheduler = get_ddp_train_objs(rank, local_rank, memlog)
    profile_model = ProfileModel(ddp_model, local_rank, rank)
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        loss = profile_model.train_epoch(train_loader, optimizer, nn.CrossEntropyLoss(), memlog)
        print(f"(Rank {rank}, Local {local_rank}) Epoch {epoch} Loss: {loss}")
        scheduler.step()


def run(num_epochs, rank, local_rank, profiler):
    train_loader, train_sampler, ddp_model, optimizer, scheduler = get_ddp_train_objs(rank, local_rank)
    print(f"(Rank {rank}, Local {local_rank}) started")
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        loss = train_epoch(ddp_model, local_rank, train_loader, optimizer)
        print(f"(Rank {rank}, Local {local_rank}) Epoch {epoch} Loss: {loss}")
        scheduler.step()
        profiler.step()


def init_process(local_rank, rank, world_size):
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])

    # MASTER_ADDR and MASTER_PORT are set by the slurm script
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    print(f'(Rank {rank}, local {local_rank}) init_process')


def clean_up(local_rank, rank):
    dist.destroy_process_group()
    print(f'(Rank {rank}, local {local_rank}) Finished.')


if __name__=="__main__":
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"]) # global rank
    local_rank = int(os.environ["SLURM_LOCALID"])
    init_process(local_rank, rank, world_size)
    if args.get_torch_trace:
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        with profile(activities=activities, \
                schedule=torch.profiler.schedule(wait=0, warmup=2, active=1)) as prof:
            run(num_epochs=args.epochs, rank=rank, local_rank=local_rank, profiler=prof)

        path = str(Path(args.trace_file_path) / f"trace_r{rank}_local{local_rank}.json")
        prof.export_chrome_trace(path)
    else:
        memlog = MemoryLogger(device=local_rank, out_csv="mem_log.csv")
        # nsys output path is set in the sbatch script
        nsys_run(num_epochs=args.epochs, rank=rank, local_rank=local_rank, memlog=memlog)

    clean_up(local_rank, rank)    
    memlog.dump()

