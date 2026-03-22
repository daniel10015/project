import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os
import torch.cuda.nvtx as nvtx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10
from torchvision import transforms, models
from memory_logger import MemoryLogger


# ==============================
# ResNet-50
# ==============================
def get_profiled_resnet50(num_classes=10):
    model = models.resnet50(num_classes=num_classes)

    def forward_patched(self, x):
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

    model.forward = forward_patched.__get__(model, models.ResNet)
    return model


# ==============================
# DataLoader
# ==============================
def get_dataloaders(batch_size=1024, image_size=224, repeat=10):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Expand dataset by repeat times using ConcatDataset -> 50,000 * repeat samples
    train_data = ConcatDataset([
        CIFAR10(root='./data', train=True, download=True, transform=transform)
        for _ in range(repeat)
    ])

    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=False,
        sampler=train_sampler, num_workers=2, pin_memory=True
    )
    return train_loader, train_sampler


# ==============================
# NCCL comm hook (NVTX)
# ==============================
def nvtx_comm_hook(state, bucket):
    nvtx.range_push("NCCL_AllReduce")
    fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

    def callback(fut):
        nvtx.range_pop()
        return fut.value()[0]

    return fut.then(callback)


# ==============================
# Training Loop
# ==============================
def train_one_epoch(model, loader, optimizer, loss_fn,
                    local_rank, memlog, max_batches=10):
    model.train()
    iterator = iter(loader)

    for i in range(max_batches):
        nvtx.range_push(f"Batch_{i}")
        try:
            memlog.reset_step_peak()
            memlog.mark(i, "batch_start")

            # (1) Data Wait
            nvtx.range_push("data_wait")
            try:
                try:
                    data, label = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    data, label = next(iterator)
            finally:
                nvtx.range_pop()
            memlog.mark(i, "after_data_wait")

            # (2) H2D: Host to Device
            nvtx.range_push("h2d")
            try:
                data  = data.to(local_rank, non_blocking=True)
                label = label.to(local_rank, non_blocking=True)
            finally:
                nvtx.range_pop()
            memlog.mark(i, "after_h2d")

            # (3) GPU Compute
            nvtx.range_push("gpu_compute")
            try:
                nvtx.range_push("zero_grad")
                try:
                    optimizer.zero_grad()
                finally:
                    nvtx.range_pop()
                memlog.mark(i, "before_forward")

                nvtx.range_push("forward")
                try:
                    pred = model(data)
                finally:
                    nvtx.range_pop()
                memlog.mark(i, "after_forward")

                nvtx.range_push("loss")
                try:
                    loss = loss_fn(pred, label)
                finally:
                    nvtx.range_pop()
                memlog.mark(i, "after_loss")

                nvtx.range_push("backward")
                try:
                    loss.backward()  # NCCL AllReduce happens automatically here
                finally:
                    nvtx.range_pop()
                memlog.mark(i, "after_backward")

                nvtx.range_push("opt_step")
                try:
                    optimizer.step()
                finally:
                    nvtx.range_pop()
                memlog.mark(i, "after_opt_step")

            finally:
                nvtx.range_pop()  # gpu_compute

            # (4) NCCL Sync: intentionally added to visualize rank synchronization in the graph
            nvtx.range_push("nccl_sync")
            try:
                dist.barrier()
            finally:
                nvtx.range_pop()

        finally:
            nvtx.range_pop()  # Batch_i

        memlog.mark(i, "step_end")


# ==============================
# Argparse
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_batches", type=int, default=20,
                        help="max number of batches to train (default: 20)")
    parser.add_argument("--image_size",  type=int, default=224,
                        help="image size for transforms.Resize (default: 224)")
    parser.add_argument("--batch_size",  type=int, default=1024,
                        help="batch size (default: 1024)")
    parser.add_argument("--repeat",      type=int, default=10,
                        help="ConcatDataset repeat count, 50000 * repeat total samples (default: 10)")
    return parser.parse_args()


# ==============================
# Main
# ==============================
def main():
    args = parse_args()

    # DDP Setup
    rank       = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ.get("SLURM_NTASKS", 4))

    dist.init_process_group(
        backend='nccl', init_method='env://',
        rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[Config] max_batches={args.max_batches}, image_size={args.image_size}, "
              f"batch_size={args.batch_size}, repeat={args.repeat}")
        print(f"[Config] total dataset size: {50000 * args.repeat:,} samples")
        batch_mem_gb = args.batch_size * 3 * args.image_size * args.image_size * 4 / 1024**3
        print(f"[Config] estimated batch memory: {batch_mem_gb:.2f} GB")

    memlog = MemoryLogger(device=local_rank, out_csv="mem_log.csv")

    # Model & Data
    train_loader, train_sampler = get_dataloaders(
        batch_size=args.batch_size,
        image_size=args.image_size,
        repeat=args.repeat
    )
    model     = get_profiled_resnet50().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    ddp_model.register_comm_hook(state=None, hook=nvtx_comm_hook)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    loss_fn   = nn.CrossEntropyLoss()

    # Training
    if rank == 0:
        print(f"Starting training on {world_size} GPUs...")

    train_sampler.set_epoch(0)
    train_one_epoch(
        ddp_model, train_loader, optimizer, loss_fn,
        local_rank, memlog, max_batches=args.max_batches
    )

    memlog.dump()

    if rank == 0:
        print("Training Complete.")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()