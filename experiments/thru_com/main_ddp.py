import argparse
import torch
import torch.distributed as dist  # DDP 
import torch.multiprocessing as mp
from socket import gethostname
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter_ns
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10 
from torchvision import transforms  # updated
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import pickle



# CUPTI-related import assumes that the cupti library is installed in the environment
try:
    from cupti import cupti
except ImportError:
    print("Warning: CUPTI import failed. Profiling utilities may not work.")
    class MockCUPTI:
        class ActivityKind:
            MEMCPY = 0
        def activity_register_callbacks(self, *args): pass
        def activity_enable(self, *args): pass
        def activity_flush_all(self, *args): pass
        def activity_disable(self, *args): pass
    cupti = MockCUPTI()

# ... (ResidualBlock, SmallResidualNetwork, CUPTI Utils, get_dataloaders_for_ddp from previous code are kept with updates) ...

my_timer = perf_counter_ns
PINNED_MEMORY = True
PREFETCH_FACTOR = 3
PREFETCH_FACTOR = 3

# ------------
# utils
# ------------
def scale_time_units(times_ns):
    """
    Scales time values so that they are expressed in the largest possible unit
    (ns, µs, ms, s) such that the values remain > 0.
    """
    units = ["ns", "µs", "ms", "s"]
    times = np.array(times_ns, dtype=float)
    idx = 0

    while np.max(times) > 1e4 and idx < len(units) - 1:
        times *= 1e-3
        idx += 1

    return times, units[idx]

# ------------------------------------
# timer for each module and FLOPs calculator
# ------------------------------------
class ExtractModel:
    def __init__(self, root_class_name):
        self.root_cls = root_class_name
        self.layers = {}
        self.flops_by_module = {} 
        self.tracking = []  # per batch  
        self.timeline = []
        self.base_time = 0

    def benchmark_ns(self, func, *args):
        # This wrapper is for timing; currently just calls the function
        ret = func(*args)
        return ret


profile = ExtractModel('SmallResnet')

# ------------------------------------
# Model definition
# ------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, do_1x1=False, block_name=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if do_1x1 else None
        self.final_relu = nn.ReLU(inplace=True)

        if block_name is not None:
            base = block_name
            self.conv1._prof_name = f"{base}.conv1"
            self.bn1._prof_name   = f"{base}.bn1"
            self.relu1._prof_name = f"{base}.relu1"
            self.conv2._prof_name = f"{base}.conv2"
            self.bn2._prof_name   = f"{base}.bn2"
            if self.do1x1 is not None:
                self.do1x1._prof_name = f"{base}.do1x1"
            self.final_relu._prof_name = f"{base}.final_relu"

    def forward(self, X):
        if self.training:
            x = profile.benchmark_ns(self.conv1, X)
            x = profile.benchmark_ns(self.bn1, x)
            x = profile.benchmark_ns(self.relu1, x)
            x = profile.benchmark_ns(self.conv2, x)
            x = profile.benchmark_ns(self.bn2, x)
            identity = X
            if self.do1x1 is not None:
                identity = profile.benchmark_ns(self.do1x1, identity)
            out = identity + x
            out = profile.benchmark_ns(self.final_relu, out)
            return out
        else:
            x = self.conv1(X)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            identity = X
            if self.do1x1 is not None:
                identity = self.do1x1(identity)
            return self.final_relu(identity + x)

class SmallResidualNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(3, stride=2)

        self.block1 = ResidualBlock(8, 16, stride=2, do_1x1=True, block_name="block1")
        self.block2 = ResidualBlock(16, 32, stride=2, do_1x1=True, block_name="block2")
        self.block3 = ResidualBlock(32, 64, stride=1, do_1x1=True, block_name="block3")

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # safer than fixed 8x8
        self.fc = nn.Linear(64, num_classes)

        self.conv1._prof_name  = "conv1"
        self.bn1._prof_name    = "bn1"
        self.relu1._prof_name  = "relu1"
        self.pool1._prof_name  = "pool1"
        self.avgpool._prof_name = "avgpool"
        self.fc._prof_name     = "fc"

    def forward(self, X):
        if self.training:
            X = profile.benchmark_ns(self.conv1, X)
            X = profile.benchmark_ns(self.bn1, X)
            X = profile.benchmark_ns(self.relu1, X)
            X = profile.benchmark_ns(self.pool1, X)

            # You can either time the block as a whole:
            X = profile.benchmark_ns(self.block1, X)
            X = profile.benchmark_ns(self.block2, X)
            X = profile.benchmark_ns(self.block3, X)

            X = profile.benchmark_ns(self.avgpool, X)
            X = X.view(X.size(0), -1)
            X = profile.benchmark_ns(self.fc, X)
        else:
            X = self.conv1(X)
            X = self.bn1(X)
            X = self.relu1(X)
            X = self.pool1(X)
            X = self.block1(X)
            X = self.block2(X)
            X = self.block3(X)
            X = self.avgpool(X)
            X = X.view(X.size(0), -1)
            X = self.fc(X)

        return X


# ==============================
# CUPTI MEMCPY collect util
# ==============================
# ----- setup memory transfer callbacks -----
debug = False
MEMCPY_KIND_STR = {
    0: "Unknown",
    1: "Host -> Device",
    2: "Device -> Host",
    3: "Host -> Array",
    4: "Array -> Host",
    5: "Array -> Array",
    6: "Array -> Device",
    7: "Device -> Array",
    8: "Device -> Device",
    9: "Host -> Host",
    10: "Peer -> Peer",
    2147483647: "FORCE_INT"
}
class MemoryCopy:
    def __init__(self):
        self.memcpy_info = []
  
    def memcpy(self, activity) -> str:
        if debug:
            print(f'activity at ({activity.start}) copies {activity.bytes} bytes for {activity.end-activity.start}ns')
        # record start and end of memcpy with positive/negative bytes
        self.memcpy_info.append((activity.start, activity.bytes, activity.copy_kind))
        self.memcpy_info.append((activity.end, -activity.bytes, activity.copy_kind))

memcpy_info = MemoryCopy()

def func_buffer_requested():
    buffer_size = 8 * 1024 * 1024  # 8MB buffer
    max_num_records = 0
    return buffer_size, max_num_records

def func_buffer_completed(activities: list):
    for activity in activities:
        # Only handle MEMCPY activities
        if activity.kind == cupti.ActivityKind.MEMCPY:
            memcpy_info.memcpy(activity)

def setup_cupti():
    # Start data collection right before the training loop
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)

def finalize_cupti(rank: int):
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)

    with open(f"memcpy_data_rank_{rank}.pkl", "wb") as f:
        pickle.dump(memcpy_info.memcpy_info, f)


# ------------------------------------
# Dataloader util (no special changes)
# ------------------------------------
def get_dataloaders_for_ddp(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    batch_size = 64
    train_data = CIFAR10(root='./', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='./', train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=False,  
        sampler=train_sampler,
        num_workers=2, 
        pin_memory=PINNED_MEMORY, 
        prefetch_factor=PREFETCH_FACTOR
    )
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    if dist.get_rank() == 0:
        print("Number of train batches:", len(train_loader))
        print("Number of test batches:", len(test_loader))
        
    return train_loader, test_loader, train_sampler


# ==============================
# Training / Validation 
# ==============================

#Update: added max_batches argument and fixed local_rank typo
def train_one_epoch(model, local_rank, train_loader, optimizer, loss_fn, max_batches=None):
    model.train(True)

    for i, (data, label) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break
        # Forward/Backward pass
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        pred = model(data)
        val = loss_fn(pred, label)
        val.backward()
        optimizer.step()

# Update: removed rank argument (run function only needs local_rank)
def run(num_epochs, local_rank, train_sampler, loader):

    if not torch.cuda.is_available():
        if dist.get_rank() == 0:
            print('Warning: CUDA is not available! Using CPU instead')

    device = torch.device('cuda', local_rank)
    model = SmallResidualNetwork(num_classes=10).to(device)

    if dist.get_rank() == 0:
        print(f"(Rank {dist.get_rank()}, Local {local_rank}) Device: {device}")

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank])  # use local_rank

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        #Update: use train_sampler instead of sampler
        train_sampler.set_epoch(epoch)
        # Call train_one_epoch
        train_one_epoch(ddp_model, local_rank, loader, optimizer, loss_fn)
        scheduler.step()

def main():

    # 1. Get DDP information from environment variables

    
    RANK = int(os.environ["SLURM_PROCID"])
    LOCAL_RANK = int(os.environ["SLURM_LOCALID"])
     # world_size
    if "SLURM_NTASKS" in os.environ:
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        world_size = 2

    # Reassign to lowercase variable names for consistency
    rank = RANK
    local_rank = LOCAL_RANK
    #world_size = WORLD_SIZE

    # 2. Initialize DDP process group (using the correct variables)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # 3. Explicitly set GPU device (using the correct variable)
    torch.cuda.set_device(local_rank)

    # 4. Create DataLoader and Sampler instances
    batch_size = 64
    train_loader, _, train_sampler = get_dataloaders_for_ddp(batch_size=batch_size)

    # ---- CUPTI start -----
    setup_cupti()

    start_time = my_timer()

    # 5. Call the training function
    run(num_epochs=2, local_rank=local_rank, train_sampler=train_sampler, loader=train_loader)

    total_ns = my_timer() - start_time

    # ---- CUPTI end ------
    # 6. Each rank saves its own data to file
    finalize_cupti(rank)
    
    # 7. Rank 0 gathers and prints data
    if rank == 0:
        all_data = []
        for i in range(world_size): 
            try:
                #  Update: use world_size to load files from all ranks
                with open(f"memcpy_data_rank_{i}.pkl", "rb") as f:
                    all_data.extend(pickle.load(f))
                os.remove(f"memcpy_data_rank_{i}.pkl")  # clean up files
            except FileNotFoundError:
                print(f"Warning: Could not find memcpy data for Rank {i}")

        combined_memcpy_info = MemoryCopy()
        combined_memcpy_info.memcpy_info = all_data

        print(f"Training complete. Took {total_ns/1e9:.3f} s")
        # Call CUPTI plot here if needed
        # plot_memcpy_timeline(combined_memcpy_info) 

    # 8. Destroy DDP process group
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
