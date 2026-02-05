from time import perf_counter_ns, time_ns

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
from matplotlib.patches import Patch
from time import perf_counter_ns
from torch.optim.lr_scheduler import StepLR
from torchvision.datasets import CIFAR10 
from torchvision import transforms  # updated
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
import pickle

from fvcore.nn import FlopCountAnalysis


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
class ModelProfile:
    def __init__(self, root_class_name):
        self.root_cls = root_class_name
        self.layers = {}
        self.flops_by_module = {} 
        self.tracking = []  # per batch  
        self.timeline = []
        self.base_time = -1


    def forward_benchmark(self, layer, x: torch.Tensor) -> torch.Tensor:
        """
        synchronizes torch.cuda
        """
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        start_time = my_timer()
        if self.base_time == -1:
            self.base_time = time_ns()
        
        ret = layer(x)

        end.record()
        torch.cuda.synchronize()   
        end_time = int(1e6*start.elapsed_time(end)) + start_time
        time_elapsed = end_time - start_time

        name = layer._prof_name 
        print("Name: ", name)
        if name not in self.layers:
            self.layers[name] = {
                "time_ns": [],
                "flops": self.flops_by_module[name],
                "throughput": []  # FLOPs per second
            }

        self.layers[name]["time_ns"].append(time_elapsed)
        fl = self.layers[name]["flops"]
        if fl and fl > 1e-10:
            self.timeline.append((start_time, end_time, fl, name))
        #do_print = False
        #if do_print:
        if fl is not None and time_elapsed > 0:
            t_sec = time_elapsed * 1e-9
            thr = fl / t_sec  # FLOPs / sec
            self.layers[name]["throughput"].append(thr)
            print(
                f'layer {name}: time={time_elapsed/1e6:.3f} ms, '
                f'FLOPs={fl}, throughput={thr/1e12:.3f} TFLOPs'
            )
        else:
            print(f'layer {name}: time={time_elapsed/1e6:.3f} ms (no FLOPs info)')

        return ret

    def set_flops_dict(d):
        self.flops_by_module = d

    
# profile = ModelProfile('SmallResnet')

# ------------------------------------
# Model definition
# ------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, do_1x1=False, block_name=None):
        super().__init__()
        self.profile = None
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if do_1x1 else None
        self.final_relu = nn.ReLU(inplace=True)

        if block_name is not None:
            base = block_name
            self.conv1._prof_name = f"{base}.conv1"
            self.bn1._prof_name   = f"{base}.bn1"
            self.relu1._prof_name = f"{base}.relu1"
            self.conv2._prof_name = f"{base}.conv2"
            self.bn2._prof_name   = f"{base}.bn2"
            if self.conv1x1 is not None:
                self.conv1x1._prof_name = f"{base}.conv1x1"
            self.final_relu._prof_name = f"{base}.final_relu"
        
    def forward(self, X):
        if self.profile is not None:
            x = self.profile.forward_benchmark(self.conv1, X)
            x = self.profile.forward_benchmark(self.bn1, x)
            x = self.profile.forward_benchmark(self.relu1, x)
            x = self.profile.forward_benchmark(self.conv2, x)
            x = self.profile.forward_benchmark(self.bn2, x)
            identity = X
            if self.conv1x1 is not None:
                identity = self.profile.forward_benchmark(self.conv1x1, identity)
            out = identity + x
            out = self.profile.forward_benchmark(self.final_relu, out)
            
            return out

        else:
            x = self.conv1(X)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            identity = X
            if self.conv1x1 is not None:
                identity = self.conv1x1(identity)
            out = identity + x
            out = self.final_relu(out)

            return out

class SmallResidualNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.profile = None
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
        if self.profile is not None:
            X = self.profile.forward_benchmark(self.conv1, X)
            X = self.profile.forward_benchmark(self.bn1, X)
            X = self.profile.forward_benchmark(self.relu1, X)
            X = self.profile.forward_benchmark(self.pool1, X)

            """
            # You can time the block as a whole:
            X = profile.forward_benchmark(self.block1, X)
            X = profile.forward_benchmark(self.block2, X)
            X = profile.forward_benchmark(self.block3, X)
            """
            X = self.block1(X)
            X = self.block2(X)
            X = self.block3(X)
            X = self.profile.forward_benchmark(self.avgpool, X)
            X = X.view(X.size(0), -1)
            X = self.profile.forward_benchmark(self.fc, X)
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

    def set_profile(self, profile):
        self.profile = profile
        self.block1.profile = profile
        self.block2.profile = profile
        self.block3.profile = profile


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

memcpy_info = []
kernel_info = []
def func_buffer_requested():
    buffer_size = 8 * 1024 * 1024  # 8MB buffer
    max_num_records = 0
    return buffer_size, max_num_records

def func_buffer_completed(activities: list):
    for activity in activities:
        # Only handle MEMCPY activities
        # if activity.kind == cupti.ActivityKind.MEMCPY:
        #    memcpy_info.memcpy(activity)
        if activity.kind == cupti.ActivityKind.MEMCPY: 
            memcpy_info.append((activity.start, activity.bytes, activity.copy_kind))
            memcpy_info.append((activity.end, -activity.bytes, activity.copy_kind))
        elif activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
            duration = activity.end - activity.start
            kernel_info.append((activity.name, duration))
def setup_cupti():
    # Start data collection right before the training loop
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    
def finalize_cupti(rank: int):
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    with open(f"memcpy_data_rank_{rank}.csv", "w") as f:
        data = "\n".join([f"{time},{b},{kind}" for time, b, kind in memcpy_info])
        f.write(data)
    with open(f'kernel_duration_rank_{rank}.csv', 'w') as f:
        data = '\n'.join(['kernel_name', 'duration(ns)'] + [f'{name},{duration}' for name, duration in kernel_info])
        f.write(data)


# ------------------------------------
# Plotting
# ------------------------------------
def plot_combined(memcpy_info:list, profile, filename:str=None):
    # ----------------------------------------
    # MEMORY COPY PREP
    # ----------------------------------------
    memcpy_info.sort(key=lambda x: x[0])
    times, sizes, kinds = zip(*memcpy_info)

    timeline = sorted(profile.timeline, key=lambda x: x[0])
    t0 = timeline[0][0]

    print(f'smallest memcpy time: {np.min(times)}, smallest throughput time: {profile.base_time}')
    smallest = min(np.min(times), profile.base_time)

    times = np.array(times)
    times = times - smallest
    times_scaled, time_units = scale_time_units(times_ns=times)

    utilization = np.cumsum(np.array(sizes))

    memcpy_colors = {
        "Host -> Device": "tab:green",
        "Device -> Host": "tab:blue",
        "Other": "tab:gray"
    }

    # ----------------------------------------
    # THROUGHPUT PREP
    # ----------------------------------------
    dif = 0 if abs(profile.base_time - smallest) < 1e-10 else profile.base_time - smallest

    events = []
    for start, end, flops, layer in timeline:
        start = start - t0 + dif
        end   = end - t0 + dif
        dur = end - start
        thr = flops / dur if (flops and dur > 0) else 0

        # scale throughput for visibility
        events.append((start, dur, thr, layer))

    # layers in first appearance order
    layers = []
    for _, _, _, layer in events:
        if layer not in layers:
            layers.append(layer)

    cmap = plt.get_cmap("tab20")
    layer_colors = {layer: cmap(i % 20) for i, layer in enumerate(layers)}

    # Convert event times to the same units as memory
    def convert_to_units(ns):
        return ns / (times[1] - times[0]) * (times_scaled[1] - times_scaled[0])

    event_plot = []
    for start, dur, thr, layer in events:
        event_plot.append((
            start * (times_scaled[-1] / times[-1]), 
            dur   * (times_scaled[-1] / times[-1]),
            thr,
            layer
        ))

    # ----------------------------------------
    # FIGURE SETUP
    # ----------------------------------------
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # MEMORY COPY PLOT (LEFT Y)
    for kind_str in ["Host -> Device", "Device -> Host", "Other"]:
        mask = np.array([
            (MEMCPY_KIND_STR.get(k, "Other") == kind_str)
            if k in MEMCPY_KIND_STR else False
            for k in kinds
        ])
        if not np.any(mask):
            continue

        ax1.step(
            times_scaled[mask],
            utilization[mask],
            where="post",
            lw=2,
            color=memcpy_colors[kind_str],
            label=f"Memcpy: {kind_str}",
        )

    ax1.set_xlabel(f"Time ({time_units})")
    ax1.set_ylabel("Memory Utilization (bytes)")
    ax1.set_yscale("log")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ----------------------------------------
    # THROUGHPUT PLOT (RIGHT Y)
    # ----------------------------------------
    ax2 = ax1.twinx()
    ax2.set_ylabel("Throughput (GFLOPs)")

    max_thr = max(thr for _, _, thr, _ in event_plot)
    label_offset_y = max_thr * 0.05

    # Draw throughput blocks (rectangles)
    for idx, (start, dur, thr, layer) in enumerate(event_plot):
        ax2.barh(
            y=thr / 2,
            width=dur,
            left=start,
            height=thr,
            color=layer_colors[layer],
            edgecolor='black',
            alpha=0.85
        )

        # index label
        cx = start + dur / 2
        ax2.text(
            cx,
            thr + label_offset_y,
            str(idx%len(layers)),
            ha='center',
            va='bottom',
            fontsize=8,
            color='black'
        )

    ax2.set_ylim(bottom=0)

    # ----------------------------------------
    # LEGENDS
    # ----------------------------------------
    h1, l1 = ax1.get_legend_handles_labels()

    layer_patches = [
        Patch(facecolor=layer_colors[layer], label=f"({i}) {layer}")
        for i, layer in enumerate(layers)
    ]

    ax1.legend(h1, l1, loc="upper left", title="Memory Copies")
    ax2.legend(layer_patches, layers, loc="upper right", title="Layers")

    plt.title("Memory Copy Timeline + Throughput Timeline (Combined)")
    plt.tight_layout()
    # save plot image
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.savefig(f"combined_rank{int(os.environ["SLURM_PROCID"])}.png")

def plot_memcpy_timeline(memcpy_info:list, filename:str=None): #, plot=False):
    # Sort events by time
    memcpy_info.sort(key=lambda x: x[0])

    # Compute cumulative utilization over time
    times, sizes, kinds, device_ids = zip(*memcpy_info)
    times = np.array(times)
    times = times - np.min(times) # offset to 0
    times, units = scale_time_units(times_ns=times)
    deltas = np.array(sizes)
    utilization = np.cumsum(deltas)

    # Convert to unit
    utilization_unit = utilization

    # Define colors
    colors = {
        "Host -> Device": "tab:green",   # CPU → GPU
        "Device -> Host": "tab:blue",     # GPU → CPU
        "Other": "tab:gray"
    }

    # Split the data into segments by kind
    plt.figure(figsize=(9, 5))
    for kind_str in ["Host -> Device", "Device -> Host", "Other"]:
        mask = np.array([
            (MEMCPY_KIND_STR.get(k, "Other") == kind_str)
            if k in MEMCPY_KIND_STR else False
            for k in kinds
        ])
        if not np.any(mask):
            continue
        plt.step(times[mask], utilization_unit[mask], where="post",
                lw=2, label=kind_str, color=colors[kind_str])

    plt.fill_between(times, utilization_unit, step="post", alpha=0.2, color="lightgray")
    plt.xlabel(f"Time ({units})")
    plt.ylabel("bytes")
    plt.yscale('log')
    plt.title("Memory Copies Over Time (log-scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    # save plot image
    if filename:
        plt.savefig(filename)
    
    """
    if plot:    # if using in GUI or ipynb, for example
        plt.show()
    """

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

# def run_one_batch()
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
    # run(num_epochs=, local_rank=local_rank, train_sampler=train_sampler, loader=train_loader)
        # module FLOPs calculation (1 times)
    model = SmallResidualNetwork(num_classes=10).to(local_rank)
    X_example = next(iter(train_loader))[0].to(local_rank)

    fa = FlopCountAnalysis(model, X_example)

    profile = ModelProfile('SmallResNet')
    profile.flops_by_module = dict(fa.by_module())
    model.set_profile(profile)
    ddp_model = DDP(model, device_ids=[local_rank]) 

    ddp_model(X_example)

    print("FLOPs by module:", profile.flops_by_module)
    print("Layers: ", profile.layers)

    total_ns = my_timer() - start_time

    # ---- CUPTI end ------
    # 6. Each rank saves its own data to file
    finalize_cupti(rank)
    plot_combined(memcpy_info, profile)

    # 8. Destroy DDP process group
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
