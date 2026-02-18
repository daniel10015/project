import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
from time import perf_counter_ns, time_ns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ==============================
# Configuration & Globals
# ==============================
my_timer = perf_counter_ns
PINNED_MEMORY = True
PREFETCH_FACTOR = 3

# ==============================
# Profiler: ExtractModel
# ==============================
class ExtractModel:
    def __init__(self, root_class_name):
        # Root model name (for grouping / metadata)
        self.root_cls = root_class_name

        # Per-layer stats (e.g., {"stem.conv1": {"time_ns": [...], "flops": None}, ...})
        self.layers = {}

        # Optional: store FLOPs per module (not used yet in this snippet)
        self.flops_by_module = {}

        # Optional: generic tracking list (not used yet in this snippet)
        self.tracking = []

        # Optional: timeline events list (not used yet in this snippet)
        self.timeline = []

        # Base timestamp to make all events relative to a common start
        self.base_time = 0

    def benchmark_ns(self, func, *args):
        # 0) Resolve a human-readable name for this module
        # If the module has _prof_name, use it; otherwise use the class name
        name = getattr(func, "_prof_name", func.__class__.__name__)

        # 1) Create CUDA Events to measure GPU time for this operation
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)

        # Initialize a base time once (useful if you later build a global timeline)
        if self.base_time == 0:
            self.base_time = time_ns()

        # Record a CPU-side timestamp (your helper timer)
        start_time = my_timer()

        # Record the start CUDA event on the current CUDA stream
        start_evt.record()

        # 2) NVTX range around the module call
        # This will show up in Nsight Systems / NVTX timeline
        torch.cuda.nvtx.range_push(name)
        try:
            ret = func(*args)
        finally:
            # Make sure we always close the NVTX range even if an error happens
            torch.cuda.nvtx.range_pop()

        # Record the end CUDA event
        end_evt.record()

        # Synchronize so the GPU work is finished before we read elapsed_time
        # (Without this, timing can be incorrect because work may still be running.)
        torch.cuda.synchronize()

        # 3) Compute elapsed time
        # elapsed_time returns milliseconds (ms) between the CUDA events
        elapsed_ms = start_evt.elapsed_time(end_evt)

        # Convert ms -> ns (1 ms = 1e6 ns)
        time_elapsed = int(elapsed_ms * 1e6)

        # Derive an end_time in the same unit as start_time (based on your timer)
        end_time = start_time + time_elapsed

        # 4) Log timing into self.layers
        # Store all measurements in a list to compute avg/min/max later
        if name not in self.layers:
            self.layers[name] = {"time_ns": [], "flops": None}
        self.layers[name]["time_ns"].append(time_elapsed)

        return ret

profile = ExtractModel('ResNet50_Profiling')

# ==============================
# ResNet-50 Wrapper with NVTX
# ==============================
def get_profiled_resnet50(num_classes=10):
    # Load a standard torchvision ResNet-50
    model = models.resnet50(num_classes=num_classes)

    # Assign profiler-friendly names to early "stem" layers
    model.conv1._prof_name = "stem.conv1"
    model.bn1._prof_name = "stem.bn1"
    model.relu._prof_name = "stem.relu"
    model.maxpool._prof_name = "stem.pool"

    # Iterate through the 4 ResNet stages (layer1..layer4) and inject names
    for i, layer_group in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], 1):
        layer_group._prof_name = f"Stage_{i}"

        for j, block in enumerate(layer_group):
            block_name = f"Stage_{i}.Block_{j}"
            block._prof_name = block_name

            # If you want finer granularity, name the internal convs too
            block.conv1._prof_name = f"{block_name}.c1_1x1"
            block.conv2._prof_name = f"{block_name}.c2_3x3"
            block.conv3._prof_name = f"{block_name}.c3_1x1"

            # Some blocks have a downsample path (skip connection projection)
            if block.downsample is not None:
                block.downsample._prof_name = f"{block_name}.downsample"

    # Name the head modules
    model.avgpool._prof_name = "global_avgpool"
    model.fc._prof_name = "classifier_fc"

    # Patch the forward method to wrap selected modules with benchmark_ns
    def forward_patched(self, x):
        # Only profile during training (so evaluation stays clean and fast)
        if self.training:
            # Stem
            x = profile.benchmark_ns(self.conv1, x)
            x = profile.benchmark_ns(self.bn1, x)
            x = profile.benchmark_ns(self.relu, x)
            x = profile.benchmark_ns(self.maxpool, x)

            # Stages (each stage is a Sequential of bottleneck blocks)
            x = profile.benchmark_ns(self.layer1, x)
            x = profile.benchmark_ns(self.layer2, x)
            x = profile.benchmark_ns(self.layer3, x)
            x = profile.benchmark_ns(self.layer4, x)

            # Head
            x = profile.benchmark_ns(self.avgpool, x)
            x = torch.flatten(x, 1)
            x = profile.benchmark_ns(self.fc, x)

        else:
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
# Data Utilities
# ==============================
def get_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # ResNet-50은 224x224에서 더 명확한 특징을 보임
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=PINNED_MEMORY, persistent_workers=True,
        prefetch_factor=PREFETCH_FACTOR
    )
    return train_loader

class CUDAPrefetcher:
    def __init__(self, loader, device="cuda"):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_inputs = None
        self.next_labels = None
        self._preload()

    def _preload(self):
        try:
            torch.cuda.nvtx.range_push("pre_data_loading")
            inputs, labels = next(self.loader)
            torch.cuda.nvtx.range_pop()
        except StopIteration:
            self.next_inputs = None
            self.next_labels = None
            return

        with torch.cuda.stream(self.stream):
            torch.cuda.nvtx.range_push("h2d_async_prefetch")
            self.next_inputs = inputs.to(self.device, non_blocking=True)
            self.next_labels = labels.to(self.device, non_blocking=True)
            torch.cuda.nvtx.range_pop()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        inputs = self.next_inputs
        labels = self.next_labels
        self._preload()
        return inputs, labels

# ==============================
# Training Loop
# ==============================
def train_one_epoch_prefetch(model, train_loader, optimizer, loss_fn, device, max_batches=20):
    model.train()
    prefetcher = CUDAPrefetcher(train_loader, device=device)

    for i in range(max_batches):
        torch.cuda.nvtx.range_push(f"step_{i:04d}")
        try:
            # 1. Data Wait
            torch.cuda.nvtx.range_push("data_wait")
            data, label = prefetcher.next()
            torch.cuda.nvtx.range_pop()

            if data is None: break

            # 2. GPU Compute
            torch.cuda.nvtx.range_push("gpu_compute")
            optimizer.zero_grad(set_to_none=True)
            
            # Forward (Profiled internally)
            pred = model(data)
            
            # Loss & Backward
            torch.cuda.nvtx.range_push("loss_backward")
            loss = loss_fn(pred, label)
            loss.backward()
            torch.cuda.nvtx.range_pop()
            
            # Step
            torch.cuda.nvtx.range_push("opt_step")
            optimizer.step()
            torch.cuda.nvtx.range_pop()
            
            torch.cuda.nvtx.range_pop() # end gpu_compute
        finally:
            torch.cuda.nvtx.range_pop() # end step_i

# ==============================
# Main
# ==============================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader = get_dataloaders(batch_size=64)
    model = get_profiled_resnet50(num_classes=10).to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    print("Starting ResNet-50 Training Loop...")
    start_time = my_timer()

    # 20개 배치만 테스트 실행
    train_one_epoch_prefetch(model, train_loader, optimizer, loss_fn, device, max_batches=20)

    total_s = (my_timer() - start_time) / 1e9
    print(f"Finished. Total time: {total_s:.3f} s")

if __name__ == "__main__":
    main()