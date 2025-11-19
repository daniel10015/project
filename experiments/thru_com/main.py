from torch import nn
from time import perf_counter_ns, time_ns

# -------------------------------------
# custom implementation of resnet 18
# -------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cupti import cupti

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fvcore.nn import FlopCountAnalysis

# set whatever time you want to use here
my_timer = perf_counter_ns
PINNED_MEMORY = True
PREFETCH_FACTOR = 3
DO_CUDA_SYNC = False

# ------------
# utils
# ------------
def scale_time_units(times_ns):
    """
    Scales time units s.t. it's in the highest possible unit
    that is > 0
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
    self.tracking = [] # per batch  
    self.timeline = []
    self.base_time = 0

  def benchmark_ns(self, func, *args):
    """
    synchronizes torch.cuda
    """
    if DO_CUDA_SYNC:
        torch.cuda.synchronize()
    if self.base_time == 0:
        self.base_time = time_ns()
    start_time = my_timer()
    ret = func(*args)
    if DO_CUDA_SYNC:
        torch.cuda.synchronize()
    end_time = my_timer()
    time_elapsed = end_time - start_time
    
    name = getattr(func, "_prof_name", func.__class__.__name__)

    if name not in self.layers:
            self.layers[name] = {
                "time_ns": [],
                "flops": self.flops_by_module.get(name, None),
                "throughput": []  # FLOPs per second
            }

    self.layers[name]["time_ns"].append(time_elapsed)
    fl = self.layers[name]["flops"]
    if fl and fl > 1e-10:
        self.timeline.append((start_time, end_time, fl, name))
    do_print = False
    if do_print:
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


profile = ExtractModel('SmallResnet')


# ------------------------------------
# Model define 
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

            # you can either time block as a whole:
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
      self.memcpy_info.append((activity.start, activity.bytes, activity.copy_kind))
      self.memcpy_info.append((activity.end, -activity.bytes, activity.copy_kind))

memcpy_info = MemoryCopy()

def func_buffer_requested():
  buffer_size = 8 * 1024 * 1024  # 8MB buffer
  max_num_records = 0
  return buffer_size, max_num_records

def func_buffer_completed(activities: list):
  for activity in activities:
    # formality conditional
    if activity.kind == cupti.ActivityKind.MEMCPY:
        memcpy_info.memcpy(activity)

def setup_cupti():
    # start data collection right before training loop
    cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)

def finalize_cupti():
    cupti.activity_flush_all(1)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)

def plot_memcpy_timeline(memcpy_info: MemoryCopy):
    # Sort events by time
    memcpy_info.memcpy_info.sort(key=lambda x: x[0])

    # Compute cumulative utilization over time
    times, sizes, kinds = zip(*memcpy_info.memcpy_info)
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
    plt.show()

def plot_throughput_timeline(profile_out):
    timeline = profile_out.timeline
    timeline = sorted(timeline, key=lambda x: x[0])

    t0 = timeline[0][0]
    events = []
    for start, end, flops, layer in timeline:
        start -= t0
        end   -= t0
        dur = end - start
        thr = flops / dur if flops and dur > 0 else 0
        events.append((start, dur, thr, layer))

    # collect layers in order of first appearance
    layers = []
    for _, _, _, layer in events:
        if layer not in layers:
            layers.append(layer)

    # map each layer -> y position

    # color palette
    cmap = plt.get_cmap("tab20")
    colors = {layer: cmap(i % 20) for i, layer in enumerate(layers)}

    max_thr = max(thr for _, _, thr, _ in events)   # highest throughput
    labeloffset_y = max_thr * 0.05                        # row above all bars

    # create plot
    plt.figure(figsize=(14, 6))

    # draw horizontal bars
    for idx, (start, dur, thr, layer) in enumerate(events):
        # note that y is actually the center of the bar so we get the center of it
        plt.barh(
            y=thr/2,
            width=dur,
            left=start,
            height=thr,
            color=colors[layer],
            edgecolor='black',
            alpha=0.9
        )

        cx = start + dur / 2  # center X position of the bar
        plt.text(
            cx,
            thr + labeloffset_y,
            str(idx%len(layers)),
            ha='center',
            va='bottom',
            fontsize=8
        )

    # label axes
    plt.xlabel("Time (seconds)")
    plt.ylabel("throughput (FLOPs)")
    plt.title("Model Execution Timeline (Throughput by Block)")
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.autoscale(enable=True)

    legend_items = [
        Patch(facecolor=colors[layer], label=f'({idx}){layer}')
        for idx, layer in enumerate(layers)
    ]

    plt.legend(handles=legend_items, title="Layers", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

def plot_combined(memcpy_info: MemoryCopy, profile_out):
    # ----------------------------------------
    # MEMORY COPY PREP
    # ----------------------------------------
    memcpy_info.memcpy_info.sort(key=lambda x: x[0])
    times, sizes, kinds = zip(*memcpy_info.memcpy_info)

    timeline = sorted(profile_out.timeline, key=lambda x: x[0])
    t0 = timeline[0][0]

    print(f'smallest memcpy time: {np.min(times)}, smallest throughput time: {profile_out.base_time}')
    smallest = min(np.min(times), profile_out.base_time)

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
    dif = 0 if abs(profile_out.base_time - smallest) < 1e-10 else profile_out.base_time - smallest

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
    ax2.set_ylabel("Throughput (FLOPs)")

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
    plt.show()

# ==============================
# dataloader util
# ==============================
def get_dataloaders(batch_size: int = 64):
    # setup transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 64
    # download data with transformations
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # load data
    train_loader = DataLoader(train_data, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=2, 
                              pin_memory=PINNED_MEMORY, 
                              prefetch_factor=PREFETCH_FACTOR)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Number of train batches:", len(train_loader))
    print("Number of test batches:", len(test_loader))
    return train_loader, test_loader


# ==============================
# Training / Validation 
# ==============================
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, max_batches=None):
    model.train(True)
    correct = 0
    total = 0

    # for only k times get the data to keep gpu throughput high
    k = 3
    target_indices = [np.floor(j * (max_batches-1)/(k-1)) for j in range(k)] if max_batches else None

    for i, (data, label) in enumerate(train_loader):
        if max_batches is not None and i >= max_batches:
            break

        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        pred = model(data)
        val = loss_fn(pred, label)
        val.backward()
        optimizer.step()

        if max_batches and i in target_indices: 
            preds = torch.argmax(pred, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return correct / total

def eval_one_epoch(model, test_loader, device):
    model.train(True)
    correct, total = 0, 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            preds = torch.argmax(pred, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)
    return correct / total



# ==============================
# main funciton
# ==============================
def main():
    batch_size = 64
    epoch_count = 1
    batch_no_count = 16

    if not torch.cuda.is_available():
        print('warning, cuda is not available! Using cpu instead')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)

    model = SmallResidualNetwork(num_classes=10).to(device)

    lr = 0.005
    momentum = 0.9

    optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    loss_fn = nn.CrossEntropyLoss()

    # module FLOPs calculation (1 times)
    X_example = next(iter(train_loader))[0].to(device)

    was_training = model.training  
    model.eval()                  

    fa = FlopCountAnalysis(model, X_example)
    profile.flops_by_module = dict(fa.by_module())

    if was_training:
        model.train()

    print("FLOPs by module:", profile.flops_by_module)


    # ----  CUPTI start -----
    setup_cupti()

    # ---- Training Loop ----
    accuracy_epoch_train = []
    accuracy_epoch_valid = []

    start_time = my_timer()

    for epoch in range(1, epoch_count + 1):
        train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, max_batches=batch_no_count
        )
        accuracy_epoch_train.append(train_acc)

        #valid_acc = eval_one_epoch(model, test_loader, device)
        #accuracy_epoch_valid.append(valid_acc)
#
        #print(f"Epoch [{epoch}/{epoch_count}] | "
        #      f"Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

    total_ns = my_timer() - start_time
    print(f"Training complete. Took {total_ns/1e9:.3f} s")

    # ---- plotting ----
    finalize_cupti()
    #plot_memcpy_timeline(memcpy_info)
    #plot_throughput_timeline(profile)
    plot_combined(memcpy_info, profile)




if __name__ == "__main__":
    main()