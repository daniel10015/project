from torch import nn
from time import perf_counter_ns

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

from fvcore.nn import FlopCountAnalysis

# set whatever time you want to use here
my_timer = perf_counter_ns

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

class ExtractModel:
  def __init__(self, root_class_name):
    self.root_cls = root_class_name
    self.layers = {}
    self.tracking = [] # per batch  
  def benchmark_ns(self, func, *args):
    """
    Assumes function is blocking, so when the function returns the execution is finished.
    """
    start_time = my_timer()
    ret = func(*args)
    time_elapsed = my_timer() - start_time
    func_name = f'{self.root_cls}.{func.__class__.__name__}'
    #print(f'function: {func_name} took {time_elapsed/1e6}ms')
    return ret

profile = ExtractModel('SmallResnet')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, do_1x1=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.do1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if do_1x1 else None
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, X):
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

        self.block1 = ResidualBlock(8, 16, stride=2, do_1x1=True)
        self.block2 = ResidualBlock(16, 32, stride=2, do_1x1=True)
        self.block3 = ResidualBlock(32, 64, stride=1, do_1x1=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # safer than fixed 8x8
        self.fc = nn.Linear(64, num_classes)

    def forward(self, X):
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu1(X)
        X = self.pool1(X)

        X = self.block1(X)
        X = self.block2(X)
        X = self.block3(X)

        X = profile.benchmark_ns(self.avgpool, X)
        X = X.view(X.size(0), -1)
        X = self.fc(X)
        return X

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


# ----- setup model and data -----
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
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


if not torch.cuda.is_available():
    print('warning, cuda is not available! Using cpu instead')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallResidualNetwork(num_classes=10).to(device)
lr = 0.005
momentum = 0.9

optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
loss_fn = nn.CrossEntropyLoss()
epoch_count = 1
batch_no_count = 2

accuracy_epoch_train = []
accuracy_epoch_valid = []

# Pull 1 batch
X = next(iter(train_loader))[0].to(device)
flops = FlopCountAnalysis(model, X)
print(f'flops by module: {flops.by_module()}')

# start data collection right before training loop
cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
cupti.activity_enable(cupti.ActivityKind.MEMCPY)

# -------------------------------------
# Training Loop
# -------------------------------------
start_time = my_timer()
for epoch in range(1, epoch_count + 1):
    model.train(True)
    correct = 0
    total = 0

    for i, (data, label) in enumerate(train_loader):
        if i >= batch_no_count:
           break
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        pred = model(data)
        val = loss_fn(pred, label)
        val.backward()
        optimizer.step()

        preds = torch.argmax(pred, dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

    train_acc = correct / total
    accuracy_epoch_train.append(train_acc)

    # Validation loop
    model.train(False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device), label.to(device)
            pred = model(data)
            preds = torch.argmax(pred, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    valid_acc = correct / total
    accuracy_epoch_valid.append(valid_acc)

    print(f"Epoch [{epoch}/{epoch_count}] | Train Acc: {train_acc:.4f} | Valid Acc: {valid_acc:.4f}")

print(f"Training complete. Took {my_timer() - start_time}")
cupti.activity_flush_all(1)
cupti.activity_disable(cupti.ActivityKind.MEMCPY)
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