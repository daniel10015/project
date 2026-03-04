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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from memory_logger import MemoryLogger


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
    Measures GPU time for a module call using CUDA events (ms resolution),
    and also adds an NVTX range with the module name for timeline visualization.
    """
    # 0) Resolve name first (so NVTX can use it)
    name = getattr(func, "_prof_name", func.__class__.__name__)

    # 1) Create CUDA events
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)

    # 2) Base time init (optional, if you use it elsewhere)
    if self.base_time == 0:
        self.base_time = time_ns()

    # 3) Record start
    start_time = my_timer()
    start_evt.record()

    # 4) NVTX range around the actual call
    torch.cuda.nvtx.range_push(name)
    try:
        ret = func(*args)
    finally:
        # ensure pop even if exception occurs
        torch.cuda.nvtx.range_pop()

    # 5) Record end + sync for accurate timing
    end_evt.record()
    torch.cuda.synchronize()

    # 6) Compute elapsed time (ns)
    # start_evt.elapsed_time(end_evt) returns milliseconds (float)
    elapsed_ms = start_evt.elapsed_time(end_evt)
    time_elapsed = int(elapsed_ms * 1e6)  # ms -> ns

    end_time = start_time + time_elapsed

    # 7) Bookkeeping
    if name not in self.layers:
        self.layers[name] = {
            "time_ns": [],
            "flops": self.flops_by_module.get(name, None),
            "throughput": []  # FLOPs per second (optional)
        }

    self.layers[name]["time_ns"].append(time_elapsed)

    fl = self.layers[name]["flops"]
    if fl is not None and fl > 1e-10:
        # timeline entries: (start, end, flops, name)
        self.timeline.append((start_time, end_time, fl, name))

    # 8) Optional debug printing
    do_print = False
    if do_print:
        if fl is not None and time_elapsed > 0:
            t_sec = time_elapsed * 1e-9
            thr = fl / t_sec  # FLOPs/sec
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
# dataloader util
# ==============================
def get_dataloaders(batch_size: int = 64):
    # setup transforms
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


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


def nvtx_push(name):
    torch.cuda.nvtx.range_push(name)

def nvtx_pop():
    torch.cuda.nvtx.range_pop()




# ==============================
# Training
# ==============================
def train_one_epoch(model, train_loader, optimizer, loss_fn, device, memlog, max_batches=None):
    model.train(True)
    correct = 0
    total = 0

    # iterator를 직접 써야 "data_wait"를 NVTX로 정확히 감쌀 수 있음
    it = iter(train_loader)
    n_steps = max_batches if max_batches is not None else len(train_loader)

    # (원래 너 코드 유지) 정확도는 일부 step만 계산
    k = 3
    target_indices = [int(np.floor(j * (n_steps - 1) / (k - 1))) for j in range(k)] if n_steps else None

    for i in range(n_steps):
        nvtx_push(f"step_{i:04d}")
        try:
            memlog.reset_step_peak()
            memlog.mark(i, "step_start")

            # (1) data_wait
            nvtx_push("data_wait")
            try:
                try:
                    data, label = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    data, label = next(it)
            finally:
                nvtx_pop()
            memlog.mark(i, "after_data_wait")

            # (2) h2d
            nvtx_push("h2d")
            try:
                data = data.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)
            finally:
                nvtx_pop()
            memlog.mark(i, "after_h2d")

            # (3) gpu_compute
            nvtx_push("gpu_compute")
            try:
                nvtx_push("zero_grad")
                try:
                    optimizer.zero_grad(set_to_none=True)
                finally:
                    nvtx_pop()
                memlog.mark(i, "before_forward")

                nvtx_push("forward")
                try:
                    pred = model(data)
                finally:
                    nvtx_pop()
                memlog.mark(i, "after_forward")

                nvtx_push("loss")
                try:
                    val = loss_fn(pred, label)
                finally:
                    nvtx_pop()
                memlog.mark(i, "after_loss")

                nvtx_push("backward")
                try:
                    val.backward()
                finally:
                    nvtx_pop()
                memlog.mark(i, "after_backward")

                nvtx_push("opt_step")
                try:
                    optimizer.step()
                finally:
                    nvtx_pop()
                memlog.mark(i, "after_opt_step")

            finally:
                nvtx_pop()  # gpu_compute 끝

            memlog.mark(i, "step_end")

        finally:
            nvtx_pop()      # step 끝 (무조건 닫힘)


        # (원래 너 코드 유지) 일부 step만 accuracy 측정
        if target_indices is not None and i in target_indices:
            preds = torch.argmax(pred, dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return (correct / total) if total > 0 else 0.0

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


    memlog = MemoryLogger(device=device, out_csv="mem_log.csv")

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


    if was_training:
        model.train()

    # ---- Training Loop ----
    accuracy_epoch_train = []
    accuracy_epoch_valid = []

    start_time = my_timer()

    for epoch in range(1, epoch_count + 1):
        train_acc = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, memlog, max_batches=batch_no_count
        )
        accuracy_epoch_train.append(train_acc)

    memlog.dump()
    
    total_ns = my_timer() - start_time
    print(f"Training complete. Took {total_ns/1e9:.3f} s")



if __name__ == "__main__":
    main()
