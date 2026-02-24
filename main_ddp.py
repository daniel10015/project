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
import json



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

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        #for only fisrst batch
        if self.base_time == 0:
            self.base_time = time_ns()

        #GPU id (local_rank)
        device_id = torch.cuda.current_device()

        start.record()
        start_time = my_timer()
        ret = func(*args)
        end.record()

        torch.cuda.synchronize()
        end_time = int(1e6*start.elapsed_time(end)) + start_time
        time_elapsed = end_time - start_time


        name = getattr(func, "_prof_name", func.__class__.__name__)

        #making a dict for information from each layers
        if name not in self.layers:
            self.layers[name] = {
                "time_ns": [],
                "flops": self.flops_by_module.get(name, None),
                "throughput": [],  # FLOPs per second
                "device_id": device_id # GPU ID
            }

        #adding time_elaspeed into layers
        self.layers[name]["time_ns"].append(time_elapsed)
        flops = self.layers[name]["flops"]

        if flops and flops > 1e-10:
            #timeline for redering
            self.timeline.append((device_id,start_time, end_time, flops, name))

        if do_print:
            if flops is not None and time_elapsed > 0:
                #convert ns to s
                t_sec = time_elapsed * 1e-9
                throughput = flops / t_sec  # FLOPs / sec
                #adding throughput into layer dict
                self.layers[name]["throughput"].append(throughput)
                print(
                    f'layer {name}: time={time_elapsed/1e6:.3f} ms, '
                    f'FLOPs={flops}, throughput={throughput/1e12:.3f} TFLOPs'
                )
            else:
                print(f'layer {name}: time={time_elapsed/1e6:.3f} ms (no FLOPs info)')

        return ret

    def save_rank_profile_json(self, rank, prefix="layer_profile"):

        clean_layers = {}
        for layer_name, info in self.layers.items():
            # time_ns
            time_ns_list = []
            for t in info["time_ns"]:
                time_ns_list.append(int(t))

            # flops
            if info["flops"] is not None:
                flops_value = float(info["flops"])
            else:
                flops_value = None

            # throughput
            throughput_list = []
            for item in info["throughput"]:
                throughput_list.append(float(item))

            # device_id
            dev_id = info.get("device_id", None)
            if dev_id is not None:
                dev_id = int(dev_id)

            clean_layers[layer_name] = {
                "time_ns": time_ns_list,
                "flops": flops_value,
                "throughput": throughput_list,
                "device_id": dev_id,
            }
        timeline_list = []

        for event in self.timeline:
            dev_id, start, end, flops, name = event
            timeline_list.append([
                int(dev_id) if dev_id is not None else None,
                int(start),
                int(end),
                float(flops),
                name
            ])


        payload = {
            "rank": rank,
            "layers": clean_layers,
            "timeline": timeline_list,
            "base_time": int(self.base_time),
        }
        
        fname = f"{prefix}_rank_{rank}.json"
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[Rank {rank}] JSON layer profile saved → {fname}")


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
    



#for debugging to make sure activity.deviceId represent each GPU
# def cupti_memcpy_callback(activity):
#     print(
#         "CUPTI memcpy:",
#         "deviceId =", getattr(activity, "deviceId", None),
#         "kind =", getattr(activity, "copy_kind", None),
#         "bytes =", activity.bytes,
#         "time =", activity.start, "→", activity.end,
#     )
#     memcpy_info.memcpy(activity)



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
    train_data = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = CIFAR10(root='./data', train=False, download=True, transform=transform)
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



# Update: removed rank argument (run function only needs local_rank)
def run(num_epochs, local_rank, train_sampler, loader):

    if not torch.cuda.is_available():
        if dist.get_rank() == 0:
            print('Warning: CUDA is not available! Using CPU instead')

    # print("local_rank =", local_rank)
    # print("torch current device =", torch.cuda.current_device())


    device = torch.device('cuda', local_rank)
    model = SmallResidualNetwork(num_classes=10).to(device)

    if dist.get_rank() == 0:
        print(f"(Rank {dist.get_rank()}, Local {local_rank}) Device: {device}")

    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[local_rank])  # use local_rank

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    loss_fn = nn.CrossEntropyLoss()

 
 

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

    # 7. File for layers info on each rank
    profile.save_rank_profile(rank, prefix="layer_profile")

    # 8. Rank 0 Memcpy data gathers
    if rank == 0:
         # ---- memcpy data -------------------------
        all_memcpy = []
        for i in range(world_size): 

            fname = f"memcpy_data_rank_{i}.json"
            try:
                with open(fname, "r") as f:
                    json_list = json.load(f)   # list of dicts
                for ev in json_list:
                    time_ns = int(ev["time_ns"])
                    size_bytes = int(ev["bytes"])
                    kind = ev["kind"]
                    dev_id = ev.get("device_id", None)
                    all_memcpy.append((time_ns, size_bytes, kind))
                
            except FileNotFoundError:
                print(f"Warning: Could not find memcpy data for Rank {i}")


        combined_memcpy_info = MemoryCopy()
        combined_memcpy_info.memcpy_info = all_memcpy
        
        # ---- layer data---------------------
        all_layers = {}       
        all_timelines = [] 
        for i in range(world_size):
            fname = f"layer_profile_rank_{i}.json"
            try:
                with open(fname, "r") as f:
                    data = json.load(f)   # { "rank": i, "layers": {...}, "timeline": [...], ...

                rank_in_file = data.get("rank", i)

                layers = data.get("layers", {})
                for layer_name, info in layers.items():
                    # time_ns
                    time_ns_list = []
                    for t in info.get("time_ns", []):
                        time_ns_list.append(int(t))

                    # flops
                    flops_value = info.get("flops", None)

                    # throughput
                    throughput_list = []
                    for x in info.get("throughput", []):
                        throughput_list.append(float(x))

                    # device_id
                    dev_id = info.get("device_id", None)
                    if dev_id is not None:
                        dev_id = int(dev_id)

                    key = (rank_in_file, dev_id, layer_name)
                    all_layers[key] = {
                        "time_ns": time_ns_list,
                        "flops": flops_value,
                        "throughput": throughput_list,
                        "device_id": dev_id,
                    }
                
                #-----------timeline  ----------------------
                timeline = data.get("timeline", [])
                for ev in timeline:
                    # ev = [device_id, start_ns, end_ns, flops, name]
                    dev_id   = ev[0]
                    start_ns = int(ev[1])
                    end_ns   = int(ev[2])
                    flops    = float(ev[3])
                    name     = ev[4]

                    all_timelines.append((rank_in_file, dev_id, start_ns, end_ns, flops, name))
                
            

            except FileNotFoundError:
                print(f"Warning: Could not find layer profile for Rank {i}")
                
        
        print(f"Training complete. Took {total_ns/1e9:.3f} s")


    # 8. Destroy DDP process group
    dist.destroy_process_group()
   

    #9. Plotting
    #plot_memcpy_timeline_by_device(combined_memcpy_info.memcpy_info)
    #plot_throughput_timeline_by_device(all_layers,all_timelines)
    #plot_combined_by_device(combined_memcpy_info.memcpy_info,all_layers,all_timelines)


if __name__ == '__main__':
    main()
