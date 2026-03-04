from torch import nn
import torch
from profiler.profiler import profiler

class BigCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Greatly increase computation, similar to VGG.
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 32x32 -> 16x16
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2) # 16x16 -> 8x8
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # 8x8 -> 1x1
        )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def do_something():
    model = BigCNN().to("cuda")

    total_batch_size = 4096 
    x_cpu = torch.randn(total_batch_size, 3, 32, 32)


    # (HtoD Copy) -> (Compute) -> (DtoH Copy)
    x_gpu = x_cpu.to("cuda")
    
    output_gpu = model(x_gpu)
    output_cpu = output_gpu.to("cpu")

    torch.cuda.synchronize()
    print(x_cpu.device, output_gpu.device, output_cpu.device)
    torch.cuda.empty_cache()

def do_profile():
    profile = profiler(do_something, ('KERNEL', 'MEMCPY'))
    profile()


# def do_torchprofile():
#     with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with torch.profiler.record_function("do_something"):
#             do_something()

from profiler import benchmark

# warmup
do_something()
regular_time = benchmark.benchmark_ns(do_something)
profile_time = benchmark.benchmark_ns(do_profile)
# torch_time = benchmark.benchmark_ns(do_torchprofile)


# print(f'regular time took: {regular_time/1e9}s\n'
#       f'profile time took: {profile_time/1e9}s\n'
#       f'torch time took: {torch_time/1e9}s\n')


print(f'regular time took: {regular_time/1e9}s\n'
      f'profile time took: {profile_time/1e9}s\n')

profile = profiler(do_something, ('KERNEL','MEMCPY'))


profile()
profile.visualize('KERNEL')

profile_info = profile.spill()



# print(f"\n=== Activity: KERNEL ({len(profile.profile_out['KERNEL'])} kernels) ===")

# for kernel_info in profile.profile_out["KERNEL"]:
#     print(kernel_info)
#     print() 

# # for metric_type, metric_out in profile_info.items():
# #   info = f'{metric_type} => {metric_out}'
# #   print(info)

# memcpy_logs = profile_info.get('MEMCPY', [])
# print(f"\n=== Activity: MEMCPY ({len(memcpy_logs)} copies) ===")
# if memcpy_logs:
#     for memcpy_info in memcpy_logs:
#             print(memcpy_info + "\n")
# else:
#     print("No MEMCPY activity captured.\n")
