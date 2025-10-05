from torch import nn
import torch
from profiler.profiler import profiler

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def do_something():
  x_cpu = torch.randn(8, 3, 32, 32)
  x_gpu = x_cpu.to("cuda")
  model = CNN().to("cuda")
  output_gpu = model(x_gpu)
  output_cpu = output_gpu.to("cpu")
  print(x_cpu.device, output_gpu.device, output_cpu.device)
  torch.cuda.empty_cache()

def do_profile():
  profile = profiler(do_something, ('MEMORY',))
  profile()

def do_torchprofile():
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as prof:
      with torch.profiler.record_function("do_something"):
          do_something()

from profiler import benchmark

# warmup
do_something()
regular_time = benchmark.benchmark_ns(do_something)
profile_time = benchmark.benchmark_ns(do_profile)
torch_time = benchmark.benchmark_ns(do_torchprofile)
torch_time = benchmark.benchmark_ns(do_torchprofile)

print(f'regular time took: {regular_time}ns\n'
      f'profile time took: {profile_time}ns\n'
      f'torch time took: {torch_time}ns')

profile = profiler(do_something, ('MEMORY',))
profile()
profile.visualize('MEMORY')

profile_info = profile.spill()
for metric_type, metric_out in profile_info.items():
  info = f'{metric_type} => {metric_out}'
  print(info)