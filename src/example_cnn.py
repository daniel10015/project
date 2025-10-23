from torch import nn
import torch
from profiler.profiler import profiler
from profiler.util import *

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
          nn.Conv2d(3, 64, 3, padding=1),  # 256x256 -> 256x256
          nn.ReLU(inplace=True),
          nn.Conv2d(64, 64, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),  # 128x128

          nn.Conv2d(64, 128, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(128, 128, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),  # 64x64

          nn.Conv2d(128, 256, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),  # 32x32

          nn.Conv2d(256, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(2),  # 16x16

          nn.Conv2d(512, 512, 3, padding=1),
          nn.ReLU(inplace=True),
          nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def do_something():
  device = 'cuda'
  model = CNN().to(device)
  x = torch.randn(16, 3, 256, 256, device=device)  # Larger batch & resolution
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
  criterion = nn.CrossEntropyLoss()
  y = torch.randint(0, 10, (16,), device=device)

  # Run multiple iterations to sustain GPU activity
  for _ in range(5):  # Adjust if faster/slower
      optimizer.zero_grad()
      out = model(x)
      loss = criterion(out, y)
      loss.backward()
      optimizer.step()

def do_profile():
  profile = profiler(do_something, ('MEMCPY',))
  profile()
  profile.visualize()

def do_torchprofile():
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA],profile_memory=True) as prof:
      with torch.profiler.record_function("do_something"):
          do_something()

do_profile()
exit(0)

from profiler import benchmark

print('This test case will use about 3.8GB of memory')

# warmup
print('warmup')
do_something()
print('finished warmup')
regular_time = benchmark.benchmark_ns(do_something)[0]
print('finished with regular time')
profile_time = benchmark.benchmark_ns(do_profile)[0]
print('finished with my_profile time')
torch_time = benchmark.benchmark_ns(do_torchprofile)[0]
print('finished with torch time')

print(f'regular time took: {ns_to_s(regular_time)}s\n'
      f'profile time took: {ns_to_s(profile_time)}s\n'
      f'torch time took: {ns_to_s(torch_time)}s')