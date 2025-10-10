import torch
import random
import time
from profiler.util import *

num_iters = 128*8
size_range = (32,1024)
device = torch.device('cuda')
random.seed(0)
torch.manual_seed(0)
def func(num_iters, size_range):
  for i in range(1, num_iters + 1):
    n = random.randint(*size_range)

    # Allocate random tensors (some on CPU, some on GPU)
    on_cpu = random.random() < 0.3
    a = torch.randn(n, n, device="cpu" if on_cpu else device)
    b = torch.randn(n, n, device="cpu" if on_cpu else device)

    # Move to GPU if needed
    if on_cpu and device.type == "cuda":
      a = a.to(device, non_blocking=True)
      b = b.to(device, non_blocking=True)

    # Matmul and optional copy back to CPU
    c = torch.matmul(a, b)
    if random.random() < 0.2:
      _ = c.to("cpu", non_blocking=(device.type == "cuda"))

    # Free most tensors every iteration to encourage malloc/free
    del a, b, c

    # Periodically clear cache and print memory info
    if device.type == "cuda" and i % 100 == 0:
      torch.cuda.synchronize()
      torch.cuda.empty_cache()
      print(f"[{i:04d}] GPU allocated: {b_to_mb(torch.cuda.memory_allocated()):.1f} MB")

from profiler.profiler import profiler

profile = profiler(fn=func, metrics=('MEMORY',))
start = time.time()
profile(num_iters, size_range)
total_time = time.time() - start
print(f'profiling function took: {1000*total_time}ms')
profile.visualize()