import torch
from cupti import cupti

print("PyTorch version:", torch.__version__)
print("CUPTI module loaded:", cupti.__name__)

def func_buffer_requested():
  buffer_size = 8 * 1024 * 1024  # 8MB buffer
  max_num_records = 0
  return buffer_size, max_num_records

def func_buffer_completed(activities: list):
  for activity in activities:
    if activity.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
      print(f"kernel name = {activity.name}")
      print(f"kernel duration (ns) = {activity.end - activity.start}")

#Step 1: Register CUPTI callbacks
cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

#Step 2: Enable CUPTI Activity Collection
cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")

c = torch.matmul(a, b)

torch.cuda.synchronize()

#Step 3: Flushing and Disabling CUPTI Activity
cupti.activity_flush_all(1)
cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)

print("Computation done, profiling captured.")
print("Result tensor shape:", c.shape)
