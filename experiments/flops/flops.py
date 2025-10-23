import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from cupti import cupti

"""
Install fvcore with 'pip install fvcore'

fvcore provides flexible floating point operation analysis
"""

device = torch.device("cuda")

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.input_size = 28*28
    self.output_size = 10

    self.layers = nn.Sequential(
      nn.Linear(self.input_size, 512),
      nn.ReLU(),
      nn.Linear(512,256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, 10),
      nn.Sigmoid()
    )

  def forward(self, X):
    return self.layers(X)
  
model = MyModel()
model.to(device)
X = torch.randn(28, 28).flatten().to(device) # Batch size 1, 1 channels, 28x28 image

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

flops = FlopCountAnalysis(model, X) # does a static analysis of FLOPs
model(X)

#Step 3: Flushing and Disabling CUPTI Activity
cupti.activity_flush_all(1)
cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
print('finished collecting metrics\n----------')

# Total FLOPs for the entire model
total_flops = flops.total()
print(f"Total FLOPs: {total_flops}\n")

# FLOPs by operator type
flops_by_operator = flops.by_operator()
print(f"FLOPs by operator: {flops_by_operator}\n")

# FLOPs by module
flops_by_module = flops.by_module()
print(f"FLOPs by module: {flops_by_module}\n")

# FLOPs by module and operator
flops_by_module_and_operator = flops.by_module_and_operator()
print(f"FLOPs by module and operator: {flops_by_module_and_operator}\n")