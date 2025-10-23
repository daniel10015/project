import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

"""
Install fvcore with 'pip install fvcore'

fvcore provides flexible floating point operation analysis
"""

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
inputs = torch.randn(28, 28).flatten() # Batch size 1, 1 channels, 28x28 image

flops = FlopCountAnalysis(model, inputs) # does a static analysis of FLOPs

# Total FLOPs for the entire model
total_flops = flops.total()
print(f"Total FLOPs: {total_flops}")

# FLOPs by operator type
flops_by_operator = flops.by_operator()
print(f"FLOPs by operator: {flops_by_operator}")

# FLOPs by module
flops_by_module = flops.by_module()
print(f"FLOPs by module: {flops_by_module}")

# FLOPs by module and operator
flops_by_module_and_operator = flops.by_module_and_operator()
print(f"FLOPs by module and operator: {flops_by_module_and_operator}")