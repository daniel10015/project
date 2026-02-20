from cupti import cupti

import torch
import torch.nn as nn

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
      nn.Softmax(dim=0)
    )

  def forward(self, X):
    return self.layers(X)

def func_buffer_requested():
  buffer_size = 8 * 1024 * 1024  # 8MB buffer
  max_num_records = 0
  return buffer_size, max_num_records

def func_buffer_completed(activities: list):
  for i, activity in enumerate(activities):
    print(f'{i+1}: {activity.items()}')
    


cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)

cupti.activity_enable(cupti.ActivityKind.METRIC)

model = MyModel().to(device)
X = torch.randn(28, 28).flatten().to(device) # Batch size 1, 1 channels, 28x28 image, but flattened
model(X)

cupti.activity_flush_all(1)
cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
print('finished collecting metrics\n----------')