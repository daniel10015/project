from cupti import cupti
import torch
import torch.nn as nn
import torch.optim as optim

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

def func_buffer_requested():
  buffer_size = 8 * 1024 * 1024  # 8MB buffer
  max_num_records = 0 # Don't put a bound on number of activity records
  return buffer_size, max_num_records

def func_buffer_completed(activities: list):
  for activity in activities:
    if activity.kind == cupti.ActivityKind.MEMCPY:
       print(f"Memcpy {MEMCPY_KIND_STR[activity.copy_kind]} of {activity.bytes} bytes on stream {activity.stream_id} duration (ns) = {activity.end - activity.start}")

cupti.activity_register_callbacks(func_buffer_requested, func_buffer_completed)
cupti.activity_enable(cupti.ActivityKind.MEMCPY)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

batch_size = 32
num_batches = 100
num_epochs = 5

# Pre-allocate all batches on GPU
all_inputs = [torch.randn(batch_size, 3, 64, 64, device=device) for _ in range(num_batches)]
all_labels = [torch.randint(0, 10, (batch_size,), device=device) for _ in range(num_batches)]

for epoch in range(num_epochs):
  for batch_idx in range(num_batches):
    x = all_inputs[batch_idx]
    y = all_labels[batch_idx]

    optimizer.zero_grad()
    out = model(x)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()


cupti.activity_flush_all(1)
cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
print("----------Training complete----------\nloss:", loss.item())