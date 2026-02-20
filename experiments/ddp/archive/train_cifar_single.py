import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torchvision.models as models


trainset = CIFAR10(root='./datasets', train=True, download=True, transform=ToTensor())
testset = CIFAR10(root='./datasets', train=False, download=True, transform=ToTensor())

train_loader = DataLoader(dataset=trainset, batch_size=256, shuffle=True)
test_loader = DataLoader(dataset=testset, batch_size=256)

num_epochs = 10
device = torch.device('cuda', 0)
model = models.resnet18(num_classes=10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(num_epochs):
    for X, y in train_loader:
        y_hat = model(X.to(device))
        y = y.to(device)
        optimizer.zero_grad()
        loss_fn(y_hat, y).backward()
        optimizer.step()

model.eval()
total = 0
total_correct = 0
for X, y in test_loader:
    y = y.to(device)
    y_hat = torch.argmax(model(X.to(device)), dim=1)    
    total_correct += (y_hat == y).sum()
    total += len(y)

print("Testset Accuracy: ", total_correct / total)

