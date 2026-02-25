from torchvision.datasets import CIFAR10
import torchvision.transforms.v2 as transforms
from torch.utils.data import ConcatDataset

transform = transforms.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), transforms.Resize(512, 512)])

def get_dataset(copies=10):
    ds = CIFAR10(root='./datasets', train=True, \
                        download=True, transform=transform)

    _ds = CIFAR10(root='./datasets', train=True, \
                            download=True, transform=transform)
    for _ in range(copies-1):
        ds = ConcatDataset([ds, _ds])

    print("Dataset length: ", len(ds))
    return ds


if __name__ == "__main__":
    get_dataset()