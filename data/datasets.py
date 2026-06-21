"""
Dataset loading for MNIST, FashionMNIST, and CIFAR-10.
"""

import torchvision
import torchvision.transforms as transforms


def load_dataset(name, root="./data"):
    """Load train and test datasets by name."""
    if name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)

    elif name == "FashionMNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)

    elif name == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train, test
