"""
CNN architectures used for federated training.
- CNNGrayscale: MNIST / FashionMNIST (1-channel, 28x28)
- CNNCIFAR:     CIFAR-10 (3-channel, 32x32)
"""

import torch.nn as nn


class CNNGrayscale(nn.Module):
    """CNN for MNIST and FashionMNIST (1-channel, 28x28)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNNCIFAR(nn.Module):
    """CNN for CIFAR-10 (3-channel, 32x32)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_model(dataset_name, device):
    """Factory: return the correct model for a given dataset, moved to device."""
    if dataset_name == "CIFAR10":
        return CNNCIFAR().to(device)
    else:
        return CNNGrayscale().to(device)
