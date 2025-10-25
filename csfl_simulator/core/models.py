from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class CNNMnist(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LightCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)  # 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_model(name: str, dataset: str, num_classes: int, device: str = "cpu", pretrained: bool = False):
    name_l = name.lower()
    if name_l in ("cnn-mnist", "cnn_mnist"):
        model = CNNMnist(num_classes)
    elif name_l in ("lightcnn", "light-cifar"):
        model = LightCIFAR(num_classes)
    elif name_l == "resnet18":
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m
    else:
        raise ValueError(f"Unknown model: {name}")
    model = model.to(device)
    return model


def maybe_load_pretrained(model: nn.Module, name: str, dataset: str):
    # Placeholder: implement loading from artifacts/checkpoints/pretrained if present
    return model
