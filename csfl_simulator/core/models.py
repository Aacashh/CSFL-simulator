from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


def _match_channels(x: torch.Tensor, expected: int) -> torch.Tensor:
    c = x.shape[1]
    if c == expected:
        return x
    if expected == 1 and c >= 3:
        # Simple luminance; fall back to mean if fewer than 3 channels
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    if expected == 3 and c == 1:
        return x.repeat(1, 3, 1, 1)
    # General fallback: trim or repeat to match expected channels
    if c > expected:
        return x[:, :expected]
    rep = (expected + c - 1) // c
    x_rep = x.repeat(1, rep, 1, 1)
    return x_rep[:, :expected]


class CNNMnist(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # Determine flattened size dynamically for given input spatial size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = F.max_pool2d(self.conv1(dummy), 2)
            h = F.max_pool2d(self.conv2(h), 2)
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class LightCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Determine flattened size dynamically based on input size and channels
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = F.relu(self.conv1(dummy))
            h = self.pool(F.relu(self.conv2(h)))
            h = self.pool(F.relu(self.conv3(h)))
            h = self.pool(h)
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _dataset_image_spec(dataset: str) -> tuple[int, int]:
    d = (dataset or "").lower()
    if d in ("mnist", "fashion-mnist", "fashionmnist"):
        return 1, 28
    if d in ("cifar10", "cifar-10", "cifar100", "cifar-100"):
        return 3, 32
    # Fallback to common 3x32
    return 3, 32


def get_model(name: str, dataset: str, num_classes: int, device: str = "cpu", pretrained: bool = False):
    name_l = name.lower()
    in_ch, img_sz = _dataset_image_spec(dataset)
    if name_l in ("cnn-mnist", "cnn_mnist"):
        model = CNNMnist(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("lightcnn", "light-cifar"):
        model = LightCIFAR(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l == "resnet18":
        m = resnet18(weights=None)
        # Adapt first conv to dataset channels when needed
        if in_ch != 3:
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m
    else:
        raise ValueError(f"Unknown model: {name}")
    model = model.to(device)
    return model


def maybe_load_pretrained(model: nn.Module, name: str, dataset: str):
    # Placeholder: implement loading from artifacts/checkpoints/pretrained if present
    return model
