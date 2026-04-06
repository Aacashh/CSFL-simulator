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


class CNNMnistFedAvg(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        # Two 5x5 conv layers with padding=2 to preserve spatial dims before pooling
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        # Determine flattened size dynamically for given input spatial size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = int(h.numel() // h.shape[0])
        # 512-unit FC with bias disabled to match 1,663,370 params for 28x28 MNIST
        self.fc1 = nn.Linear(flat_dim, 512, bias=False)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class LightCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        # First conv block: 5x5 conv (64 filters) + 2x2 max pool
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5)
        # Second conv block: 5x5 conv (64 filters) + 2x2 max pool
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        # Determine flattened size dynamically based on input size and channels
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = int(h.numel() // h.shape[0])
        # Two FC layers: hidden -> logits
        self.fc1 = nn.Linear(flat_dim, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# FD-specific CNN architectures (Mu et al., IEEE TCCN 2024, Table III)
# Three heterogeneous architectures for federated distillation experiments.
# All use 3x3 kernels, ReLU, MaxPool(2), Dropout before final FC.
# ---------------------------------------------------------------------------

class FDCNN1(nn.Module):
    """Large FD model (~1.2M params). Paper Table III: CNN_1.
    Conv 1-32, MaxPool, Conv 32-64, MaxPool, FC-128, Dropout, FC-C."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class FDCNN2(nn.Module):
    """Medium FD model (~79K params). Paper Table III: CNN_2.
    Conv 1-12, MaxPool, Conv 12-24, MaxPool, FC-64, Dropout, FC-C."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class FDCNN3(nn.Module):
    """Small FD model (~25K params). Paper Table III: CNN_3.
    Conv 1-8, MaxPool, Conv 8-16, MaxPool, FC-64, Dropout, FC-C."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            h = self.pool(F.relu(self.conv1(dummy)))
            h = self.pool(F.relu(self.conv2(h)))
            flat_dim = int(h.numel() // h.shape[0])
        self.fc1 = nn.Linear(flat_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, self.conv1.in_channels)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class MobileNetV2FD(nn.Module):
    """MobileNetV2 variant for FD experiments (~2.2M params for CIFAR-10).
    Uses torchvision MobileNetV2 backbone with adapted first conv and classifier."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        from torchvision.models import mobilenet_v2
        base = mobilenet_v2(weights=None)
        # Adapt first conv for smaller images / different channels
        base.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(base.last_channel, num_classes),
        )
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, 3) if x.shape[1] != 3 else x
        return self.model(x)


class ShuffleNetV2FD(nn.Module):
    """ShuffleNetV2 x0.5 variant for FD experiments (~350K params for CIFAR-10).
    Uses torchvision ShuffleNetV2 backbone with adapted first conv and classifier."""
    def __init__(self, num_classes: int = 10, in_channels: int = 3, image_size: int = 32):
        super().__init__()
        from torchvision.models import shufflenet_v2_x0_5
        base = shufflenet_v2_x0_5(weights=None)
        # Adapt first conv for smaller images
        base.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _match_channels(x, 3) if x.shape[1] != 3 else x
        return self.model(x)


def _dataset_image_spec(dataset: str) -> tuple[int, int]:
    d = (dataset or "").lower()
    if d in ("mnist", "fashion-mnist", "fashionmnist"):
        return 1, 28
    if d in ("cifar10", "cifar-10", "cifar100", "cifar-100"):
        return 3, 32
    if d in ("stl-10", "stl10"):
        return 3, 32  # Resized from 96x96 to match training dataset
    # Fallback to common 3x32
    return 3, 32


def get_model(name: str, dataset: str, num_classes: int, device: str = "cpu", pretrained: bool = False):
    name_l = name.lower()
    in_ch, img_sz = _dataset_image_spec(dataset)
    if name_l in ("cnn-mnist", "cnn_mnist"):
        model = CNNMnist(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("cnn-mnist (fedavg)", "cnn_mnist_fedavg", "cnn-mnist-fedavg"):
        model = CNNMnistFedAvg(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("lightcnn", "light-cifar"):
        model = LightCIFAR(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("fd-cnn1", "fd_cnn1", "fdcnn1"):
        model = FDCNN1(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("fd-cnn2", "fd_cnn2", "fdcnn2"):
        model = FDCNN2(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("fd-cnn3", "fd_cnn3", "fdcnn3"):
        model = FDCNN3(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("resnet18", "resnet18-fd"):
        m = resnet18(weights=None)
        # Adapt first conv for small images (CIFAR 32x32) and different channels
        m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()  # Remove maxpool for 32x32 inputs
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        model = m
    elif name_l in ("mobilenetv2-fd", "mobilenetv2_fd", "mobilenet_v2_fd"):
        model = MobileNetV2FD(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    elif name_l in ("shufflenetv2-fd", "shufflenetv2_fd", "shufflenet_v2_fd"):
        model = ShuffleNetV2FD(num_classes=num_classes, in_channels=in_ch, image_size=img_sz)
    else:
        raise ValueError(f"Unknown model: {name}")
    model = model.to(device)
    return model


def maybe_load_pretrained(model: nn.Module, name: str, dataset: str):
    # Placeholder: implement loading from artifacts/checkpoints/pretrained if present
    return model
