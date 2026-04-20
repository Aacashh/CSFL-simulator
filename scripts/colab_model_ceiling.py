"""
Standalone Colab script to test the ceiling accuracy of FD models on CIFAR-10.

Tests both:
  1. Our current FD-CNN1/2/3 (tiny models from the MNIST architecture in the paper)
  2. The paper's actual CIFAR-10 models: ResNet18, MobileNetV2, ShuffleNetV2

This helps verify whether the models have sufficient capacity for CIFAR-10
before running expensive federated distillation experiments.

Usage: Just run all cells in Google Colab (GPU runtime recommended)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# Model Definitions
# ============================================================

# --- Our current FD-CNN models (paper's MNIST architectures) ---

class FDCNN1(nn.Module):
    """~545K params. Paper Table III CNN_1 (designed for MNIST)."""
    def __init__(self, num_classes=10, in_channels=3, image_size=32):
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class FDCNN2(nn.Module):
    """~102K params. Paper Table III CNN_2 (designed for MNIST)."""
    def __init__(self, num_classes=10, in_channels=3, image_size=32):
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class FDCNN3(nn.Module):
    """~68K params. Paper Table III CNN_3 (designed for MNIST)."""
    def __init__(self, num_classes=10, in_channels=3, image_size=32):
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# --- Paper's actual CIFAR-10 models (Table IV: ResNet, MobileNet, ShuffleNet) ---

def make_resnet18(num_classes=10):
    """ResNet18 adapted for CIFAR-10 (32x32 images)."""
    model = models.resnet18(weights=None, num_classes=num_classes)
    # Replace first conv: 7x7 stride 2 -> 3x3 stride 1 (standard for CIFAR)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool (images are already small)
    return model


def make_mobilenetv2(num_classes=10):
    """MobileNetV2 for CIFAR-10."""
    model = models.mobilenet_v2(weights=None, num_classes=num_classes)
    # Adapt first conv for 32x32
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model


def make_shufflenetv2(num_classes=10):
    """ShuffleNetV2 x0.5 for CIFAR-10."""
    model = models.shufflenet_v2_x0_5(weights=None, num_classes=num_classes)
    # Adapt first conv for 32x32
    model.conv1[0] = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ============================================================
# Data
# ============================================================

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

print("Downloading CIFAR-10...")
train_ds = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
test_ds = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)


# ============================================================
# Training + Evaluation
# ============================================================

def train_and_eval(model, name, epochs=50):
    """Train a model on full CIFAR-10 and report best test accuracy."""
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"  {name}  ({n_params:,} params)")
    print(f"{'='*60}")

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    start = time.time()

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        # Eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        acc = correct / total

        if acc > best_acc:
            best_acc = acc

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:3d}/{epochs}: acc={acc:.4f}  best={best_acc:.4f}  "
                  f"loss={train_loss/len(train_loader):.4f}  [{elapsed:.0f}s]")

    elapsed = time.time() - start
    print(f"\n  >>> BEST ACCURACY: {best_acc:.4f}  ({elapsed:.0f}s total)")
    return best_acc


# ============================================================
# Run all models
# ============================================================

results = {}

print("\n" + "#" * 60)
print("#  PART 1: Our current FD-CNN models (MNIST architectures)")
print("#  These are what exp02 is currently using on CIFAR-10")
print("#" * 60)

results["FD-CNN1 (ours, 545K)"] = train_and_eval(
    FDCNN1(num_classes=10, in_channels=3, image_size=32), "FD-CNN1 (ours)", epochs=50)
results["FD-CNN2 (ours, 102K)"] = train_and_eval(
    FDCNN2(num_classes=10, in_channels=3, image_size=32), "FD-CNN2 (ours)", epochs=50)
results["FD-CNN3 (ours, 68K)"] = train_and_eval(
    FDCNN3(num_classes=10, in_channels=3, image_size=32), "FD-CNN3 (ours)", epochs=50)

print("\n" + "#" * 60)
print("#  PART 2: Paper's actual CIFAR-10 models (Table IV)")
print("#  ResNet18, MobileNetV2, ShuffleNetV2")
print("#" * 60)

results["ResNet18 (paper)"] = train_and_eval(
    make_resnet18(num_classes=10), "ResNet18", epochs=50)
results["MobileNetV2 (paper)"] = train_and_eval(
    make_mobilenetv2(num_classes=10), "MobileNetV2", epochs=50)
results["ShuffleNetV2 (paper)"] = train_and_eval(
    make_shufflenetv2(num_classes=10), "ShuffleNetV2 x0.5", epochs=50)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("  SUMMARY: Centralized CIFAR-10 Ceiling Accuracy (50 epochs)")
print("=" * 60)
print(f"  {'Model':<30s} {'Params':>10s} {'Best Acc':>10s}")
print(f"  {'-'*50}")

param_counts = {
    "FD-CNN1 (ours, 545K)": "545K",
    "FD-CNN2 (ours, 102K)": "102K",
    "FD-CNN3 (ours, 68K)": "68K",
    "ResNet18 (paper)": "~11.2M",
    "MobileNetV2 (paper)": "~2.2M",
    "ShuffleNetV2 (paper)": "~0.35M",
}

for name, acc in results.items():
    params = param_counts.get(name, "?")
    marker = " <-- paper uses these for CIFAR-10" if "paper" in name else ""
    print(f"  {name:<30s} {params:>10s} {acc:>9.2%}{marker}")

print(f"\n  Paper's FD results (error-prone, alpha=0.5, Table IV):")
print(f"    ResNet:     51.77%")
print(f"    MobileNet:  53.58%")
print(f"    ShuffleNet: 46.02%")
print(f"\n  Note: FD accuracy is always lower than centralized ceiling")
print(f"  due to non-IID data, channel noise, and partial knowledge sharing.")
print(f"  Expect FD to reach ~50-70% of centralized ceiling.")
