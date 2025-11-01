from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

from .utils import DATA_ROOT


_MNIST_MEAN, _MNIST_STD = (0.1307,), (0.3081,)
_FMNIST_MEAN, _FMNIST_STD = (0.2860,), (0.3530,)
_CIFAR10_MEAN, _CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN, _CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def get_transforms(name: str, train: bool = True):
    name = name.lower()
    if name in ("mnist",):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD)
        ])
    if name in ("fashion-mnist", "fashionmnist"):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_FMNIST_MEAN, _FMNIST_STD)
        ])
    if name in ("cifar10",):
        aug = []
        if train:
            aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        return transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD)
        ])
    if name in ("cifar100",):
        aug = []
        if train:
            aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        return transforms.Compose(aug + [
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD)
        ])
    raise ValueError(f"Unknown dataset: {name}")


def get_dataset(name: str, train: bool = True, root: Path | None = None, download: bool = True):
    root = root or DATA_ROOT
    name_l = name.lower()
    if name_l == "mnist":
        return datasets.MNIST(root, train=train, download=download, transform=get_transforms(name_l, train))
    if name_l in ("fashion-mnist", "fashionmnist"):
        return datasets.FashionMNIST(root, train=train, download=download, transform=get_transforms(name_l, train))
    if name_l == "cifar-10" or name_l == "cifar10":
        return datasets.CIFAR10(root, train=train, download=download, transform=get_transforms("cifar10", train))
    if name_l == "cifar-100" or name_l == "cifar100":
        return datasets.CIFAR100(root, train=train, download=download, transform=get_transforms("cifar100", train))
    raise ValueError(f"Unknown dataset {name}")


def get_full_data(name: str, root: Path | None = None):
    return get_dataset(name, True, root), get_dataset(name, False, root)


def make_loader(dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 4):
    # Use pinned memory only when CUDA is available to avoid warnings on CPU-only runs
    pin = torch.cuda.is_available()
    # Optimize DataLoader for better throughput
    # persistent_workers and prefetch_factor improve performance significantly
    persistent = num_workers > 0
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=2 if num_workers > 0 else None
    )


def make_loaders_from_indices(dataset, indices, batch_size: int = 64, num_workers: int = 4):
    sub = Subset(dataset, indices)
    return make_loader(sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
