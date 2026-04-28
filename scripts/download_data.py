#!/usr/bin/env python3
"""
Utility to pre-download supported datasets to the project's data directory.

Supported: MNIST, Fashion-MNIST, KMNIST, EMNIST, CIFAR-10, CIFAR-100, STL-10

Examples:
  python scripts/download_data.py --all
  python scripts/download_data.py --datasets mnist fashion-mnist
  python scripts/download_data.py --root /tmp/data --datasets cifar10 cifar100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, Iterable, List

from torchvision import datasets

from csfl_simulator.core.utils import DATA_ROOT


def normalize_name(name: str) -> str:
    lowered = name.strip().lower()
    if lowered in {"fashionmnist", "fmnist", "fashion-mnist", "fashion_mnist"}:
        return "fashion-mnist"
    if lowered in {"kmnist", "kuzushiji", "kuzushiji-mnist", "kuzushiji_mnist"}:
        return "kmnist"
    if lowered in {"emnist", "extended-mnist", "extended_mnist"}:
        return "emnist"
    if lowered in {"cifar10", "cifar-10", "cifar_10"}:
        return "cifar10"
    if lowered in {"cifar100", "cifar-100", "cifar_100"}:
        return "cifar100"
    if lowered in {"mnist"}:
        return "mnist"
    if lowered in {"stl10", "stl-10", "stl_10"}:
        return "stl10"
    return lowered


def get_supported() -> Dict[str, Callable[[Path], None]]:
    def _mnist(root: Path) -> None:
        datasets.MNIST(root, train=True, download=True)
        datasets.MNIST(root, train=False, download=True)

    def _fmnist(root: Path) -> None:
        datasets.FashionMNIST(root, train=True, download=True)
        datasets.FashionMNIST(root, train=False, download=True)

    def _kmnist(root: Path) -> None:
        datasets.KMNIST(root, train=True, download=True)
        datasets.KMNIST(root, train=False, download=True)

    def _emnist(root: Path) -> None:
        # 'digits' split — 10 classes, drop-in for the FD-CNN1/2/3 head.
        # The simulator subsamples it at load time (see EMNISTSubsampled);
        # the underlying torchvision archive is the full 240k/40k blob.
        datasets.EMNIST(root, split="digits", train=True, download=True)
        datasets.EMNIST(root, split="digits", train=False, download=True)

    def _cifar10(root: Path) -> None:
        datasets.CIFAR10(root, train=True, download=True)
        datasets.CIFAR10(root, train=False, download=True)

    def _cifar100(root: Path) -> None:
        datasets.CIFAR100(root, train=True, download=True)
        datasets.CIFAR100(root, train=False, download=True)

    def _stl10(root: Path) -> None:
        datasets.STL10(root, split="train", download=True)
        datasets.STL10(root, split="test", download=True)
        datasets.STL10(root, split="unlabeled", download=True)

    return {
        "mnist": _mnist,
        "fashion-mnist": _fmnist,
        "kmnist": _kmnist,
        "emnist": _emnist,
        "cifar10": _cifar10,
        "cifar100": _cifar100,
        "stl10": _stl10,
    }


def parse_args() -> argparse.Namespace:
    supported = sorted(get_supported().keys())
    parser = argparse.ArgumentParser(description="Pre-download datasets to the data directory")
    parser.add_argument(
        "--root",
        type=Path,
        default=DATA_ROOT,
        help=f"Destination directory for datasets (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        metavar="NAME",
        help=f"Subset of datasets to download. Supported: {', '.join(supported)}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all supported datasets",
    )
    return parser.parse_args()


def resolve_datasets(args: argparse.Namespace) -> List[str]:
    supported = get_supported()
    if args.all or not args.datasets:
        return list(supported.keys())
    requested = [normalize_name(n) for n in args.datasets]
    unknown = [n for n in requested if n not in supported]
    if unknown:
        raise SystemExit(
            f"Unknown dataset(s): {', '.join(unknown)}. Supported: {', '.join(sorted(supported.keys()))}"
        )
    return requested


def download(names: Iterable[str], root: Path) -> None:
    supported = get_supported()
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"\n==> Downloading {name} to {root} ...")
        supported[name](root)
        print(f"Done: {name}")


def main() -> None:
    args = parse_args()
    names = resolve_datasets(args)
    download(names, args.root)
    print("\nAll requested datasets are present.")


if __name__ == "__main__":
    main()


