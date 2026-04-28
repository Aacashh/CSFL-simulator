from pathlib import Path
from typing import Tuple, Optional

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

from .utils import DATA_ROOT


_MNIST_MEAN, _MNIST_STD = (0.1307,), (0.3081,)
_FMNIST_MEAN, _FMNIST_STD = (0.2860,), (0.3530,)
# KMNIST (Kuzushiji-MNIST): 28x28 grayscale, 10-class cursive Japanese characters.
# Drop-in replacement for MNIST/FMNIST tensor-wise. Stats from rois-codh/kmnist.
_KMNIST_MEAN, _KMNIST_STD = (0.1918,), (0.3483,)
# EMNIST/digits split: 28x28 grayscale, 10-class handwritten Latin digits drawn
# from NIST SD 19. Drop-in replacement for MNIST tensor-wise; full split is
# 240k/40k, which we subsample to 60k/10k for fair scale comparison with the
# paper's FMNIST baseline (see EMNISTSubsampled below). Stats are the standard
# MNIST values — empirically within 0.01 of EMNIST/digits' own stats and what
# the bulk of the EMNIST literature uses.
_EMNIST_MEAN, _EMNIST_STD = (0.1307,), (0.3081,)
_CIFAR10_MEAN, _CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
_CIFAR100_MEAN, _CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
_STL10_MEAN, _STL10_STD = (0.4467, 0.4398, 0.4066), (0.2604, 0.2563, 0.2713)


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
    if name in ("kmnist",):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_KMNIST_MEAN, _KMNIST_STD)
        ])
    if name in ("emnist",):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_EMNIST_MEAN, _EMNIST_STD)
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
    if name in ("stl-10", "stl10"):
        # STL-10 is 96x96; resize to target_size (default 32 for CIFAR compat)
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(_STL10_MEAN, _STL10_STD)
        ])
    raise ValueError(f"Unknown dataset: {name}")


class EMNISTSubsampled(datasets.EMNIST):
    """EMNIST/digits restricted to a deterministic 60k train / 10k test stratified subsample.

    WHY:
        torchvision.datasets.EMNIST(split='digits') ships with 240k/40k samples — 4x the
        size of the paper's FMNIST baseline. Running EXP 2 at native scale would conflate
        the cross-dataset SCOPE-FD claim with a "more data per round" effect, since with
        N=50 clients each Dirichlet partition would hold ~4x more samples than the FMNIST
        equivalent. This class clips to MNIST/FMNIST scale (60k/10k) at __init__ time so
        the simulator's downstream code (partition.py, simulator.setup, FD distillation
        sampler) sees an EMNIST that is byte-for-byte size-equivalent to the FMNIST
        baseline used in EXP 1.

    HOW:
        Stratified per-class random subsample with a fixed seed (42). Operates by
        rewriting self.data and self.targets in-place after the parent constructor runs.
        Preserves .classes and the standard .data / .targets attributes that
        partition.dirichlet_partition and FLSimulator.setup both rely on — a plain
        torch.utils.data.Subset would break that contract because Subset has no .targets.

    SEED:
        Hardcoded to 42 (paper's single-seed protocol). Subsample selection is
        independent of cfg.seed so that re-runs of EXP 2 see the same EMNIST snapshot
        even when the simulator's seed changes.
    """
    _SUBSAMPLE_SEED = 42
    _TARGET_TRAIN = 60000
    _TARGET_TEST = 10000

    def __init__(self, root, train: bool = True, download: bool = True, transform=None):
        super().__init__(root, split="digits", train=train, download=download, transform=transform)
        target = self._TARGET_TRAIN if train else self._TARGET_TEST
        if len(self) <= target:
            return  # Nothing to do (e.g. someone changed the targets above current size).

        labels = self.targets.numpy() if hasattr(self.targets, "numpy") else np.asarray(self.targets)
        rng = np.random.RandomState(self._SUBSAMPLE_SEED + (0 if train else 1))
        n_classes = int(labels.max()) + 1
        per_class = target // n_classes
        keep_idx = []
        for c in range(n_classes):
            class_idx = np.where(labels == c)[0]
            if len(class_idx) <= per_class:
                keep_idx.append(class_idx)
            else:
                keep_idx.append(rng.choice(class_idx, size=per_class, replace=False))
        keep_idx = np.concatenate(keep_idx)
        # Top up to exact target if integer-division rounded down.
        if len(keep_idx) < target:
            remaining = np.setdiff1d(np.arange(len(self)), keep_idx, assume_unique=False)
            extra = rng.choice(remaining, size=(target - len(keep_idx)), replace=False)
            keep_idx = np.concatenate([keep_idx, extra])
        keep_idx.sort()  # deterministic ordering

        self.data = self.data[keep_idx]
        # self.targets is a torch.Tensor in modern torchvision.
        if hasattr(self.targets, "__getitem__"):
            self.targets = self.targets[keep_idx] if isinstance(self.targets, torch.Tensor) \
                else type(self.targets)([self.targets[i] for i in keep_idx.tolist()])


def get_dataset(name: str, train: bool = True, root: Path | None = None, download: bool = True):
    root = root or DATA_ROOT
    name_l = name.lower()
    if name_l == "mnist":
        return datasets.MNIST(root, train=train, download=download, transform=get_transforms(name_l, train))
    if name_l in ("fashion-mnist", "fashionmnist"):
        return datasets.FashionMNIST(root, train=train, download=download, transform=get_transforms(name_l, train))
    if name_l == "emnist":
        # Always EMNIST/digits, subsampled to MNIST scale. See EMNISTSubsampled docstring.
        try:
            return EMNISTSubsampled(root, train=train, download=download, transform=get_transforms(name_l, train))
        except RuntimeError as e:
            if "Error downloading" in str(e) or "downloading" in str(e).lower():
                raise RuntimeError(
                    f"EMNIST download failed — biometrics.nist.gov may be unreachable from this host.\n"
                    f"Recovery: run `bash scripts/fetch_emnist.sh` (handles retries + manual SCP fallback).\n"
                    f"Or place the EMNIST gzip archive manually in {root}/EMNIST/raw/.\n"
                    f"Original error: {e}"
                ) from e
            raise
    if name_l == "kmnist":
        try:
            return datasets.KMNIST(root, train=train, download=download, transform=get_transforms(name_l, train))
        except RuntimeError as e:
            # torchvision wraps urllib network errors in RuntimeError("Error downloading ...")
            # The official KMNIST mirror (codh.rois.ac.jp) is frequently unreachable from
            # academic clusters. Re-raise with an actionable recovery hint instead of the
            # bare network trace.
            if "Error downloading" in str(e) or "downloading" in str(e).lower():
                raise RuntimeError(
                    f"KMNIST download failed — codh.rois.ac.jp may be unreachable from this host.\n"
                    f"Recovery: run `bash scripts/fetch_kmnist.sh` (handles retries + manual SCP fallback).\n"
                    f"Or place the 4 .gz IDX files manually in {root}/KMNIST/raw/.\n"
                    f"Original error: {e}"
                ) from e
            raise
    if name_l == "cifar-10" or name_l == "cifar10":
        return datasets.CIFAR10(root, train=train, download=download, transform=get_transforms("cifar10", train))
    if name_l == "cifar-100" or name_l == "cifar100":
        return datasets.CIFAR100(root, train=train, download=download, transform=get_transforms("cifar100", train))
    if name_l in ("stl-10", "stl10"):
        split = "train" if train else "test"
        return datasets.STL10(root, split=split, download=download, transform=get_transforms("stl10", train))
    raise ValueError(f"Unknown dataset {name}")


def get_full_data(name: str, root: Path | None = None):
    return get_dataset(name, True, root), get_dataset(name, False, root)


def make_loader(dataset, batch_size: int = 64, shuffle: bool = True, num_workers: int = 0):
    import sys
    # Force num_workers=0 on Windows to avoid multiprocessing spawn crashes
    if sys.platform == "win32":
        num_workers = 0
    # Use pinned memory only when CUDA is available to avoid warnings on CPU-only runs
    pin = torch.cuda.is_available()
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


def make_loaders_from_indices(dataset, indices, batch_size: int = 64, num_workers: int = 0):
    sub = Subset(dataset, indices)
    return make_loader(sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def _public_transform_for_training_dataset(training_dataset: str) -> transforms.Compose:
    """Build a transform for the public dataset that matches the training dataset's input spec."""
    d = training_dataset.lower()
    if d in ("mnist",):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(_MNIST_MEAN, _MNIST_STD),
        ])
    if d in ("fashion-mnist", "fashionmnist"):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(_FMNIST_MEAN, _FMNIST_STD),
        ])
    if d in ("kmnist",):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(_KMNIST_MEAN, _KMNIST_STD),
        ])
    if d in ("emnist",):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(28),
            transforms.ToTensor(),
            transforms.Normalize(_EMNIST_MEAN, _EMNIST_STD),
        ])
    if d in ("cifar10", "cifar-10"):
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
    if d in ("cifar100", "cifar-100"):
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR100_MEAN, _CIFAR100_STD),
        ])
    # Fallback: resize to 32, 3 channels, CIFAR-10 normalization
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
    ])


def get_public_dataset(
    name: str,
    size: int,
    training_dataset: str,
    seed: int = 42,
    root: Path | None = None,
) -> Subset:
    """Load a public dataset for federated distillation.

    Args:
        name: "same" (sample from test split of training_dataset),
              "STL-10" (unlabeled STL-10, resized to match training dataset),
              "FMNIST" / "Fashion-MNIST" (Fashion-MNIST test split).
        size: Number of public samples to use.
        training_dataset: The name of the private training dataset (for input spec matching).
        seed: Random seed for reproducible sampling.
        root: Data root directory.

    Returns:
        A Subset of `size` samples with transforms matching the training dataset.
    """
    root = root or DATA_ROOT
    name_l = name.lower()
    rng = np.random.RandomState(seed)

    if name_l == "same":
        ds = get_dataset(training_dataset, train=False, root=root)
        indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
        return Subset(ds, indices)

    if name_l in ("stl-10", "stl10"):
        tfm = _public_transform_for_training_dataset(training_dataset)
        ds = datasets.STL10(root, split="unlabeled", download=True, transform=tfm)
        indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
        return Subset(ds, indices)

    if name_l in ("fashion-mnist", "fashionmnist", "fmnist"):
        tfm = _public_transform_for_training_dataset(training_dataset)
        ds = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
        indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
        return Subset(ds, indices)

    if name_l == "kmnist":
        tfm = _public_transform_for_training_dataset(training_dataset)
        ds = datasets.KMNIST(root, train=False, download=True, transform=tfm)
        indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
        return Subset(ds, indices)

    if name_l == "emnist":
        tfm = _public_transform_for_training_dataset(training_dataset)
        # When EMNIST itself is the public dataset, use the full unsampled
        # 'digits' split — public-set sampling already trims via `size`.
        ds = datasets.EMNIST(root, split="digits", train=False, download=True, transform=tfm)
        indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
        return Subset(ds, indices)

    # Generic fallback: try loading as test split
    ds = get_dataset(name, train=False, root=root)
    indices = rng.choice(len(ds), size=min(size, len(ds)), replace=False).tolist()
    return Subset(ds, indices)


def get_labels(dataset) -> list[int]:
    """Utility to fetch integer class labels from a torchvision-style dataset."""
    try:
        # Many datasets expose 'targets' (list or tensor)
        t = getattr(dataset, "targets", None)
        if t is None:
            t = getattr(dataset, "labels", None)
        if t is not None:
            # Convert to Python ints
            return [int(x) for x in (t.tolist() if hasattr(t, "tolist") else list(t))]
    except Exception:
        pass
    # Fallback: index the dataset
    try:
        return [int(dataset[i][1]) for i in range(len(dataset))]
    except Exception:
        return []
