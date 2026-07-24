"""MNIST loading, normalised into the range the diffusion process assumes.

The one thing that matters here is the normalisation. DDPM's forward process
mixes the image with standard Gaussian noise, so the image has to live on a
comparable scale: symmetric about zero and roughly unit-ish in extent. MNIST
arrives as bytes in the range zero to two hundred and fifty-five, which
``ToTensor`` maps to zero-to-one; the extra affine step centres that on zero and
stretches it to minus-one-to-one. Get this wrong — train on zero-to-one data —
and the model still trains, but samples come out with a systematic brightness
offset that is annoyingly hard to trace back to its cause.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

__all__ = ["get_mnist_dataset", "get_mnist_dataloader", "denormalize", "MNIST_STATS"]

#: The affine used to reach the range from minus one to one. Kept as a named
#: constant so `viz.py` can invert exactly this and not an approximation.
MNIST_STATS = {"mean": 0.5, "std": 0.5}

IMAGE_SIZE = 28
IMAGE_CHANNELS = 1


def _require_torchvision():
    """Import torchvision, or explain how to get it.

    Only the MNIST loaders need it — the model, the diffusion code and the whole
    test suite run without it, so the dependency is deliberately not imported at
    module scope.
    """
    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - environment problem, not logic
        raise ImportError(
            "torchvision is required for the MNIST loaders. Install the pinned "
            "requirements with:  pip install -r requirements.txt"
        ) from exc
    return torchvision


def _transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.ToTensor(),  # uint8 [0, 255] -> float32 [0, 1], shape (1, 28, 28)
            transforms.Normalize((MNIST_STATS["mean"],), (MNIST_STATS["std"],)),
        ]
    )


def get_mnist_dataset(
    root: str = "./data",
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """The MNIST split, with pixels normalised to the range minus one to one.

    Args:
        root: Directory to read the raw files from, and download them into if
            they are not there. Gitignored.
        train: Training split if true, test split otherwise.
        download: Fetch the files if missing.

    Returns:
        A dataset yielding ``(image, label)`` with image shape ``(1, 28, 28)``,
        dtype float32. The labels are unused by the unconditional model but are
        left in place so a class-conditional variant is a small change rather
        than a rewrite of this file.
    """
    _require_torchvision()
    from torchvision.datasets import MNIST

    return MNIST(root=root, train=train, download=download, transform=_transform())


def get_mnist_dataloader(
    batch_size: int = 128,
    root: str = "./data",
    train: bool = True,
    download: bool = True,
    num_workers: int = 2,
    shuffle: Optional[bool] = None,
    pin_memory: bool = False,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """A ``DataLoader`` over :func:`get_mnist_dataset`.

    ``shuffle`` and ``drop_last`` default to the training split's conventions
    (both on) and are flipped off for the test split unless overridden.
    ``drop_last`` matters more than it looks: a ragged final batch changes the
    loss scale slightly, which puts a visible sawtooth in the logged curve.
    """
    if shuffle is None:
        shuffle = train
    if drop_last is None:
        drop_last = train

    return DataLoader(
        get_mnist_dataset(root=root, train=train, download=download),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def denormalize(x: Tensor) -> Tensor:
    """Invert the training normalisation: minus-one-to-one back to zero-to-one.

    Clamped, because model output is not bounded and a single out-of-range pixel
    otherwise rescales an entire displayed grid.
    """
    return (x * MNIST_STATS["std"] + MNIST_STATS["mean"]).clamp(0.0, 1.0)


def random_batch(batch_size: int = 8, seed: Optional[int] = None) -> Tensor:
    """A batch of synthetic images shaped like MNIST, for tests and smoke runs.

    Not noise: smooth low-frequency blobs on a dark background, so that a model
    overfitting to them has something with spatial structure to latch onto. Lets
    the test suite exercise the pipeline without a hundred-megabyte download.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    coarse = torch.rand(
        batch_size, IMAGE_CHANNELS, 7, 7, generator=generator
    )
    smooth = torch.nn.functional.interpolate(
        coarse, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return smooth * 2.0 - 1.0
