"""Shared fixtures. Everything runs on CPU and with a fixed seed.

The models built here are absurdly small on purpose. These tests check that the
diffusion maths is *correct*, not that it produces good images, and a
two-hundred-parameter-per-layer network exercises exactly the same code paths as
a real one while letting the sampling test run a full thousand-step chain in a
couple of seconds.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNet  # noqa: E402


@pytest.fixture(autouse=True)
def deterministic():
    """Fix the global seed before every test."""
    torch.manual_seed(1234)
    yield


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_model() -> UNet:
    """A UNet small enough to run a thousand forward passes in seconds."""
    model = UNet(in_channels=1, base_channels=8, channel_mults=(1, 2),
                 num_res_blocks=1, attention_levels=(), dropout=0.0)
    model.eval()
    return model


@pytest.fixture
def small_model() -> UNet:
    """Slightly wider, for the optimisation test — one that can actually learn."""
    return UNet(in_channels=1, base_channels=16, channel_mults=(1, 2),
                num_res_blocks=1, attention_levels=(1,), dropout=0.0)


def structured_batch(batch_size: int = 8, size: int = 28, seed: int = 0) -> torch.Tensor:
    """Smooth blobs in the range minus one to one — MNIST-shaped stand-in data.

    Deliberately not white noise: an overfitting test needs targets with spatial
    structure, or the only thing the model can learn is the mean.
    """
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(batch_size, 1, size // 4, size // 4, generator=generator)
    smooth = torch.nn.functional.interpolate(
        coarse, size=(size, size), mode="bilinear", align_corners=False
    )
    return smooth * 2.0 - 1.0


@pytest.fixture
def x0() -> torch.Tensor:
    """A batch of eight clean twenty-eight by twenty-eight images."""
    return structured_batch(8)
