"""Plotting helpers: image grids, forward-process strips, sample sheets.

Two of these call into :mod:`diffusion`, so they depend on the forward and
reverse processes being implemented; both are. :func:`make_grid` and
:func:`save_image_grid` work standalone and are useful for eyeballing the
dataloader on its own.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Sequence

import torch
from torch import Tensor

from data import denormalize

__all__ = [
    "make_grid",
    "save_image_grid",
    "save_forward_process_grid",
    "save_sample_grid",
    "plot_loss_curve",
]


def make_grid(images: Tensor, nrow: Optional[int] = None, padding: int = 2, pad_value: float = 0.0) -> Tensor:
    """Tile a batch of images into a single image.

    Args:
        images: Shape ``(B, C, H, W)``, values in the range zero to one.
        nrow: Images per row. Defaults to the square root of the batch, which
            gives a roughly square sheet.
        padding: Pixels of gutter between and around tiles.
        pad_value: Colour of the gutter, in the same zero-to-one range.

    Returns:
        Shape ``(C, H_grid, W_grid)``.
    """
    if images.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(images.shape)}")

    images = images.detach().cpu().float()
    batch, channels, height, width = images.shape
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(batch)))
    ncol = int(math.ceil(batch / nrow))

    grid_h = ncol * (height + padding) + padding
    grid_w = nrow * (width + padding) + padding
    grid = images.new_full((channels, grid_h, grid_w), pad_value)

    for index in range(batch):
        row, col = divmod(index, nrow)
        top = row * (height + padding) + padding
        left = col * (width + padding) + padding
        grid[:, top : top + height, left : left + width] = images[index]
    return grid


def _to_pil(grid: Tensor):
    from PIL import Image

    array = (grid.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    if array.shape[-1] == 1:
        return Image.fromarray(array[..., 0], mode="L")
    return Image.fromarray(array, mode="RGB")


def save_image_grid(
    images: Tensor,
    path: str,
    nrow: Optional[int] = None,
    normalized: bool = True,
    scale: int = 1,
) -> str:
    """Write a batch of images to disk as one tiled PNG.

    Args:
        images: Shape ``(B, C, H, W)``.
        path: Destination. Parent directories are created.
        nrow: Images per row.
        normalized: True if ``images`` are in the training range of minus one to
            one and need inverting first; False if they are already zero to one.
        scale: Integer nearest-neighbour upscale, since a twenty-eight pixel
            tile is hard to judge at native size.

    Returns:
        The path written.
    """
    if normalized:
        images = denormalize(images)
    grid = make_grid(images, nrow=nrow)
    image = _to_pil(grid)
    if scale > 1:
        from PIL import Image

        image = image.resize(
            (image.width * scale, image.height * scale), resample=Image.NEAREST
        )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    image.save(path)
    return path


def save_forward_process_grid(
    x0: Tensor,
    path: str,
    timesteps: Optional[Sequence[int]] = None,
    total_timesteps: Optional[int] = None,
    seed: Optional[int] = 0,
    scale: int = 2,
) -> str:
    """Show the forward process: one row per image, one column per timestep.

    The single most useful diagnostic for :func:`diffusion.q_sample`. A correct
    implementation gives a left column that is visually identical to the input
    and a right column that is indistinguishable from static, with the digit
    dissolving smoothly in between. The two failure modes it catches instantly
    are a swapped pair of coefficients (noise appears immediately and then the
    image comes *back*) and an off-by-one in the schedule indexing (the last
    column still faintly shows the digit).

    Expect the dissolve to look front-loaded: with the linear schedule and
    columns spaced evenly in t, the digit is already hard to make out a third of
    the way along and every remaining column looks like static. That is a real
    property of the linear schedule, not a bug in your code — it spends most of
    its length on images that have already been destroyed, which is precisely
    the observation the cosine schedule of Nichol & Dhariwal was introduced to
    fix. Passing explicit ``timesteps`` weighted towards the start gives a more
    informative picture.

    Args:
        x0: Clean images, shape ``(B, C, H, W)``, in the range minus one to one.
        path: Destination PNG.
        timesteps: Which timesteps to show. Defaults to eight roughly evenly
            spaced steps spanning the schedule.
        total_timesteps: Schedule length, used only to pick the defaults.
        seed: Fixes the noise so repeated calls are comparable.
        scale: Integer upscale of the written PNG.

    Returns:
        The path written.
    """
    from diffusion import DEFAULT_TIMESTEPS, q_sample

    total = total_timesteps or DEFAULT_TIMESTEPS
    if total != DEFAULT_TIMESTEPS:
        # q_sample indexes the process-global schedule, so picking columns from
        # a different length would noise t=99 of a nominal 100-step run with the
        # 1000-step coefficients — a last column still showing the digit, which
        # is exactly the bug signature this function advertises catching.
        raise ValueError(
            f"total_timesteps={total} does not match the process-global schedule "
            f"of {DEFAULT_TIMESTEPS} steps; the strip would misreport."
        )
    if timesteps is None:
        timesteps = torch.linspace(0, total - 1, steps=8).round().long().tolist()

    generator = torch.Generator(device="cpu").manual_seed(seed) if seed is not None else None

    rows = []
    for image in x0:
        image = image.unsqueeze(0)
        for step in timesteps:
            noise = torch.randn(image.shape, generator=generator).to(image.device, image.dtype)
            t = torch.full((1,), int(step), dtype=torch.long, device=image.device)
            rows.append(q_sample(image, t, noise).squeeze(0))

    return save_image_grid(
        torch.stack(rows), path, nrow=len(timesteps), normalized=True, scale=scale
    )


def save_sample_grid(
    model,
    path: str,
    num_samples: int = 16,
    image_size: int = 28,
    channels: int = 1,
    device: Optional[torch.device] = None,
    scale: int = 3,
) -> str:
    """Generate images with the reverse process and write them as a grid.

    Args:
        model: A trained noise-prediction network.
        path: Destination PNG.
        num_samples: How many images to draw.
        image_size: Spatial size to generate at.
        channels: Image channels.
        device: Where to run. Defaults to the model's device.
        scale: Integer upscale of the written PNG.

    Returns:
        The path written.
    """
    from diffusion import p_sample_loop

    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()
    try:
        samples = p_sample_loop(
            model, (num_samples, channels, image_size, image_size), device=device
        )
    finally:
        model.train(was_training)

    return save_image_grid(samples, path, normalized=True, scale=scale)


def plot_loss_curve(csv_path: str, path: str, smooth: int = 50) -> str:
    """Plot the training loss from the CSV that `train.py` writes.

    The raw per-step curve for DDPM is extremely noisy — the loss depends on
    which timesteps happened to be drawn for that batch, and the objective's
    scale varies a lot across ``t`` — so a running mean is the only way to see
    whether anything is actually improving. Do not read the raw trace.

    Args:
        csv_path: The ``loss.csv`` produced by training.
        path: Destination PNG.
        smooth: Window of the running mean, in steps.

    Returns:
        The path written.
    """
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, losses = [], []
    with open(csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))

    if not steps:
        raise ValueError(f"no rows in {csv_path}")

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(steps, losses, linewidth=0.5, alpha=0.3, label="per step")
    if len(losses) >= smooth > 1:
        kernel = torch.ones(smooth) / smooth
        smoothed = torch.conv1d(
            torch.tensor(losses).view(1, 1, -1), kernel.view(1, 1, -1)
        ).view(-1)
        axis.plot(steps[smooth - 1 :], smoothed.tolist(), linewidth=1.5, label=f"mean over {smooth}")
    axis.set_xlabel("step")
    axis.set_ylabel("loss")
    axis.legend()
    figure.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path
