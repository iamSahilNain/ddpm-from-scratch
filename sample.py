"""Generate images from a trained checkpoint.

Usage:
    python sample.py --checkpoint runs/mnist/checkpoints/last.pt -n 64
    python sample.py --checkpoint runs/mnist/checkpoints/last.pt --no-ema
    python sample.py --checkpoint runs/mnist/checkpoints/last.pt --forward-grid
"""

from __future__ import annotations

import argparse
import os

import torch

from model import UNet
from viz import save_forward_process_grid, save_sample_grid


def load_model(checkpoint_path: str, device: torch.device, use_ema: bool = True) -> UNet:
    """Rebuild the network from a checkpoint's stored config and load weights.

    Defaults to the EMA weights, which is nearly always what you want to look
    at — the live weights at any given step are noticeably worse than the
    training loss implies.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config", {})

    model = UNet(
        in_channels=1,
        base_channels=config.get("base_channels", 64),
        channel_mults=tuple(config.get("channel_mults", (1, 2, 2))),
        num_res_blocks=config.get("num_res_blocks", 2),
        dropout=config.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state["model"])

    if use_ema and "ema" in state:
        shadow = state["ema"]["shadow"]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in shadow:
                    param.copy_(shadow[name].to(param.dtype).to(device))

    model.eval()
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("-n", "--num-samples", type=int, default=64)
    parser.add_argument("--out", default="samples/sample.png")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--no-ema", action="store_true", help="use the live weights")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--forward-grid",
        action="store_true",
        help="also write a forward-process strip, to eyeball q_sample",
    )
    args = parser.parse_args(argv)

    from train import pick_device

    device = pick_device(args.device)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    model = load_model(args.checkpoint, device, use_ema=not args.no_ema)
    path = save_sample_grid(model, args.out, num_samples=args.num_samples, device=device)
    print(f"wrote {path}")

    if args.forward_grid:
        from data import get_mnist_dataset

        dataset = get_mnist_dataset(train=False)
        batch = torch.stack([dataset[i][0] for i in range(4)])
        strip = os.path.join(os.path.dirname(args.out) or ".", "forward_process.png")
        print(f"wrote {save_forward_process_grid(batch, strip)}")


if __name__ == "__main__":
    main()
