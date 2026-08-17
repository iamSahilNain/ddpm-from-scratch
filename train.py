"""Training loop for the DDPM: mixed precision, checkpointing, CSV loss log.

The objective it optimises is :func:`diffusion.p_losses`, and the sample
sheets it writes go through :func:`diffusion.p_sample_loop`.

Usage:
    python train.py --epochs 20 --batch-size 128
    python train.py --resume runs/mnist/checkpoints/last.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass

import torch

from data import get_mnist_dataloader
from model import UNet, count_parameters


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    base_channels: int = 64
    channel_mults: tuple = (1, 2, 2)
    num_res_blocks: int = 2
    dropout: float = 0.1

    timesteps: int = 1000
    ema_decay: float = 0.999

    data_root: str = "./data"
    out_dir: str = "./runs/mnist"
    num_workers: int = 2
    seed: int = 0

    amp: bool = True
    device: str = "auto"
    log_every: int = 50
    sample_every: int = 5  # epochs; 0 disables
    num_samples: int = 16


def pick_device(requested: str = "auto") -> torch.device:
    """Resolve ``--device auto`` to the best thing actually available."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device, requested: bool) -> tuple[bool, torch.dtype]:
    """Decide whether to actually use autocast, and in which dtype.

    Mixed precision is only a clear win on CUDA, where the gradient scaler and
    fp16 tensor cores both exist. MPS autocast is available, but the speedup on
    a model this small is noise and the numerics are less well trodden, so it
    is off there unconditionally — there is no flag to force it on — and off on
    CPU too. Only ``--no-amp`` exists, and only CUDA is affected by it.
    """
    if not requested or device.type == "cpu":
        return False, torch.float32
    if device.type == "cuda":
        return True, torch.float16
    return False, torch.float32


# --------------------------------------------------------------------------- #
# EMA
# --------------------------------------------------------------------------- #


class EMA:
    """Exponential moving average of the model weights.

    Sampling from an EMA copy rather than the live weights is close to free and
    makes a large, immediately visible difference to DDPM sample quality — the
    live weights bounce around enough that individual checkpoints look worse
    than the training loss suggests they should.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone().float()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=1 - self.decay)

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name].to(param.dtype))

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = state["decay"]
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}


# --------------------------------------------------------------------------- #
# Logging & checkpoints
# --------------------------------------------------------------------------- #


class CSVLogger:
    """Append-only CSV of per-step training stats.

    Flushed every write. A training run that dies at hour three should still
    leave a readable loss curve behind.
    """

    FIELDS = ("step", "epoch", "loss", "lr", "seconds")

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        is_new = not os.path.exists(path)
        self.handle = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.FIELDS)
        if is_new:
            self.writer.writeheader()
            self.handle.flush()

    def log(self, **row) -> None:
        self.writer.writerow({key: row.get(key, "") for key in self.FIELDS})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def save_checkpoint(path: str, **state) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic, so an interrupted save cannot corrupt last.pt


def build_model(config: TrainConfig) -> UNet:
    return UNet(
        in_channels=1,
        base_channels=config.base_channels,
        channel_mults=tuple(config.channel_mults),
        num_res_blocks=config.num_res_blocks,
        dropout=config.dropout,
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


def train(config: TrainConfig) -> str:
    from diffusion import DEFAULT_TIMESTEPS, p_losses  # late so --help stays fast

    # The schedule length is global to the process (see diffusion.py's module
    # docstring): q_sample and p_sample_loop always use DEFAULT_TIMESTEPS. A
    # shorter --timesteps would train the model on a fraction of the noise
    # range and then sample it from pure noise, producing mush with no error
    # anywhere, so refuse the flag rather than honour it halfway.
    if config.timesteps != DEFAULT_TIMESTEPS:
        raise ValueError(
            f"--timesteps {config.timesteps} is not supported: the diffusion "
            f"schedule is process-global at {DEFAULT_TIMESTEPS} steps. Training "
            "on a different length would silently mismatch the sampler."
        )

    torch.manual_seed(config.seed)
    device = pick_device(config.device)
    use_amp, amp_dtype = amp_settings(device, config.amp)

    checkpoint_dir = os.path.join(config.out_dir, "checkpoints")
    sample_dir = os.path.join(config.out_dir, "samples")
    log_path = os.path.join(config.out_dir, "logs", "loss.csv")
    resume_path = getattr(config, "_resume", None)
    if resume_path and not os.path.exists(resume_path):
        # Checked before the dataloader, which on a real MNIST setup can take
        # long enough that a typo'd path is an expensive way to learn to type.
        raise FileNotFoundError(f"--resume {resume_path} does not exist")

    os.makedirs(config.out_dir, exist_ok=True)

    loader = get_mnist_dataloader(
        batch_size=config.batch_size,
        root=config.data_root,
        train=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config).to(device)
    ema = EMA(model, decay=config.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    logger = CSVLogger(log_path)

    start_epoch, global_step = 0, 0
    if resume_path:
        start_epoch, global_step = load_for_resume(
            resume_path, model, ema, optimizer, scaler, device
        )
        # --epochs is an absolute target, not "this many more". Without this
        # check, resuming a finished run prints the banners, trains nothing and
        # exits zero, which is indistinguishable from success.
        if start_epoch >= config.epochs:
            raise ValueError(
                f"checkpoint is already at epoch {start_epoch} and --epochs is "
                f"{config.epochs}, so there is nothing to do. --epochs is an "
                f"absolute target: pass --epochs {start_epoch + 10} to train ten more."
            )
        # The optimizer and EMA restore their checkpointed hyperparameters, so
        # these flags would be silently ignored while being recorded as though
        # they had applied.
        defaults = TrainConfig()
        for flag, value, restored in (
            ("--lr", config.lr, optimizer.param_groups[0]["lr"]),
            ("--ema-decay", config.ema_decay, ema.decay),
        ):
            if value != restored and value != getattr(defaults, flag[2:].replace("-", "_")):
                raise ValueError(
                    f"{flag}={value} cannot be applied on resume; the checkpoint "
                    f"restores {restored}. Omit it, or start a fresh run."
                )

    # Written only once everything that can fail has succeeded, so a config.json
    # on disk always describes a run that actually started.
    with open(os.path.join(config.out_dir, "config.json"), "w") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)

    print(f"device={device}  amp={use_amp}({amp_dtype})  params={count_parameters(model):,}")
    print(f"batches/epoch={len(loader)}  out_dir={config.out_dir}")

    started = time.time()
    model.train()
    for epoch in range(start_epoch, config.epochs):
        epoch_loss, batches = 0.0, 0
        for images, _labels in loader:
            images = images.to(device, non_blocking=True)
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = p_losses(model, images, t)

            scaler.scale(loss).backward()
            if config.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            value = float(loss.detach())
            epoch_loss += value
            batches += 1
            global_step += 1

            if global_step % config.log_every == 0:
                logger.log(
                    step=global_step,
                    epoch=epoch,
                    loss=f"{value:.6f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.3e}",
                    seconds=f"{time.time() - started:.1f}",
                )
                print(f"epoch {epoch}  step {global_step}  loss {value:.4f}")

        mean_loss = epoch_loss / max(batches, 1)
        print(f"[epoch {epoch}] mean loss {mean_loss:.4f}  ({time.time() - started:.0f}s elapsed)")

        state = {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch + 1,
            "global_step": global_step,
            "config": asdict(config),
            # Without this a resumed run reseeds from config.seed and replays
            # the exact timestep and noise draws it already trained on. A run
            # resumed once per epoch would see the same few hundred timestep
            # vectors for ever, with a completely normal-looking loss curve.
            "rng_state": torch.get_rng_state(),
        }
        save_checkpoint(os.path.join(checkpoint_dir, "last.pt"), **state)
        save_checkpoint(os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pt"), **state)

        if config.sample_every and (epoch + 1) % config.sample_every == 0:
            write_samples(model, ema, config, sample_dir, epoch, device)

    logger.close()
    return checkpoint_dir


def write_samples(model, ema, config, sample_dir, epoch, device) -> None:
    """Draw a sheet of samples from the EMA weights without disturbing training."""
    from viz import save_sample_grid

    backup = {name: param.detach().clone() for name, param in model.named_parameters()}
    ema.copy_to(model)
    try:
        path = save_sample_grid(
            model,
            os.path.join(sample_dir, f"epoch_{epoch:04d}.png"),
            num_samples=config.num_samples,
            device=device,
        )
        print(f"  wrote {path}")
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(backup[name])
        model.train()


def load_for_resume(path, model, ema, optimizer, scaler, device) -> tuple[int, int]:
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    ema.load_state_dict(state["ema"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    if "rng_state" in state:
        # Continue the random stream rather than restarting it. Checkpoints
        # written before this was saved simply carry on reseeded.
        torch.set_rng_state(state["rng_state"].to(torch.uint8).cpu())
    else:
        print("  (checkpoint predates RNG-state saving; the noise stream restarts)")
    print(f"resumed from {path} at epoch {state['epoch']}, step {state['global_step']}")
    return state["epoch"], state["global_step"]


def parse_args(argv=None) -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--base-channels", type=int, default=defaults.base_channels)
    parser.add_argument("--num-res-blocks", type=int, default=defaults.num_res_blocks)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--timesteps", type=int, default=defaults.timesteps)
    parser.add_argument("--ema-decay", type=float, default=defaults.ema_decay)
    parser.add_argument("--data-root", default=defaults.data_root)
    parser.add_argument("--out-dir", default=defaults.out_dir)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default=defaults.device, help="auto|cpu|cuda|mps")
    parser.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--sample-every", type=int, default=defaults.sample_every,
                        help="epochs between sample sheets; 0 disables")
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--resume", default=None, help="path to a checkpoint to continue")
    args = parser.parse_args(argv)

    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        base_channels=args.base_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        timesteps=args.timesteps,
        ema_decay=args.ema_decay,
        data_root=args.data_root,
        out_dir=args.out_dir,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        device=args.device,
        log_every=args.log_every,
        sample_every=args.sample_every,
        num_samples=args.num_samples,
    )
    config._resume = args.resume
    return config


if __name__ == "__main__":
    train(parse_args())
