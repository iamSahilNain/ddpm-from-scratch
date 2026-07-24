# DDPM from scratch

A denoising diffusion probabilistic model (Ho et al., 2020) on MNIST, in plain
PyTorch. No `diffusers`, no HuggingFace generative libraries — the point is that
the diffusion process is written out by hand.

The repo is scaffolding around a hole. The UNet, dataloader, training loop,
plotting helpers and test suite are complete; the four functions that *are* the
diffusion model are stubs with contracts and no bodies.

## Status

| Piece | State |
| --- | --- |
| `model.py` — UNet, sinusoidal timestep embedding | done |
| `data.py` — MNIST, normalised to the range −1…1 | done |
| `train.py` — AMP, EMA, checkpoint/resume, CSV logging | done |
| `viz.py` / `sample.py` — grids, forward-process strip, loss curve | done |
| `tests/` — 76 tests | done |
| `diffusion.py` — the four checkpoint functions | **stubs** |

## Setup

```bash
pip install -r requirements.txt
```

`torchvision` is only needed for the MNIST loaders. Everything else — the model,
the diffusion code, the whole test suite — runs without it.

## The checkpoint

Four functions in [`diffusion.py`](diffusion.py), each with a docstring stating
its exact contract (shapes, dtypes, indexing, and what the maths has to mean)
and a `raise NotImplementedError`:

1. `make_beta_schedule(kind, timesteps)` — the variance schedule everything else
   is derived from.
2. `q_sample(x0, t, noise)` — the closed-form forward marginal: jump from a
   clean image to any timestep in one shot.
3. `p_losses(model, x0, t)` — the simplified noise-prediction training objective.
4. `p_sample_loop(model, shape)` — ancestral sampling, noise to image.

Timesteps are zero-indexed throughout: `t = 0` is the first, gentlest noising
step and `t = T - 1` is the last. The tests assume this.

Suggested order: 1 → 2 → verify visually with the forward-process grid → 3 →
train something → 4. Sampling is the one that is hardest to debug from the loss
alone, so leave it until the first three are solid.

## Running the tests

```bash
pytest
```

Every test in `tests/test_model.py` passes today; everything else fails with
`NotImplementedError` until the stubs are filled in. That is the intended
starting state — the red tests are the specification.

Skip the two slow files (full thousand-step sampling chains and an optimisation
loop, about a minute together):

```bash
pytest -m "not slow"
```

What the suite checks, beyond shapes and finiteness:

- **`test_schedule.py`** — the cumulative products decrease monotonically, start
  near one, and end near zero. The last is what licenses sampling to begin from
  pure noise at all.
- **`test_q_sample.py`** — behavioural only. The first step barely changes the
  image; the last is standard Gaussian and has forgotten the input entirely;
  correlation with the clean image falls monotonically in between. Nothing here
  recomputes the closed form, so passing means the *process* is right rather
  than that it matches one particular arrangement of square roots.
- **`test_losses.py`** — a finite scalar, reduced as a mean rather than a sum,
  with gradients reaching every parameter including the timestep embedding.
- **`test_sampling.py`** — structural checks, plus one test with real teeth:
  sampling with a denoiser whose optimum is known in closed form has to return
  standard Gaussian samples. Sign errors, swapped alpha terms and a mis-scaled
  posterior variance all pass the structural checks and fail that one.
- **`test_overfit.py`** — the loss on a fixed batch of eight comes down from
  about 1.0 to under 0.25. Composition, not correctness of any single function.

Two deliberate notes on what the tests do *not* do:

- `tests/test_schedule.py` has one helper, `alpha_bars_from`, that re-derives the
  cumulative products from the betas. It has to: `make_beta_schedule` returns
  only the betas, and the properties worth asserting are properties of the
  products. It restates what the docstring already says in prose. Delete it and
  the four tests using it if you would rather work with nothing in front of you.
- No test asserts anything about the *magnitude* of samples from an untrained
  model. The reverse process is only contractive once the network actually
  predicts noise; with random weights a correct implementation drifts into the
  hundreds. A test forbidding that would fail on correct code.

## Training

```bash
python train.py --epochs 20 --batch-size 128
```

Writes to `runs/mnist/`: `checkpoints/` (rolling `last.pt` plus one per epoch),
`logs/loss.csv`, `samples/`, and the resolved `config.json`.

```bash
python train.py --resume runs/mnist/checkpoints/last.pt
python train.py --base-channels 32 --device cpu --no-amp   # smaller/slower box
python sample.py --checkpoint runs/mnist/checkpoints/last.pt -n 64
```

Mixed precision turns itself on only for CUDA. On MPS and CPU the speedup on a
model this small does not justify the numerics, so `--amp` is a no-op there.

Sampling defaults to the EMA weights. They are worth having: the live weights at
any given step look distinctly worse than the loss curve suggests.

The default model is about 6.8M parameters (3 levels, base width 64, attention
at 14×14 and 7×7). `--base-channels 32` gives 1.7M and trains acceptably.

Read the loss curve smoothed — `viz.plot_loss_curve` does this — because the raw
per-step trace is dominated by which timesteps happened to be drawn.

## Layout

```
diffusion.py   the four stubs — forward process, objective, sampler
model.py       UNet, sinusoidal timestep embedding, FiLM-conditioned res blocks
data.py        MNIST at [-1, 1], plus a synthetic batch for tests
train.py       training loop, EMA, AMP, checkpointing, CSV logging
sample.py      generate from a checkpoint
viz.py         image grids, forward-process strip, loss curve
tests/         pytest suite
NOTES.md       build log
```

## A note on the architecture

The residual blocks condition on the timestep with a learned scale and shift
applied *after* the group norm (the adaptive group norm of Nichol & Dhariwal,
2021), not by adding the embedding before it as the original DDPM code does.
This is not a stylistic preference. A per-channel constant added before a
normalisation layer is partly — and at narrow widths exactly — subtracted back
out by that layer, at which point the network is silently unconditional. It
still trains to a plausible loss and samples pure mush. This repo hit that
during construction: `test_the_timestep_actually_changes_the_output` caught it
with a max output difference of 1e-6 between `t = 0` and `t = 999`.

## References

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, 2020 —
  arXiv:2006.11239. The linear schedule endpoints and the simplified objective.
- Nichol, Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, 2021 —
  arXiv:2102.09672. Cosine schedule, adaptive group norm.
