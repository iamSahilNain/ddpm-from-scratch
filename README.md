# DDPM from scratch

A denoising diffusion probabilistic model (Ho et al., 2020) on MNIST, in plain
PyTorch. No `diffusers`, no HuggingFace generative libraries — the point is that
the diffusion process is written out by hand.

The repo was scaffolding around a hole: the UNet, dataloader, training loop,
plotting helpers and test suite were complete, and the four functions that
*are* the diffusion model were stubs with contracts and no bodies. All four are
now written. The whole suite is green and the reverse process generates images.

## Status

| Piece | State |
| --- | --- |
| `model.py` — UNet, sinusoidal timestep embedding | done |
| `data.py` — MNIST, normalised to the range −1…1 | done |
| `train.py` — AMP, EMA, checkpoint/resume, CSV logging | done |
| `viz.py` / `sample.py` — grids, forward-process strip, loss curve | done |
| `tests/` — 104 tests | done |
| `diffusion.py` — schedule, forward marginal | done |
| `diffusion.py` — training objective, sampler | done |

## Setup

```bash
pip install -r requirements.txt
```

`torchvision` is only needed for the MNIST loaders. Everything else — the model,
the diffusion code, the whole test suite — runs without it.

## The four functions

Four functions in [`diffusion.py`](diffusion.py), each with a docstring stating
its exact contract (shapes, dtypes, indexing, and what the maths has to mean):

1. `make_beta_schedule(kind, timesteps)` — the variance schedule everything else
   is derived from. Linear (Ho et al.) and cosine (Nichol & Dhariwal).
2. `q_sample(x0, t, noise)` — the closed-form forward marginal: jump from a
   clean image to any timestep in one shot.
3. `p_losses(model, x0, t)` — the simplified noise-prediction training
   objective. Four lines on top of `q_sample`.
4. `p_sample_loop(model, shape)` — ancestral sampling, noise to image.

The reverse step uses the **exact forward posterior variance**,
`β̃ₜ = βₜ(1−ᾱₜ₋₁)/(1−ᾱₜ)` — Ho et al.'s lower bound, not the `βₜ` upper bound.
They report the two performing about equally; this one is picked because it
falls to exactly zero at `t = 0`, so the "add no noise on the last step"
carve-out is what the algebra says rather than a special case bolted on. The
loop still branches on `t > 0` explicitly, to keep a pointless draw from
advancing the RNG stream and making reproducibility depend on it.

The derived quantities the schedule feeds — `alphas`, `alpha_bars`, the two
square-root terms, `alpha_bars_prev` and `posterior_variance` — are cached per
`(kind, timesteps, device, dtype)` by the private `_derived` helper, which is
what all four functions read. The derivation runs in float64 on the
CPU and is cast only at the end, so the square roots are taken before any
precision is lost. That matters for the cosine schedule, whose final
`alpha_bar` of 2.4e-09 flushes to exactly zero in float16 while its square root
of 4.9e-05 survives; the linear schedule bottoms out at 4.0e-05 and is never in
danger. The buffers are shared process-wide and must be treated as read-only.

Timesteps are zero-indexed throughout: `t = 0` is the first, gentlest noising
step and `t = T - 1` is the last. The tests assume this.

They were built in the order 1 → 2 → verify visually with the forward-process
grid → 3 → train something → 4, which is the order to redo them in if you strip
the bodies out to work through it yourself. Sampling is the one that is hardest
to debug from the loss alone, so it goes last.

## Running the tests

```bash
pytest
```

Everything passes: **102 passed, 2 skipped** out of 104 collected. The two skips
are the optional `quadratic` and `sigmoid` schedules, which correctly raise
`ValueError`.

Skip the slow tests (full thousand-step sampling chains and an optimisation
loop — two whole files plus one test in `test_derived.py`, about two minutes
together):

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
  than that it matches one particular arrangement of square roots. Be clear
  about what that costs: mutation testing says `mean = alpha_bar` instead of
  `sqrt(alpha_bar)`, a `+1` index shift, and `q_sample` running on the cosine
  schedule while everything else runs on linear all pass all thirteen tests.
  The behavioural suite specifies the process; it does not pin the function.
- **`test_derived.py`** — which is what pins it, now that the functions exist.
  Recomputes the closed forms in float64, asserts `q_sample` and `p_losses`
  read the same schedule `make_beta_schedule` produces, cross-checks the folded
  reverse step against Ho et al. eq. 7 written out the long way, and covers the
  cache and the input contracts. All three mutants above die here. Kept in a
  separate file so the behavioural suites stay specifications rather than
  regression nets.
- **`test_losses.py`** — a finite scalar, reduced as a mean rather than a sum,
  with gradients reaching every parameter including the timestep embedding.
  Worth knowing its limit: `test_gradients_reach_the_timestep_embedding` does
  not catch conditioning being dropped. Replace `model(x_t, t)` with
  `model(x_t, zeros)` and the embedding still receives gradient, so the whole
  file stays green — only `test_derived.py` notices.
- **`test_derived.py`** (again, for the reverse process) — this is where the
  recursion is actually guarded. Two independent checks: the unfolded
  recomputation, and a **point-mass oracle** whose optimal denoiser is analytic
  and whose reverse chain must contract unit variance down to nothing and land
  on the point. That second one exists because the Gaussian oracle above is
  blind to whole classes of coefficient error, whereas a point mass misses them
  by orders of magnitude. Mutants caught by nothing else in the suite: the
  missing carve-out, the loop running forwards, the loop skipping its final
  step, dropped timestep conditioning, timesteps permuted across the batch, and
  three ways of clamping the schedule so part of it is never trained.
- **`test_sampling.py`** — structural checks, plus a closed-form oracle on
  standard Gaussian data, currently returning mean −0.017 and std 0.998
  against tolerances of 0.10 and 0.15. It is weaker than it looks and than an
  earlier version of this README claimed: it catches a sign error and a badly
  mis-scaled variance, but measurement says it catches **none** of seven
  swapped-alpha and off-by-one mutants, all of which land between std 0.9835
  and 1.0052. N(0, I) is the wrong probe, because Ho et al. note `βₜ` is
  optimal for exactly that data while this code uses `β̃ₜ`, so the two branches
  agree most closely precisely where the test measures.
- **`test_overfit.py`** — the loss on a fixed batch of eight comes down from
  about 1.0 to under 0.25. Composition, not correctness of any single function.

Two deliberate notes on what the tests do *not* do:

- `tests/test_schedule.py` has one helper, `alpha_bars_from`, that re-derives the
  cumulative products from the betas. It has to: `make_beta_schedule` returns
  only the betas, and the properties worth asserting are properties of the
  products. It restates what the docstring already says in prose. Delete it and
  the five tests using it if you would rather work with nothing in front of you.
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
model this small does not justify the numerics, so it stays off there whether or
not `--no-amp` is passed.

Sampling defaults to the EMA weights. They are worth having: the live weights at
any given step look distinctly worse than the loss curve suggests.

The default model is 7,278,401 parameters (3 levels, base width 64, attention
at 14×14 and 7×7). `--base-channels 32` gives 1,826,721 and trains acceptably.

Read the loss curve smoothed — `viz.plot_loss_curve` does this — because the raw
per-step trace is dominated by which timesteps happened to be drawn.

## Layout

```
diffusion.py   schedule, forward marginal, objective, ancestral sampler
model.py       UNet, sinusoidal timestep embedding, FiLM-conditioned res blocks
data.py        MNIST at [-1, 1], plus a synthetic batch for tests
train.py       training loop, EMA, AMP, checkpointing, CSV logging
sample.py      generate from a checkpoint
viz.py         image grids, forward-process strip, loss curve
tests/         pytest suite; test_derived.py pins the forward process
NOTES.md       build log
```

## A note on the architecture

The residual blocks condition on the timestep with a learned scale and shift
applied *after* the group norm (the mechanism described in Nichol & Dhariwal
2021's appendix, and named AdaGN a few months later in Dhariwal & Nichol 2021),
not by adding the embedding before it as the original DDPM code does. This is
not a stylistic preference. A per-channel constant added before a normalisation
layer is partly subtracted back out by that layer, at which point the network
is silently unconditional. It still trains to a plausible loss and samples pure
mush. This repo hit that during construction:
`test_the_timestep_actually_changes_the_output` caught it with a max output
difference of 1e-6 between `t = 0` and `t = 999`.

How complete the cancellation is depends on the grouping, and the 1e-6 figure
is the worst case: it needs one channel per group, where group norm degenerates
into instance norm and the constant is removed exactly. At the groupings
`_group_norm` can actually produce — it enforces at least four channels per
group — a pre-norm constant survives, with a t=0 versus t=999 difference around
0.14. So the trap is narrower than the anecdote suggests, and `min_per_group`
is now what keeps the repo out of it.

## References

- Ho, Jain, Abbeel, *Denoising Diffusion Probabilistic Models*, 2020 —
  arXiv:2006.11239. The linear schedule endpoints and the simplified objective.
- Nichol, Dhariwal, *Improved Denoising Diffusion Probabilistic Models*, 2021 —
  arXiv:2102.09672. Cosine schedule (eq. 17), and the post-norm scale-and-shift
  conditioning in appendix A.
- Dhariwal, Nichol, *Diffusion Models Beat GANs on Image Synthesis*, 2021 —
  arXiv:2105.05233. Where that conditioning is actually named AdaGN.
