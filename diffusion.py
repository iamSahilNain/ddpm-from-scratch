"""Forward and reverse diffusion processes for a DDPM (Ho et al., 2020).

This module is the *point* of the project. Everything else in the repo —
the UNet, the dataloader, the training loop, the plotting helpers — is
scaffolding that exists so that these four functions have something to be
plugged into. Two of them are now written: :func:`make_beta_schedule` and
:func:`q_sample`. :func:`p_losses` and :func:`p_sample_loop` are still
deliberately left unimplemented.

The docstrings below are contracts, not hints: they fix the shapes, dtypes,
indexing convention and mathematical meaning that the rest of the repo (and
`tests/`) assumes. Implement to the contract and the test suite will pass;
implement something self-consistent but different and it will not, which is
the useful part.

Indexing convention used throughout
-----------------------------------
Timesteps are zero-indexed: ``t = 0`` is the *first* (smallest) noising step
and ``t = timesteps - 1`` is the last, at which a sample should be
indistinguishable from standard Gaussian noise. Every schedule array is
therefore indexed by the same ``t`` that gets handed to the model.

Where the schedule comes from
-----------------------------
Only :func:`make_beta_schedule` takes a schedule length. The other three take
none, so they have to source one themselves — :func:`default_betas` below is
there for exactly that, and it memoises, so calling it per step is cheap. The
consequence is that the schedule length is global to a process: sampling runs
for ``DEFAULT_TIMESTEPS`` steps, and the test suite budgets for that by using
very small models and images rather than by shortening the chain.

A note on structure
-------------------
A class that precomputes the schedule buffers once (``Diffusion(betas)``) is a
perfectly good refactor, and arguably the better one. If you go that way, keep
these four names as thin module-level entry points, because that is the only
surface `tests/` and `train.py` touch.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Sequence

import torch
from torch import Tensor

__all__ = [
    "DEFAULT_TIMESTEPS",
    "make_beta_schedule",
    "default_betas",
    "q_sample",
    "p_losses",
    "p_sample_loop",
]

#: Number of diffusion steps used by Ho et al. for their reported results, and
#: the default everywhere in this repo that a schedule length is not given.
DEFAULT_TIMESTEPS = 1000


def make_beta_schedule(kind: str = "linear", timesteps: int = DEFAULT_TIMESTEPS) -> Tensor:
    """Build the variance schedule that defines the forward process.

    The schedule is the only place the forward process is parameterised. Every
    other quantity in DDPM is derived from it, so getting this wrong silently
    poisons everything downstream.

    Parameters
    ----------
    kind:
        Which family of schedule to build. ``"linear"`` is required — it is the
        one the paper uses and the one the tests assume by name. Other kinds
        (e.g. the cosine schedule of Nichol & Dhariwal, 2021) are optional; if
        you do not implement a requested kind, raise ``ValueError``.
    timesteps:
        Length ``T`` of the schedule.

    Returns
    -------
    Tensor
        Shape ``(timesteps,)``, dtype ``torch.float32``, on the CPU. Every entry
        must lie strictly inside the open interval from zero to one. For the
        linear kind the sequence must be strictly increasing.

    The linear schedule (Ho et al., 2020)
    -------------------------------------
    The betas are spaced evenly between two fixed endpoints: the first beta is
    one ten-thousandth (1e-4) and the last is two hundredths (0.02). Those
    constants are tuned for ``T = 1000``; they are small relative to the data
    scale so that each individual step perturbs the image only slightly, while
    a thousand of them compound into something Gaussian. If you instantiate a
    much shorter schedule with the same endpoints the process will no longer
    destroy the signal by the final step — for short schedules the endpoints
    have to be scaled up to compensate, and the tests exercising near-Gaussian
    behaviour at the last step use the full-length schedule for that reason.

    What is derived from it
    -----------------------
    Two derived sequences do all the work downstream, and it is worth building
    them here (or in whatever object caches this schedule) rather than
    recomputing them per call:

    ``alphas``
        The per-step *signal retention* coefficients: the complement of each
        beta. One alpha per timestep, same shape as ``betas``.
    ``alpha_bars``
        The running cumulative product of ``alphas`` from index zero up to and
        including index ``t``. This is the quantity that lets you jump from
        ``x_0`` to any ``x_t`` in a single shot instead of simulating the chain,
        and it is what :func:`q_sample` consumes.

    Two properties of ``alpha_bars`` are load-bearing and are asserted by the
    test suite: it must decrease monotonically in ``t``, and its final entry
    must be very close to zero. The second is what makes the reverse process
    able to start from pure noise at all — if the final ``alpha_bar`` is not
    negligible then ``x_T`` still carries image information that sampling has
    no way to supply.
    """
    # Truncating a float here would hand back a schedule of a length the caller
    # never asked for, so anything non-integral is rejected rather than coerced.
    if isinstance(timesteps, bool) or int(timesteps) != timesteps:
        raise ValueError(f"timesteps must be an integer, got {timesteps!r}")
    timesteps = int(timesteps)
    if timesteps < 1:
        raise ValueError(f"timesteps must be at least one, got {timesteps}")

    # Every tensor below is pinned to the CPU: the contract promises a CPU
    # tensor, and `torch.set_default_device` would otherwise silently move the
    # schedule — onto MPS, where the float64 cosine cannot even be built.
    if kind == "linear":
        return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32, device="cpu")

    if kind == "cosine":
        # Nichol & Dhariwal (2021), eq. 17. The offset keeps beta_0 away from
        # zero; f is evaluated at T + 1 points so that consecutive alpha_bars
        # can be divided out into per-step betas.
        offset = 0.008
        steps = torch.arange(timesteps + 1, dtype=torch.float64, device="cpu")
        f = torch.cos(((steps / timesteps) + offset) / (1.0 + offset) * math.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        betas = 1.0 - alpha_bars[1:] / alpha_bars[:-1]
        # The tail of the cosine schedule runs beta arbitrarily close to one,
        # which would collapse every later alpha_bar to zero outright.
        return betas.clamp(max=0.999).to(torch.float32)

    raise ValueError(f"unknown schedule kind {kind!r}; expected 'linear' or 'cosine'")


def q_sample(x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """Sample from the forward marginal: jump straight from ``x_0`` to ``x_t``.

    Parameters
    ----------
    x0:
        Clean images, shape ``(B, C, H, W)``, floating point, already
        normalised to the range from minus one to one.
    t:
        Shape ``(B,)``, dtype ``torch.int64``, each entry in the closed range
        from zero to ``T - 1``. Entries may differ across the batch — this is
        the normal case during training, where each image in a batch is noised
        to its own randomly drawn timestep. Broadcasting the per-sample scalar
        coefficients against a four-dimensional image tensor is the fiddly part
        of the implementation.
    noise:
        Standard Gaussian noise, same shape, dtype and device as ``x0``. It is
        passed in rather than drawn inside so that callers (and tests) can make
        the function deterministic.

    Returns
    -------
    Tensor
        ``x_t``, same shape, dtype and device as ``x0``.

    The forward marginal
    --------------------
    The whole reason DDPM is trainable is that the forward chain admits a
    closed form: conditioned on ``x_0``, the distribution of ``x_t`` is Gaussian
    for every ``t``, with no need to walk the chain step by step. Its mean is
    ``x_0`` scaled down by the square root of ``alpha_bar`` at that timestep,
    and its covariance is isotropic — a single scalar variance, equal to the
    complement of ``alpha_bar`` at that timestep, on every pixel independently.

    Note that the mean coefficient and the variance are tied: the squared mean
    scaling plus the variance is exactly one at every ``t``. That is what keeps
    the marginal at unit scale as ``t`` grows, and it is worth checking your
    implementation preserves it.

    The reparameterisation
    ----------------------
    Rather than calling into a sampler, draw the sample as its mean plus the
    supplied ``noise`` scaled by the standard deviation of that Gaussian. This
    is the standard reparameterisation trick: it makes ``x_t`` a deterministic,
    differentiable function of ``x_0`` with all the randomness pushed out into
    the externally supplied ``noise`` argument. Two consequences the tests lean
    on: at ``t = 0`` the result is ``x_0`` almost unchanged, because the first
    beta is tiny; and at ``t = T - 1`` the result is distributed as standard
    Gaussian almost regardless of ``x_0``, because the mean coefficient has
    decayed to nearly nothing.
    """
    if not x0.is_floating_point():
        raise TypeError(
            f"x0 must be floating point, got {x0.dtype}; an integer dtype would "
            "truncate every schedule coefficient to zero"
        )
    if t.dim() != 1:
        raise ValueError(f"t must be one-dimensional, shape (B,), got {tuple(t.shape)}")

    schedule = _derived(device=x0.device, dtype=x0.dtype)
    timesteps = schedule.alpha_bars.shape[0]
    # Torch would wrap a negative index round to the far end of the schedule,
    # turning "one step before the start" into "fully destroyed" in silence.
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= timesteps):
        raise IndexError(
            f"t must lie in [0, {timesteps}), got [{int(t.min())}, {int(t.max())}]; "
            f"the process-global schedule has {timesteps} steps"
        )

    index = t.to(schedule.sqrt_alpha_bars.device)
    view = (-1,) + (1,) * (x0.dim() - 1)

    mean_coefficient = schedule.sqrt_alpha_bars[index].view(view)
    std = schedule.sqrt_one_minus_alpha_bars[index].view(view)
    return mean_coefficient * x0 + std * noise


def p_losses(model: torch.nn.Module, x0: Tensor, t: Tensor) -> Tensor:
    """Training objective for one batch: the simplified noise-prediction loss.

    Parameters
    ----------
    model:
        A noise-prediction network. It is called as ``model(x_t, t)`` and
        returns a tensor the same shape as ``x_t``, interpreted as its estimate
        of the noise that was added. :class:`model.UNet` satisfies this.
    x0:
        Clean images, shape ``(B, C, H, W)``, normalised to the range from
        minus one to one.
    t:
        Shape ``(B,)``, dtype ``torch.int64``, entries in the closed range from
        zero to ``T - 1``. During training these are drawn uniformly at random,
        independently per image; this function does not draw them, it receives
        them.

    Returns
    -------
    Tensor
        A zero-dimensional (scalar) tensor. It must be finite, and it must be
        connected to ``model``'s parameters in the autograd graph — calling
        ``.backward()`` on it has to populate ``.grad`` on every parameter that
        participated.

    What the function does
    ----------------------
    Draw fresh standard Gaussian noise with the shape of ``x0``, noise the
    images to their respective timesteps with :func:`q_sample`, ask the model to
    recover the noise from the noised image and the timestep, and score the
    prediction against the noise that was actually used. The score is a mean
    squared error reduced over every axis — batch, channel and both spatial
    axes — down to a single number.

    Note that the noise is drawn *inside* this function. That means the target
    is not a fixed function of the inputs and the loss is stochastic even for a
    fixed batch and fixed ``t``; it also means the minimum attainable value is
    not zero for any model that has not memorised the training images. Do not
    expect an overfitting run to drive this to zero.

    Relation to the variational bound
    ---------------------------------
    The proper training objective is a variational upper bound on the negative
    log-likelihood, which decomposes into one KL term per timestep, each
    comparing the learned reverse step against the tractable forward posterior.
    Rewriting each of those terms in terms of noise prediction leaves a squared
    error like the one above, multiplied by a timestep-dependent weighting
    factor. The simplified objective is what you get by discarding that
    weighting — setting every term's weight to one. It is therefore *not* the
    bound, but a reweighted version of it that upweights the large-``t`` terms
    relative to the bound; Ho et al. found it both simpler and empirically
    better for sample quality, which is a slightly uncomfortable but very
    reproducible result.
    """
    # Drawn here rather than taken as an argument: the target has to be fresh
    # every call, or the model is scored against a noise sample it has already
    # seen for this image and the objective stops being an expectation.
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, t, noise)
    # `t` goes to the model as well as to q_sample. Without it the timestep MLP
    # sits outside the graph, and the network quietly learns one average
    # denoiser for every noise level.
    predicted_noise = model(x_t, t)
    return torch.nn.functional.mse_loss(predicted_noise, noise)


@torch.no_grad()
def p_sample_loop(
    model: torch.nn.Module,
    shape: Sequence[int],
    device: Optional[torch.device] = None,
) -> Tensor:
    """Run the full reverse process: pure noise in, a generated image out.

    Parameters
    ----------
    model:
        The trained noise-prediction network, called as ``model(x_t, t)``.
        Switching it to eval mode is the caller's job (:func:`viz.save_sample_grid`
        does it and restores afterwards); what this function owes is running the
        whole chain without building a graph.
    shape:
        The shape to generate, as ``(B, C, H, W)``. For MNIST at the settings in
        this repo that is a batch of one-channel twenty-eight by twenty-eight
        images.
    device:
        Where to allocate the chain. Defaults to the device the model's
        parameters live on.

    Returns
    -------
    Tensor
        Shape exactly equal to ``shape``, finite everywhere. Values should land
        roughly in the range from minus one to one — the same normalisation the
        training data uses — though nothing clamps them there by construction,
        and early in training they will not.

    The ancestral sampling recursion
    --------------------------------
    Start by drawing ``x_T`` as standard Gaussian noise of the requested shape.
    This is the step that :func:`make_beta_schedule` had to earn: it is only
    legitimate because the final ``alpha_bar`` is small enough that the forward
    marginal at ``T - 1`` really is approximately standard Gaussian.

    Then walk ``t`` downwards from ``T - 1`` to ``0``, and at each step produce
    ``x_{t-1}`` from ``x_t`` alone. The structure of a single step is:

    1. Ask the model for its noise estimate at the current ``x_t`` and ``t``.
    2. Invert the forward marginal to turn that estimate into an implied clean
       image — the ``x_0`` that, together with the predicted noise, would have
       produced this ``x_t``. This is an algebraic rearrangement of the
       relationship documented in :func:`q_sample`, nothing new.
    3. Plug that implied clean image into the forward process' posterior over
       ``x_{t-1}`` given both ``x_t`` and ``x_0``, which is Gaussian with a
       tractable mean, and take that mean as the centre of the step. In
       practice this collapses to a linear combination of ``x_t`` and the
       predicted noise, so steps two and three can be folded into one
       expression — but derive it in that order, because folding it up front is
       how sign errors get in.
    4. Add noise.

    Where the posterior variance enters
    -----------------------------------
    Step four is the only stochastic part of the reverse process, and the scale
    of the noise it adds is the posterior variance. Ho et al. fix it rather than
    learning it, and report two choices that work about equally well: ``beta``
    at the current timestep, or the exact variance of the forward posterior,
    which is that same ``beta`` scaled by a ratio built from the ``alpha_bar``
    values at the current and previous timesteps. The first is the upper bound
    on the reverse-step variance and the second is the lower bound; pick one
    and be consistent.

    The single carve-out: at the final step, when ``x_{-1}`` would be the
    returned image itself, no noise is added and the posterior mean is returned
    directly. Forgetting this leaves a visible layer of grain over every sample
    and is by far the most common bug in this function.
    """
    parameter_device = next(model.parameters()).device
    if device is None:
        device = parameter_device
    device = torch.device(device)
    if device != parameter_device:
        raise ValueError(
            f"cannot sample on {device} with a model whose parameters live on "
            f"{parameter_device}; move the model first"
        )

    schedule = _derived(device=device, dtype=torch.float32)
    timesteps = schedule.alpha_bars.shape[0]

    # x_T. Legitimate only because the final alpha_bar is negligible, which is
    # what make_beta_schedule had to earn. Pinned to `device` rather than left
    # to `torch.set_default_device`.
    x = torch.randn(tuple(shape), device=device)

    for step in reversed(range(timesteps)):
        t = torch.full((x.shape[0],), step, dtype=torch.long, device=device)
        predicted_noise = model(x, t)

        # Two steps, written as one. Inverting the forward marginal gives the
        # implied clean image, x0 = (x_t - sqrt(1 - abar) * eps) / sqrt(abar);
        # feeding that into the forward posterior's mean and cancelling the
        # sqrt(abar) terms collapses to the line below. Derived in that order —
        # folding it up front is how the sign errors get in.
        mean = (
            x
            - schedule.betas[step] / schedule.sqrt_one_minus_alpha_bars[step] * predicted_noise
        ) / schedule.alphas[step].sqrt()

        if step > 0:
            x = mean + schedule.posterior_variance[step].sqrt() * torch.randn_like(x)
        else:
            # The carve-out. posterior_variance[0] is already exactly zero, but
            # branching keeps the RNG stream from advancing on a draw that gets
            # multiplied away, which would make reproducibility depend on it.
            x = mean

    return x


# --------------------------------------------------------------------------- #
# Scaffolding below this line — not part of the checkpoint.
# --------------------------------------------------------------------------- #

_BETA_CACHE: dict[tuple[str, int], Tensor] = {}


def default_betas(kind: str = "linear", timesteps: int = DEFAULT_TIMESTEPS) -> Tensor:
    """Memoised :func:`make_beta_schedule`, for callers that want the default.

    Purely a convenience so that `train.py` and the plotting helpers do not
    rebuild the schedule on every call. Like every buffer in this module, the
    tensor it returns is shared and must not be mutated in place.
    """
    key = (kind, int(timesteps))
    if key not in _BETA_CACHE:
        _BETA_CACHE[key] = make_beta_schedule(kind, timesteps)
    return _BETA_CACHE[key]


# --------------------------------------------------------------------------- #
# Checkpoint work again below — the buffers the four functions above read from.
# --------------------------------------------------------------------------- #


class _Derived(NamedTuple):
    """The schedule and everything the forward and reverse processes read off it."""

    betas: Tensor
    alphas: Tensor
    alpha_bars: Tensor
    sqrt_alpha_bars: Tensor
    sqrt_one_minus_alpha_bars: Tensor
    #: ``alpha_bars`` shifted one step later, with one prepended for ``t = 0``,
    #: where "the previous timestep" is the clean image and no signal has been
    #: removed yet. Only the reverse process needs it.
    alpha_bars_prev: Tensor
    #: The exact variance of the forward posterior over ``x_{t-1}`` given both
    #: ``x_t`` and ``x_0`` — Ho et al.'s lower bound on the reverse-step
    #: variance. Zero at ``t = 0`` by construction.
    posterior_variance: Tensor


_DERIVED_CACHE: dict[tuple, _Derived] = {}


def _derived(
    kind: str = "linear",
    timesteps: int = DEFAULT_TIMESTEPS,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> _Derived:
    """Cached schedule buffers, already on the device and in the dtype asked for.

    The derivation runs in float64 on the CPU and is cast only at the end. Two
    reasons, one of which is *not* the obvious one:

    - Deriving in the caller's dtype would take the cumulative product there
      too. The linear schedule survives that in float16 — its final
      ``alpha_bar`` is 4.0e-05, comfortably above the smallest subnormal — but
      the cosine schedule does not: its final ``alpha_bar`` is 2.4e-09 and
      flushes to exactly zero, which would make the mean coefficient vanish at
      the last step. Taking the square roots before the cast rescues it, since
      1.5e-05 is representable where its square is not.
    - float64 keeps the cumulative product accurate for float64 callers, who
      would otherwise silently receive float32-accurate coefficients.

    Note that autocast is not the case this protects against: under
    ``torch.autocast`` the tensors handed to :func:`q_sample` stay float32, so
    the low-precision path is reached only by an explicitly halved caller.

    The returned buffers are shared across every caller in the process and must
    be treated as read-only. Derive from them freely — ``roll``, indexing and
    arithmetic all copy — but never write into them in place, or the schedule
    is corrupted for everything downstream with no error raised.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    key = (kind, int(timesteps), str(device), dtype)
    if key not in _DERIVED_CACHE:
        betas = default_betas(kind, timesteps).to(torch.float64)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.ones(1, dtype=torch.float64, device="cpu"), alpha_bars[:-1]]
        )
        buffers = _Derived(
            betas=betas,
            alphas=alphas,
            alpha_bars=alpha_bars,
            sqrt_alpha_bars=alpha_bars.sqrt(),
            sqrt_one_minus_alpha_bars=(1.0 - alpha_bars).sqrt(),
            alpha_bars_prev=alpha_bars_prev,
            posterior_variance=betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars),
        )
        # `.to` is a no-op returning the same object when the dtype and device
        # already match, which would alias these buffers to `_BETA_CACHE` and
        # let one caller's in-place edit poison the schedule process-wide.
        _DERIVED_CACHE[key] = _Derived(
            *(buffer.to(device=device, dtype=dtype).clone() for buffer in buffers)
        )
    return _DERIVED_CACHE[key]
