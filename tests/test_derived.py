"""The schedule cache behind ``q_sample``, and the input contracts it enforces.

Deliberately a different kind of test from ``test_q_sample.py``. That file is
behavioural on purpose — it never restates the closed form, so that passing it
means the forward *process* is right rather than that it matches one particular
arrangement of square roots. That is the right stance for a file whose job is to
specify an unwritten function.

This file has the opposite job: the function is written, and these tests pin it
so it cannot drift. They do recompute the closed form, and they do assert that
``q_sample`` reads the same schedule the rest of the repo does. Both are things
the behavioural suite cannot see — a ``q_sample`` silently running on the cosine
schedule, or using ``alpha_bar`` where it means ``sqrt(alpha_bar)``, passes all
thirteen tests over there.
"""

from __future__ import annotations

import pytest
import torch

from conftest import structured_batch
from diffusion import (
    DEFAULT_TIMESTEPS,
    _derived,
    default_betas,
    make_beta_schedule,
    q_sample,
)

LAST = DEFAULT_TIMESTEPS - 1


# --------------------------------------------------------------------------- #
# The cache agrees with the schedule it is derived from
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", ["linear", "cosine"])
def test_buffers_match_the_schedule_they_come_from(kind):
    schedule = _derived(kind, DEFAULT_TIMESTEPS, dtype=torch.float64)
    betas = make_beta_schedule(kind, DEFAULT_TIMESTEPS).to(torch.float64)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    assert torch.allclose(schedule.betas, betas)
    assert torch.allclose(schedule.alphas, 1.0 - betas)
    assert torch.allclose(schedule.alpha_bars, alpha_bars)
    assert torch.allclose(schedule.sqrt_alpha_bars, alpha_bars.sqrt())
    assert torch.allclose(schedule.sqrt_one_minus_alpha_bars, (1.0 - alpha_bars).sqrt())


def test_q_sample_uses_the_default_linear_schedule():
    """The regression pin: q_sample on any other schedule fails here.

    Swapping the kind under ``q_sample`` leaves every behavioural test green,
    because a cosine forward process is also a perfectly well-behaved forward
    process — just not the one the sampler and the training loop assume.
    """
    x0 = structured_batch(4, seed=11).to(torch.float64)
    noise = torch.randn_like(x0)
    alpha_bars = torch.cumprod(1.0 - default_betas().to(torch.float64), dim=0)

    for step in [0, 1, 250, 700, LAST]:
        t = torch.full((4,), step, dtype=torch.long)
        expected = alpha_bars[step].sqrt() * x0 + (1.0 - alpha_bars[step]).sqrt() * noise
        assert torch.allclose(q_sample(x0, t, noise), expected, atol=1e-12), (
            f"q_sample disagrees with the closed form at t={step}"
        )


def test_the_variance_preserving_identity_holds():
    schedule = _derived()
    unity = schedule.sqrt_alpha_bars**2 + schedule.sqrt_one_minus_alpha_bars**2
    assert torch.allclose(unity, torch.ones_like(unity))


# --------------------------------------------------------------------------- #
# Caching does not leak shared state
# --------------------------------------------------------------------------- #


def test_buffers_are_not_aliased_to_the_beta_cache():
    """`.to()` is a no-op when the dtype and device already match.

    Left uncloned, the cached ``betas`` would be the very object
    ``default_betas`` hands out, and one in-place edit anywhere would poison
    both caches for the rest of the process.
    """
    schedule = _derived("linear", DEFAULT_TIMESTEPS, dtype=torch.float32)
    assert schedule.betas is not default_betas("linear", DEFAULT_TIMESTEPS)

    before = default_betas().clone()
    schedule.betas.add_(1.0)  # deliberate abuse; must not reach the other cache
    assert torch.equal(default_betas(), before)
    schedule.betas.sub_(1.0)


def test_repeated_calls_hit_the_cache():
    assert _derived("linear", 50) is _derived("linear", 50)


# --------------------------------------------------------------------------- #
# Precision across dtypes
# --------------------------------------------------------------------------- #


def test_float64_callers_get_float64_accuracy():
    """Deriving in float32 and casting up would silently cap accuracy at 1e-7."""
    schedule = _derived(dtype=torch.float64)
    betas = make_beta_schedule("linear", DEFAULT_TIMESTEPS).to(torch.float64)
    exact = torch.cumprod(1.0 - betas, dim=0)
    assert (schedule.alpha_bars - exact).abs().max().item() < 1e-15


def test_cosine_survives_the_cast_to_float16():
    """The case the float32-then-cast ordering actually exists to protect.

    The cosine schedule's last ``alpha_bar`` is 2.4e-09, which flushes to zero
    in float16 — but its square root, 4.9e-05, is representable. Taking the
    roots before the cast is what keeps the mean coefficient alive at the end
    of the chain.
    """
    schedule = _derived("cosine", DEFAULT_TIMESTEPS, dtype=torch.float16)
    assert (schedule.sqrt_alpha_bars == 0).sum().item() == 0
    assert torch.isfinite(schedule.sqrt_alpha_bars).all()


# --------------------------------------------------------------------------- #
# Contracts are enforced loudly, not silently
# --------------------------------------------------------------------------- #


def test_the_schedule_stays_on_the_cpu_whatever_the_default_device_is():
    """`torch.set_default_device` must not relocate the schedule.

    Beyond the stated contract, MPS has no float64, so an unpinned cosine
    branch raises TypeError rather than building a schedule at all.
    """
    torch.set_default_device("cpu")
    try:
        assert make_beta_schedule("linear", 20).device.type == "cpu"
        assert make_beta_schedule("cosine", 20).device.type == "cpu"
    finally:
        torch.set_default_device(None)


@pytest.mark.parametrize("bad", [-1, -1000, DEFAULT_TIMESTEPS, DEFAULT_TIMESTEPS + 5])
def test_out_of_range_timesteps_are_rejected(x0, bad):
    """Negative indices wrap in torch: t=-1 would silently mean t=T-1."""
    with pytest.raises(IndexError):
        q_sample(x0, torch.full((x0.shape[0],), bad, dtype=torch.long), torch.randn_like(x0))


def test_integer_images_are_rejected():
    """Every coefficient lies in [0, 1) and truncates to zero in an int dtype."""
    x0 = torch.full((2, 1, 4, 4), 100, dtype=torch.int64)
    with pytest.raises(TypeError):
        q_sample(x0, torch.zeros(2, dtype=torch.long), torch.zeros_like(x0))


def test_misshaped_timestep_tensors_are_rejected(x0):
    with pytest.raises(ValueError):
        q_sample(x0, torch.zeros(2, 4, dtype=torch.long), torch.randn_like(x0))


@pytest.mark.parametrize("bad", [10.9, 2.5, "10", True])
def test_non_integral_schedule_lengths_are_rejected(bad):
    """`int()` would truncate 10.9 to a ten-step schedule with no signal."""
    with pytest.raises(ValueError):
        make_beta_schedule("linear", bad)
