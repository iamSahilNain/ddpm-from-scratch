"""Properties the beta schedule has to satisfy for anything downstream to work.

A subtlety about this file: ``make_beta_schedule`` returns betas, but the
properties that actually matter are properties of the *cumulative* products
derived from them. With only the betas to hold on to, the tests have to
re-derive those here. :func:`alpha_bars_from` therefore restates the definition
already given in prose in ``diffusion.make_beta_schedule``'s docstring — it is
not extra information, but if you would rather work with no derivation in front
of you at all, write your schedule first and read this file second.
"""

from __future__ import annotations

import pytest
import torch

from diffusion import DEFAULT_TIMESTEPS, make_beta_schedule


def alpha_bars_from(betas: torch.Tensor) -> torch.Tensor:
    """Re-derive the cumulative signal-retention sequence from a beta schedule."""
    return torch.cumprod(1.0 - betas, dim=0)


# --------------------------------------------------------------------------- #
# Basic shape and range
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("timesteps", [10, 100, DEFAULT_TIMESTEPS])
def test_returns_one_beta_per_timestep(timesteps):
    betas = make_beta_schedule("linear", timesteps)
    assert isinstance(betas, torch.Tensor)
    assert betas.shape == (timesteps,)
    assert betas.dtype == torch.float32


@pytest.mark.parametrize("timesteps", [10, 100, DEFAULT_TIMESTEPS])
def test_betas_are_valid_variances(timesteps):
    """Every beta has to sit strictly inside the open unit interval.

    At zero the step does nothing; at one the step destroys the signal outright
    and the cumulative products collapse to zero for every later timestep.
    """
    betas = make_beta_schedule("linear", timesteps)
    assert torch.all(betas > 0.0), "a beta of zero or below is not a variance"
    assert torch.all(betas < 1.0), "a beta at or above one destroys the chain"
    assert torch.isfinite(betas).all()


def test_linear_schedule_is_increasing():
    """Noise is added gently at first and aggressively later, never the reverse."""
    betas = make_beta_schedule("linear", DEFAULT_TIMESTEPS)
    assert torch.all(betas[1:] > betas[:-1])


def test_linear_schedule_hits_the_documented_endpoints():
    """The endpoints from Ho et al. — one ten-thousandth up to two hundredths."""
    betas = make_beta_schedule("linear", DEFAULT_TIMESTEPS)
    assert betas[0].item() == pytest.approx(1e-4, rel=1e-3)
    assert betas[-1].item() == pytest.approx(0.02, rel=1e-3)


def test_linear_schedule_is_evenly_spaced():
    betas = make_beta_schedule("linear", DEFAULT_TIMESTEPS)
    gaps = betas[1:] - betas[:-1]
    assert torch.allclose(gaps, gaps[0].expand_as(gaps), atol=1e-7)


# --------------------------------------------------------------------------- #
# The properties the rest of the model depends on
# --------------------------------------------------------------------------- #


def test_alpha_bar_is_monotonically_decreasing():
    """Signal can only ever be removed as t grows, never added back."""
    alpha_bars = alpha_bars_from(make_beta_schedule("linear", DEFAULT_TIMESTEPS))
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])
    assert torch.all(alpha_bars > 0.0)
    assert torch.all(alpha_bars <= 1.0)


def test_alpha_bar_starts_near_one():
    """The first step barely touches the image."""
    alpha_bars = alpha_bars_from(make_beta_schedule("linear", DEFAULT_TIMESTEPS))
    assert alpha_bars[0].item() > 0.999


def test_alpha_bar_ends_near_zero():
    """The licence for sampling to start from pure noise.

    If this fails, the reverse process begins from a distribution that is not
    the one the forward process actually ends at, and samples will be subtly
    wrong in a way no amount of extra training fixes.
    """
    alpha_bars = alpha_bars_from(make_beta_schedule("linear", DEFAULT_TIMESTEPS))
    assert alpha_bars[-1].item() < 1e-3


def test_alpha_bar_decays_smoothly():
    """No cliff: the largest single-step drop stays small.

    Catches schedules that are technically decreasing but concentrate all the
    destruction into a handful of steps, which the reverse process cannot undo.
    """
    alpha_bars = alpha_bars_from(make_beta_schedule("linear", DEFAULT_TIMESTEPS))
    assert (alpha_bars[:-1] - alpha_bars[1:]).max().item() < 0.02


# --------------------------------------------------------------------------- #
# Other schedule kinds are optional, but must not silently misbehave
# --------------------------------------------------------------------------- #


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        make_beta_schedule("definitely-not-a-schedule", 50)


@pytest.mark.parametrize("kind", ["cosine", "quadratic", "sigmoid"])
def test_optional_kinds_if_present_are_well_formed(kind):
    """If you implement an extra schedule, it obeys the same invariants."""
    try:
        betas = make_beta_schedule(kind, DEFAULT_TIMESTEPS)
    except ValueError:
        pytest.skip(f"{kind!r} schedule not implemented")

    assert betas.shape == (DEFAULT_TIMESTEPS,)
    assert torch.all(betas > 0.0) and torch.all(betas < 1.0)

    alpha_bars = alpha_bars_from(betas)
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])
    assert alpha_bars[-1].item() < 1e-2
