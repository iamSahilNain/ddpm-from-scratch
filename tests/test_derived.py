"""The schedule cache behind the diffusion process, and the contracts it enforces.

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
from test_sampling import GaussianOracle
from diffusion import (
    DEFAULT_TIMESTEPS,
    _derived,
    default_betas,
    make_beta_schedule,
    p_losses,
    p_sample_loop,
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


# --------------------------------------------------------------------------- #
# The posterior buffers the reverse process reads
# --------------------------------------------------------------------------- #


def test_posterior_buffers_match_their_closed_forms():
    schedule = _derived(dtype=torch.float64)
    alpha_bars = schedule.alpha_bars

    expected_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bars[:-1]])
    assert torch.allclose(schedule.alpha_bars_prev, expected_prev)

    expected_variance = schedule.betas * (1.0 - expected_prev) / (1.0 - alpha_bars)
    assert torch.allclose(schedule.posterior_variance, expected_variance)


def test_the_posterior_variance_vanishes_at_the_first_step():
    """``alpha_bars_prev[0]`` is one, so the algebra already says "add no noise".

    The loop still branches on it rather than relying on this, but if the buffer
    were built with a roll instead of a prepended one, this would come out as
    ``alpha_bars[-1]`` and the last step would get a full beta of grain.
    """
    assert _derived(dtype=torch.float64).posterior_variance[0].item() == 0.0


# --------------------------------------------------------------------------- #
# The objective and the reverse recursion, pinned
# --------------------------------------------------------------------------- #


def test_p_losses_scores_the_noise_not_the_image(tiny_model, x0):
    """Recompute the objective with the same seeded noise draw.

    The behavioural file cannot see this: a loss scored against ``x0``, or one
    reduced with ``sum``, or a model called without ``t``, all still produce a
    finite order-one scalar with gradients flowing.
    """
    t = torch.full((x0.shape[0],), 300, dtype=torch.long)

    torch.manual_seed(5)
    actual = p_losses(tiny_model, x0, t)

    torch.manual_seed(5)
    noise = torch.randn_like(x0)
    expected = torch.nn.functional.mse_loss(tiny_model(q_sample(x0, t, noise), t), noise)

    assert torch.allclose(actual, expected), f"{actual.item()} != {expected.item()}"


@pytest.mark.slow
def test_the_reverse_step_matches_the_unfolded_posterior():
    """Cross-check the folded recursion against Ho et al. eq. 7 written out.

    ``p_sample_loop`` collapses "invert the forward marginal, then take the
    posterior mean" into one expression. This runs the same chain the long way —
    implied ``x_0`` first, then the posterior mean as an explicit combination of
    ``x_0`` and ``x_t`` — so a folding error shows up as a mismatch rather than
    being duplicated into the reference.

    It also pins the no-noise carve-out at the last step, which is otherwise
    nearly invisible: beta_0 is 1e-4, so wrongly adding that noise moves the
    output std by 5e-05 and no statistical test in the suite would notice.

    Both chains consume the global RNG in the same order, which is what makes
    them comparable at all. That is a deliberate coupling — if the draw order in
    ``p_sample_loop`` ever changes, this test is supposed to break and be
    re-read, not silently updated.

    Run against the oracle rather than an untrained UNet on purpose. An
    untrained network drifts into the hundreds over a thousand steps, and the
    float32 rounding difference between the two algebraic routes drifts with it
    — to about 1e-02, which would swallow the 1e-02 of spurious grain. The
    oracle keeps the chain at unit scale, where the two routes agree to 1e-06
    and the carve-out bug stands out by four orders of magnitude.
    """
    model = GaussianOracle().eval()
    shape = (2, 1, 8, 8)
    schedule = _derived(dtype=torch.float32)

    torch.manual_seed(31)
    actual = p_sample_loop(model, shape, device=torch.device("cpu"))

    torch.manual_seed(31)
    x = torch.randn(shape)
    for step in reversed(range(DEFAULT_TIMESTEPS)):
        t = torch.full((shape[0],), step, dtype=torch.long)
        with torch.no_grad():
            predicted_noise = model(x, t)

        alpha_bar = schedule.alpha_bars[step]
        alpha_bar_prev = schedule.alpha_bars_prev[step]
        beta = schedule.betas[step]

        implied_x0 = (x - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt()
        mean = (
            alpha_bar_prev.sqrt() * beta / (1.0 - alpha_bar) * implied_x0
            + schedule.alphas[step].sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * x
        )

        if step > 0:
            x = mean + schedule.posterior_variance[step].sqrt() * torch.randn_like(x)
        else:
            x = mean

    assert torch.allclose(actual, x, atol=1e-4), (
        f"max deviation {(actual - x).abs().max().item():.3e} between the folded "
        "recursion and the unfolded posterior — the carve-out bug shows up here "
        "at around 1e-02"
    )


def test_sampling_refuses_a_device_the_model_is_not_on(tiny_model):
    """Otherwise this surfaces as a RuntimeError from inside the first forward."""
    with pytest.raises(ValueError):
        p_sample_loop(tiny_model, (1, 1, 8, 8), device=torch.device("meta"))
