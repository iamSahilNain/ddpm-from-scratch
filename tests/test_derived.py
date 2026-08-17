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
    """An in-place edit of a derived buffer must not reach ``_BETA_CACHE``.

    Honest note on what this does and does not pin. It cannot fail while the
    derivation casts to float64, because that cast copies unconditionally — so
    deleting the `.clone()` in `_derived` leaves this green. It is kept as a
    guard on the *property*, which would stop holding if anyone made the
    derivation float32 again, not as evidence that the clone earns its place.

    What genuinely is not protected is documented in
    :func:`test_derived_buffers_are_shared_and_mutable` below.

    Restoration here must be `copy_`, never `add_(1.0)` then `sub_(1.0)`. That
    round-trip does not restore a float32 tensor — 1e-04 comes back as
    1.0001659e-04, perturbing 993 of 1000 betas — and because the buffer is
    shared process-wide, it silently poisons the schedule for every test that
    runs afterwards, including the reverse-step pin that is the only guard on
    the recursion's coefficients.
    """
    schedule = _derived("linear", DEFAULT_TIMESTEPS, dtype=torch.float32)
    assert schedule.betas is not default_betas("linear", DEFAULT_TIMESTEPS)

    before = default_betas().clone()
    original = schedule.betas.clone()
    try:
        schedule.betas.add_(1.0)  # deliberate abuse; must not reach the other cache
        assert torch.equal(default_betas(), before)
    finally:
        schedule.betas.copy_(original)
    assert torch.equal(schedule.betas, original)


def test_derived_buffers_are_shared_and_mutable():
    """Characterisation, not endorsement: the cache hands out live references.

    Every caller receives the same tensors, so one in-place write corrupts the
    schedule for the whole process and can leave a ``_Derived`` internally
    inconsistent — ``alphas`` no longer equal to ``1 - betas`` — with nothing
    raised. ``q_sample`` would keep returning plausible finite values while
    ``p_sample_loop`` divided by a corrupted ``sqrt(alphas)``.

    Torch has no read-only tensor, so this is a documented contract rather than
    an enforced one. The test exists so the contract is visible and so that
    anyone who later adds real protection has something to update deliberately.
    """
    first = _derived("linear", 64, dtype=torch.float32)
    assert first is _derived("linear", 64, dtype=torch.float32)

    original = first.betas.clone()
    try:
        first.betas.mul_(0.0)
        again = _derived("linear", 64, dtype=torch.float32)
        assert again.betas.abs().max().item() == 0.0, "expected shared mutable state"
        assert not torch.allclose(again.alphas, 1.0 - again.betas), (
            "expected the tuple to be internally inconsistent after the write"
        )
    finally:
        first.betas.copy_(original)


def test_default_betas_validates_before_consulting_its_cache():
    """A warm cache must not let a bad argument through.

    ``int(timesteps)`` turns True into 1 and 1000.7 into 1000, so a key can be
    built for an argument that ``make_beta_schedule`` would reject — and on a
    warm cache that check is never reached.
    """
    default_betas("linear", 1)
    default_betas("linear", DEFAULT_TIMESTEPS)

    for bad in (True, 1000.7, 2.5):
        with pytest.raises(ValueError):
            default_betas("linear", bad)


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

    The timesteps here must be **distinct per sample**, and that is the whole
    point of the vector below. With a constant ``t`` this test is blind to
    every way of corrupting the timestep that preserves its value: permuting
    ``t`` across the batch, replacing it with a constant, or clamping the
    schedule so an untrained range is never sampled. All of those pass the
    entire rest of the suite too — `test_gradients_reach_the_timestep_embedding`
    does not catch them, because destroying the timestep makes the embedding's
    gradient *larger*, not zero.
    """
    t = torch.tensor([0, 7, 137, 300, 512, 743, 900, 999], dtype=torch.long)[: x0.shape[0]]

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
    float32 rounding difference between the two algebraic routes drifts with
    it — to about 1e-02, which would swallow the 1e-02 of spurious grain the
    carve-out bug produces.

    Run in **float64**, also on purpose, and this is what gives the test its
    teeth. At float32 the two routes agree only to 6.7e-05 against a 1e-04
    tolerance — correct code clears by 1.15x, which is not a margin, it is a
    coin flip waiting for a slightly larger batch. In float64 they agree to
    5.8e-13, so the tolerance below sits three orders above the noise floor
    and five or more below the smallest real bug. ``p_sample_loop`` takes its
    working dtype from the model's parameters, which is what makes this
    possible.
    """
    dtype = torch.float64
    model = GaussianOracle().eval().to(dtype)
    shape = (2, 1, 8, 8)
    schedule = _derived(dtype=dtype)

    torch.manual_seed(31)
    actual = p_sample_loop(model, shape, device=torch.device("cpu"))
    assert actual.dtype == dtype, "the chain did not adopt the model's dtype"

    torch.manual_seed(31)
    x = torch.randn(shape, dtype=dtype)
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

    assert torch.allclose(actual, x, atol=1e-9), (
        f"max deviation {(actual - x).abs().max().item():.3e} between the folded "
        "recursion and the unfolded posterior; the float64 noise floor is 6e-13 "
        "and the smallest real bug seen here is 1e-04"
    )


def test_sampling_refuses_a_device_the_model_is_not_on(tiny_model):
    """Otherwise this surfaces as a RuntimeError from inside the first forward."""
    with pytest.raises(ValueError):
        p_sample_loop(tiny_model, (1, 1, 8, 8), device=torch.device("meta"))


def test_sampling_accepts_an_unindexed_form_of_the_models_own_device(tiny_model):
    """`torch.device("cpu") != torch.device("cpu", 0)`, and .to() reports the latter.

    The regression this guards is a device check that compares the two forms
    naively. `train.pick_device` returns the unindexed form while model
    parameters report the indexed one, so such a check rejects the one device
    the caller definitely got right, and takes the entire GPU path down.
    """
    parameter_device = next(tiny_model.parameters()).device
    for form in {parameter_device, torch.device(parameter_device.type)}:
        out = p_sample_loop(tiny_model, (1, 1, 8, 8), device=form)
        assert out.shape == (1, 1, 8, 8)


class _PointMassOracle(torch.nn.Module):
    """The optimal denoiser when the data is a single point, ``x0 = value``.

    For a point mass the posterior mean of ``x_0`` is that point regardless of
    ``x_t``, so the optimal noise estimate is whatever noise would have carried
    ``value`` to the observed ``x_t``: ``(x_t - sqrt(abar) * value) / sqrt(1 - abar)``.
    """

    def __init__(self, value: float = 0.5):
        super().__init__()
        self.unused = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        schedule = _derived(dtype=torch.float64)
        self.register_buffer("sqrt_ab", schedule.sqrt_alpha_bars.float())
        self.register_buffer("sqrt_1mab", schedule.sqrt_one_minus_alpha_bars.float())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        index = t.reshape(-1, 1, 1, 1)
        return (x - self.sqrt_ab[index] * self.value) / self.sqrt_1mab[index]


@pytest.mark.slow
def test_sampling_collapses_onto_a_point_mass():
    """A known-answer test the N(0, I) oracle cannot provide.

    ``test_sampling.py``'s oracle uses standard Gaussian data, which is the
    worst possible probe for this recursion: Ho et al. note that the *other*
    variance choice is optimal for exactly that distribution, so the check is
    graded against the branch this code does not take. Measured, it fails to
    catch any of the classic alpha-swap or off-by-one errors — every one of
    them lands within its tolerance.

    A point mass is the opposite extreme: the chain has to contract a
    unit-variance start down to essentially nothing and land on the point.

    Be precise about what this does and does not cover, because it is easy to
    overrate. Measured against eight off-by-one mutants it catches three. It
    misses ``betas[step-1]``, ``posterior_variance[step-1]``,
    ``alphas[step-1]`` and ``sqrt_1mab[step-1]``, all of which still return
    mean exactly 0.5 and std exactly 0. The reason is structural: at ``t = 0``
    the coefficient ``beta_0 / (1 - alpha_bar_0)`` is exactly 1, so the last
    step projects onto the point regardless of what happened upstream. This is
    a *final-step and contraction* test, not a whole-recursion test.

    The recursion itself is guarded by
    :func:`test_the_reverse_step_matches_the_unfolded_posterior`, which catches
    all four of those at deviations from 2.2e-02 to 4.3e-02.
    """
    torch.manual_seed(17)
    out = p_sample_loop(_PointMassOracle(0.5).eval(), (16, 1, 8, 8), device=torch.device("cpu"))

    assert torch.isfinite(out).all()
    assert abs(out.mean().item() - 0.5) < 1e-3, (
        f"chain settled at {out.mean().item():.6f}, not on the point mass at 0.5"
    )
    assert out.std().item() < 1e-4, (
        f"residual std {out.std().item():.3e} — the chain is not contracting"
    )
