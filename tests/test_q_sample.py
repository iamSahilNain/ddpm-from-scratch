"""The forward marginal, checked at both ends of the schedule and in between.

Every test here is behavioural: it noises real tensors and looks at what comes
out. None of them recompute the closed form, so passing them means the process
your implementation defines has the right endpoints and the right statistics,
not that it matches a particular arrangement of square roots.
"""

from __future__ import annotations

import pytest
import torch

from conftest import structured_batch
from diffusion import DEFAULT_TIMESTEPS, q_sample

LAST = DEFAULT_TIMESTEPS - 1


def full(batch_size: int, value: int) -> torch.Tensor:
    return torch.full((batch_size,), value, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Shapes and plumbing
# --------------------------------------------------------------------------- #


def test_output_matches_input_shape(x0):
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, full(x0.shape[0], 500), noise)
    assert x_t.shape == x0.shape
    assert x_t.dtype == x0.dtype
    assert x_t.device == x0.device
    assert torch.isfinite(x_t).all()


def test_handles_a_batch_of_mixed_timesteps(x0):
    """Each image in the batch is noised to its own t. This is the training case.

    The failure this catches is broadcasting the timestep coefficients against
    the wrong axis, which either raises loudly or — worse — silently applies one
    image's noise level to the whole batch.
    """
    batch = x0.shape[0]
    t = torch.tensor([0, 1, 7, 99, 250, 500, 900, LAST], dtype=torch.long)[:batch]
    noise = torch.randn_like(x0)

    mixed = q_sample(x0, t, noise)
    assert mixed.shape == x0.shape
    assert torch.isfinite(mixed).all()

    # Noising the batch together must equal noising each element on its own.
    for index in range(batch):
        alone = q_sample(
            x0[index : index + 1], t[index : index + 1], noise[index : index + 1]
        )
        assert torch.allclose(mixed[index], alone[0], atol=1e-5), (
            f"element {index} at t={t[index].item()} changed when batched"
        )


@pytest.mark.parametrize("shape", [(1, 1, 28, 28), (4, 1, 8, 8), (3, 3, 16, 16)])
def test_shape_agnostic(shape):
    x = torch.randn(shape)
    x_t = q_sample(x, full(shape[0], 123), torch.randn(shape))
    assert x_t.shape == torch.Size(shape)


def test_is_a_deterministic_function_of_the_supplied_noise(x0):
    """All the randomness lives in ``noise``; nothing is drawn internally."""
    noise = torch.randn_like(x0)
    t = full(x0.shape[0], 300)
    assert torch.equal(q_sample(x0, t, noise), q_sample(x0, t, noise))


def test_does_not_modify_its_inputs(x0):
    noise = torch.randn_like(x0)
    x0_before, noise_before = x0.clone(), noise.clone()
    q_sample(x0, full(x0.shape[0], 400), noise)
    assert torch.equal(x0, x0_before), "q_sample mutated x0 in place"
    assert torch.equal(noise, noise_before), "q_sample mutated noise in place"


# --------------------------------------------------------------------------- #
# The first step: almost nothing happens
# --------------------------------------------------------------------------- #


def test_first_timestep_barely_changes_the_image(x0):
    """At t=0 only the smallest beta has been applied, so x_t is nearly x_0.

    The tolerance is loose enough to accommodate that first sliver of noise and
    far tighter than any coefficient mix-up could survive — swapping the two
    scalings moves the output by order one, not order one hundredth.
    """
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, full(x0.shape[0], 0), noise)

    assert (x_t - x0).abs().max().item() < 0.1
    assert (x_t - x0).abs().mean().item() < 0.02


def test_first_timestep_is_not_exactly_the_image(x0):
    """Nearly x_0, but some noise really was added — t=0 is not a no-op."""
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, full(x0.shape[0], 0), noise)
    assert not torch.equal(x_t, x0)
    assert (x_t - x0).abs().mean().item() > 1e-4


# --------------------------------------------------------------------------- #
# The last step: the image is gone
# --------------------------------------------------------------------------- #


def test_last_timestep_is_approximately_standard_gaussian():
    """The property the whole reverse process is built on.

    Sampling starts by drawing standard Gaussian noise and calling it x_T. That
    is only legitimate if the forward process genuinely lands there, which means
    zero mean and unit standard deviation at the final timestep.
    """
    x0 = structured_batch(64, seed=7)
    noise = torch.randn_like(x0)
    x_t = q_sample(x0, full(64, LAST), noise)

    assert abs(x_t.mean().item()) < 0.05, "x_T is off-centre"
    assert abs(x_t.std().item() - 1.0) < 0.05, "x_T has the wrong scale"


def test_last_timestep_forgets_the_input():
    """Two completely different images, the same noise, indistinguishable output.

    A stronger statement than the moment check above: not merely that x_T has
    Gaussian statistics, but that essentially none of x_0 survives in it.
    """
    a = structured_batch(16, seed=1)
    b = -structured_batch(16, seed=2)
    noise = torch.randn_like(a)

    a_t = q_sample(a, full(16, LAST), noise)
    b_t = q_sample(b, full(16, LAST), noise)

    assert (a_t - b_t).abs().max().item() < 0.05
    assert (a - b).abs().max().item() > 0.5, "the two inputs were not different enough"


# --------------------------------------------------------------------------- #
# In between: a monotone dissolve, not a jump
# --------------------------------------------------------------------------- #


def test_signal_decays_monotonically_across_the_schedule():
    """Correlation with the clean image falls as t rises, at every t sampled.

    The diagnostic that catches a reversed schedule, an off-by-one in the
    indexing, or a cumulative product taken along the wrong axis — all of which
    leave the endpoints looking fine while the middle of the chain runs
    backwards.
    """
    x0 = structured_batch(64, seed=3)
    noise = torch.randn_like(x0)

    correlations = []
    for step in [0, 50, 100, 200, 400, 600, 800, LAST]:
        x_t = q_sample(x0, full(64, step), noise)
        flat_x0 = (x0 - x0.mean()).flatten()
        flat_xt = (x_t - x_t.mean()).flatten()
        correlations.append(
            (flat_x0 @ flat_xt / (flat_x0.norm() * flat_xt.norm())).item()
        )

    assert correlations[0] > 0.98, "the image is not intact at the first step"
    assert correlations[-1] < 0.1, "the image is still visible at the last step"
    for earlier, later in zip(correlations, correlations[1:]):
        assert later < earlier + 1e-3, f"signal increased with t: {correlations}"


def test_variance_stays_bounded_across_the_schedule():
    """No timestep blows up or collapses; the process is variance preserving.

    DDPM's forward process is constructed so the marginal stays at roughly unit
    scale throughout, rather than accumulating variance the way an unnormalised
    random walk would.
    """
    x0 = structured_batch(64, seed=5)
    noise = torch.randn_like(x0)

    for step in [0, 1, 10, 100, 500, 900, LAST]:
        std = q_sample(x0, full(64, step), noise).std().item()
        assert 0.3 < std < 1.6, f"std {std:.3f} at t={step} is out of range"
