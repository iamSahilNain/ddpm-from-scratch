"""The training objective: is it a scalar, is it finite, does it train anything.

``p_losses`` draws its own noise, so the loss is stochastic even for fixed
inputs. Everything here is written to be robust to that.
"""

from __future__ import annotations

import pytest
import torch

from diffusion import DEFAULT_TIMESTEPS, p_losses


def random_t(batch_size: int) -> torch.Tensor:
    return torch.randint(0, DEFAULT_TIMESTEPS, (batch_size,), dtype=torch.long)


# --------------------------------------------------------------------------- #
# Shape and value
# --------------------------------------------------------------------------- #


def test_returns_a_finite_scalar(tiny_model, x0):
    loss = p_losses(tiny_model, x0, random_t(x0.shape[0]))

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0, f"expected a scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), "loss is nan or inf"
    assert loss.item() >= 0.0, "a squared error cannot be negative"


def test_is_reduced_over_the_whole_batch(tiny_model, x0):
    """A mean, not a sum: doubling the batch must not roughly double the loss.

    Catches ``reduction="sum"`` and reductions that leave the spatial axes in,
    both of which train fine but silently couple the effective learning rate to
    the batch size and the image resolution.
    """
    torch.manual_seed(0)
    small = float(p_losses(tiny_model, x0[:2], random_t(2)))
    torch.manual_seed(0)
    large = float(p_losses(tiny_model, x0.repeat(4, 1, 1, 1), random_t(32)))

    assert large < 4.0 * small, f"loss scales with batch size ({small:.3f} -> {large:.3f})"


def test_magnitude_is_plausible_at_initialisation(tiny_model, x0):
    """An untrained network predicts roughly nothing, so the loss is order one.

    The target is standard Gaussian noise, whose mean square is one. A loss far
    below that at step zero means the target is not the noise; a loss orders of
    magnitude above it means the reduction or the normalisation is wrong.
    """
    losses = [float(p_losses(tiny_model, x0, random_t(x0.shape[0]))) for _ in range(8)]
    average = sum(losses) / len(losses)
    assert 0.2 < average < 10.0, f"initial loss {average:.3f} is not order one"


@pytest.mark.parametrize("timestep", [0, 1, 500, DEFAULT_TIMESTEPS - 1])
def test_finite_at_every_extreme_of_the_schedule(tiny_model, x0, timestep):
    """Both ends of the chain, where the coefficients are most lopsided."""
    t = torch.full((x0.shape[0],), timestep, dtype=torch.long)
    assert torch.isfinite(p_losses(tiny_model, x0, t))


def test_accepts_a_batch_of_mixed_timesteps(tiny_model, x0):
    t = torch.tensor([0, 3, 60, 199, 400, 700, 950, DEFAULT_TIMESTEPS - 1], dtype=torch.long)
    loss = p_losses(tiny_model, x0, t[: x0.shape[0]])
    assert loss.ndim == 0 and torch.isfinite(loss)


# --------------------------------------------------------------------------- #
# Autograd
# --------------------------------------------------------------------------- #


def test_gradients_reach_every_parameter(small_model, x0):
    """The loss has to be differentiably connected to all of the weights.

    The usual way this breaks is a stray ``detach``, a ``no_grad`` block around
    the model call, or building x_t from a tensor that was never on the graph.
    Those give a finite, plausible-looking loss and a model that never moves.
    """
    small_model.zero_grad(set_to_none=True)
    loss = p_losses(small_model, x0, random_t(x0.shape[0]))
    loss.backward()

    missing = [name for name, p in small_model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing[:5]}"

    total = sum(float(p.grad.pow(2).sum()) for p in small_model.parameters())
    assert total > 0.0, "every gradient is exactly zero"
    assert all(torch.isfinite(p.grad).all() for p in small_model.parameters())


def test_gradients_reach_the_timestep_embedding(small_model, x0):
    """Specifically the conditioning path — evidence that t is actually used.

    If t never reaches the network, the timestep MLP sits outside the graph and
    these parameters get zero gradient while everything else trains. The model
    then learns a single average denoiser for all noise levels, which produces
    grey mush at sampling time and an otherwise unremarkable loss curve.
    """
    small_model.zero_grad(set_to_none=True)
    p_losses(small_model, x0, random_t(x0.shape[0])).backward()

    time_grad = sum(
        float(p.grad.pow(2).sum())
        for name, p in small_model.named_parameters()
        if name.startswith("time_mlp") and p.grad is not None
    )
    assert time_grad > 0.0, "the timestep embedding received no gradient"


def test_a_single_optimiser_step_changes_the_weights(small_model, x0):
    before = {name: p.detach().clone() for name, p in small_model.named_parameters()}
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)

    optimizer.zero_grad(set_to_none=True)
    p_losses(small_model, x0, random_t(x0.shape[0])).backward()
    optimizer.step()

    changed = [
        name
        for name, p in small_model.named_parameters()
        if not torch.equal(p.detach(), before[name])
    ]
    assert changed, "no parameter moved after an optimiser step"
