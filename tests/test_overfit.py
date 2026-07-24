"""End-to-end learning check: can the objective actually drive a model down.

Everything else in the suite tests one function against its contract. This tests
that the pieces compose into something trainable, which is the only property
that catches a bug that is individually consistent everywhere and wrong overall
— a target that the model cannot in principle predict, the timestep conditioning
being dropped on the floor, a sign flip that makes the loss minimised by the
wrong weights.

What it does *not* do is expect the loss to reach zero. ``p_losses`` draws fresh
noise every call, so even a model that has perfectly memorised these eight
images faces a new target each step. A substantial, sustained drop is the signal
here; convergence is not.
"""

from __future__ import annotations

import pytest
import torch

from conftest import structured_batch
from diffusion import DEFAULT_TIMESTEPS, p_losses
from model import UNet

pytestmark = pytest.mark.slow

STEPS = 300
WINDOW = 25

#: The loss an untrained model scores. It predicts roughly nothing, and the
#: target is standard Gaussian noise, so the mean squared error is about one.
UNTRAINED = 1.0


def run_optimisation(steps: int = STEPS, seed: int = 0) -> list[float]:
    """Optimise one fixed batch of eight images and return the loss per step."""
    torch.manual_seed(seed)
    model = UNet(
        in_channels=1,
        base_channels=16,
        channel_mults=(1, 2),
        num_res_blocks=1,
        attention_levels=(1,),
        dropout=0.0,
    )
    model.train()
    x0 = structured_batch(8, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    history = []
    for _ in range(steps):
        t = torch.randint(0, DEFAULT_TIMESTEPS, (8,), dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = p_losses(model, x0, t)
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history


@pytest.fixture(scope="module")
def history() -> list[float]:
    return run_optimisation()


def test_the_loss_stays_finite_throughout(history):
    assert all(value == value and abs(value) != float("inf") for value in history)
    assert all(value >= 0.0 for value in history)


def test_the_very_first_step_scores_about_one(history):
    """Before any weight update, the loss sits at the untrained level.

    Note this is the first step alone, not a window: with this learning rate the
    model has already improved substantially by step twenty-five, so a windowed
    average is not a measurement of the untrained loss.
    """
    assert 0.4 < history[0] < 6.0, f"initial loss {history[0]:.3f} is not order one"


def test_the_loss_comes_down_substantially(history):
    """The actual assertion: the objective is optimisable.

    A windowed mean at the end, because the per-step loss depends on which
    timesteps were drawn and is far too noisy to read point to point. Both an
    absolute and a relative bound: the absolute one is what catches a model that
    starts low by accident, the relative one catches a model that improves but
    only trivially.
    """
    start = sum(history[:WINDOW]) / WINDOW
    end = sum(history[-WINDOW:]) / WINDOW

    assert end < 0.25 * UNTRAINED, (
        f"loss ended at {end:.3f} after {STEPS} steps — a model that is not "
        f"learning stays near {UNTRAINED:.1f}"
    )
    assert end < 0.5 * start, (
        f"loss went from {start:.3f} to {end:.3f} in {STEPS} steps — "
        "the model is not learning from this objective"
    )


def test_the_improvement_is_sustained_not_a_dip(history):
    """The last quarter is better than the first quarter, on average.

    Guards against a run that happens to end on a lucky batch of easy timesteps.
    """
    quarter = len(history) // 4
    first = sum(history[:quarter]) / quarter
    last = sum(history[-quarter:]) / quarter
    assert last < first, f"first quarter {first:.3f}, last quarter {last:.3f}"


def test_learning_is_not_seed_dependent():
    """Repeat the whole thing on another seed, so one lucky init cannot pass it."""
    history = run_optimisation(steps=200, seed=7)
    end = sum(history[-WINDOW:]) / WINDOW
    assert end < 0.3 * UNTRAINED, f"seed 7 ended at {end:.3f}"
