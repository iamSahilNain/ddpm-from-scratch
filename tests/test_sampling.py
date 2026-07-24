"""The reverse process, checked structurally and then against a known answer.

Most of this file runs an untrained network, so it says nothing about how the
samples look — only that the chain runs to completion, honours the requested
shape, stays finite over a thousand steps, and is stochastic. Note in
particular that there is no assertion here about the *magnitude* of what comes
out of an untrained model: the reverse process is only contractive when the
network genuinely predicts the noise, so a correct implementation driven by
random weights will happily drift to values in the hundreds. That is not a bug,
and a test that forbade it would fail on correct code.

The last test is the one with teeth. It replaces the network with a denoiser
whose optimal behaviour is known in closed form, which pins down the whole
recursion — coefficients, indexing, and the variance term — against an answer
computed independently of it. See :class:`GaussianOracle`.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from diffusion import DEFAULT_TIMESTEPS, make_beta_schedule, p_sample_loop

pytestmark = pytest.mark.slow


def test_returns_the_requested_shape(tiny_model):
    shape = (2, 1, 8, 8)
    out = p_sample_loop(tiny_model, shape)
    assert out.shape == torch.Size(shape)


@pytest.mark.parametrize("shape", [(1, 1, 8, 8), (3, 1, 8, 8)])
def test_honours_the_batch_dimension(tiny_model, shape):
    assert p_sample_loop(tiny_model, shape).shape == torch.Size(shape)


def test_output_is_finite(tiny_model):
    """A thousand steps is long enough for any instability to be obvious."""
    out = p_sample_loop(tiny_model, (2, 1, 8, 8))
    assert torch.isfinite(out).all(), "the chain diverged"


def test_is_stochastic(tiny_model):
    """Two calls give two different images: the chain starts from fresh noise."""
    torch.manual_seed(11)
    first = p_sample_loop(tiny_model, (2, 1, 8, 8))
    torch.manual_seed(22)
    second = p_sample_loop(tiny_model, (2, 1, 8, 8))
    assert not torch.allclose(first, second)


def test_is_reproducible_under_a_fixed_seed(tiny_model):
    torch.manual_seed(99)
    first = p_sample_loop(tiny_model, (2, 1, 8, 8))
    torch.manual_seed(99)
    second = p_sample_loop(tiny_model, (2, 1, 8, 8))
    assert torch.allclose(first, second), "sampling used entropy outside the global seed"


def test_does_not_return_a_graph(tiny_model):
    """Sampling is inference. Holding a thousand steps of graph would not fit."""
    out = p_sample_loop(tiny_model, (2, 1, 8, 8))
    assert not out.requires_grad
    assert out.grad_fn is None


def test_respects_an_explicit_device(tiny_model):
    out = p_sample_loop(tiny_model, (2, 1, 8, 8), device=torch.device("cpu"))
    assert out.device.type == "cpu"


def test_leaves_model_parameters_untouched(tiny_model):
    """Sampling reads the weights, it does not update them."""
    before = {name: p.detach().clone() for name, p in tiny_model.named_parameters()}
    p_sample_loop(tiny_model, (2, 1, 8, 8))
    for name, p in tiny_model.named_parameters():
        assert torch.equal(p.detach(), before[name]), f"{name} changed during sampling"


# --------------------------------------------------------------------------- #
# The correctness test: a reverse process with a known answer
# --------------------------------------------------------------------------- #


class GaussianOracle(nn.Module):
    """The exactly optimal noise predictor for one specific data distribution.

    Take the "data" to be standard Gaussian noise. Then the forward process
    maps a standard Gaussian to a standard Gaussian at every timestep, and the
    best possible estimate of the noise given a noised image works out to that
    image scaled by a single number that depends only on t — no network needed,
    no training, and derivable from the schedule alone.

    Feed that into the reverse process and the answer is known: if the recursion
    is right, sampling with this oracle must return standard Gaussian samples,
    because the reverse of a process that maps this distribution to itself maps
    it back to itself. So the test below has a ground truth that was computed
    without reference to how the recursion is written.

    That makes it the sharpest check in the suite. Every structural test above
    passes for a chain with a sign error, a swapped pair of alpha terms, a
    posterior variance off by a factor, or an off-by-one in the loop bounds; all
    of those move the output distribution away from standard Gaussian, most of
    them by orders of magnitude.
    """

    def __init__(self, timesteps: int = DEFAULT_TIMESTEPS):
        super().__init__()
        # Present so that `next(model.parameters())` works for device inference.
        self.unused = nn.Parameter(torch.zeros(1))
        alpha_bars = torch.cumprod(1.0 - make_beta_schedule("linear", timesteps), dim=0)
        self.register_buffer("coefficient", (1.0 - alpha_bars).sqrt())

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        return self.coefficient[t].reshape(-1, 1, 1, 1) * x


def test_reverse_process_inverts_the_forward_process():
    """With the optimal denoiser, sampling must return standard Gaussians.

    Both of the fixed variance choices Ho et al. describe satisfy this, so the
    test does not force a preference between them. Nothing else does: the
    tolerances here are a few percent, and a recursion that is wrong in any of
    the usual ways overshoots them by a factor of a hundred or more.
    """
    torch.manual_seed(4)
    oracle = GaussianOracle().eval()

    out = p_sample_loop(oracle, (64, 1, 8, 8), device=torch.device("cpu"))

    assert torch.isfinite(out).all()
    assert abs(out.mean().item()) < 0.10, (
        f"sample mean {out.mean().item():+.3f} — the reverse chain is off-centre"
    )
    assert abs(out.std().item() - 1.0) < 0.15, (
        f"sample std {out.std().item():.3f} — the reverse chain does not "
        "restore the starting distribution"
    )
