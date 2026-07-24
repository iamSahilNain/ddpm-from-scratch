"""Tests for the scaffolding: the UNet and the timestep embedding.

Not part of the learning checkpoint. These exist so that a red test run points
at the diffusion code rather than leaving open the question of whether the
network underneath it is sound.
"""

from __future__ import annotations

import pytest
import torch

from model import UNet, count_parameters, timestep_embedding


# --------------------------------------------------------------------------- #
# Timestep embedding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 7])
def test_embedding_shape(dim):
    emb = timestep_embedding(torch.arange(5), dim)
    assert emb.shape == (5, dim)
    assert torch.isfinite(emb).all()


def test_embedding_is_bounded():
    emb = timestep_embedding(torch.arange(0, 1000), 64)
    assert emb.abs().max().item() <= 1.0 + 1e-6


def test_embedding_separates_timesteps():
    emb = timestep_embedding(torch.tensor([0, 1, 500, 999]), 64)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[0], emb[2])


def test_embedding_is_smooth_in_t():
    """Adjacent timesteps sit closer together than distant ones.

    This is the whole point of a sinusoidal encoding over a learned per-step
    table: the network gets to share what it learns across nearby noise levels.
    """
    emb = timestep_embedding(torch.tensor([100, 101, 900]), 64)
    assert (emb[0] - emb[1]).norm() < (emb[0] - emb[2]).norm()


# --------------------------------------------------------------------------- #
# UNet
# --------------------------------------------------------------------------- #


def test_output_shape_matches_input():
    model = UNet(base_channels=16, channel_mults=(1, 2, 2), num_res_blocks=1)
    x = torch.randn(3, 1, 28, 28)
    out = model(x, torch.randint(0, 1000, (3,)))
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "mults,size", [((1, 2), 8), ((1, 2), 28), ((1, 2, 2), 28), ((1, 2, 2), 32)]
)
def test_shapes_across_depths_and_sizes(mults, size):
    model = UNet(base_channels=8, channel_mults=mults, num_res_blocks=1)
    x = torch.randn(2, 1, size, size)
    assert model(x, torch.zeros(2, dtype=torch.long)).shape == x.shape


def test_rejects_indivisible_spatial_size():
    model = UNet(base_channels=8, channel_mults=(1, 2, 2), num_res_blocks=1)
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 30, 30), torch.zeros(1, dtype=torch.long))


def test_rejects_mismatched_batch_of_timesteps():
    model = UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    with pytest.raises(ValueError):
        model(torch.randn(4, 1, 8, 8), torch.zeros(3, dtype=torch.long))


def test_accepts_a_scalar_timestep():
    """Convenience for sampling loops, where every image shares one t."""
    model = UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    x = torch.randn(2, 1, 8, 8)
    assert model(x, torch.tensor(7)).shape == x.shape
    assert model(x, 7).shape == x.shape


def test_the_timestep_actually_changes_the_output():
    """Conditioning is wired up, not decorative."""
    model = UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1, dropout=0.0)
    model.eval()
    x = torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        early = model(x, torch.zeros(2, dtype=torch.long))
        late = model(x, torch.full((2,), 999, dtype=torch.long))
    assert not torch.allclose(early, late, atol=1e-4)


def test_each_image_is_conditioned_on_its_own_timestep():
    """No cross-talk: element i of the batch depends only on t[i]."""
    model = UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1, dropout=0.0)
    model.eval()
    x = torch.randn(2, 1, 8, 8)
    with torch.no_grad():
        batched = model(x, torch.tensor([0, 999]))
        alone = torch.cat(
            [model(x[:1], torch.tensor([0])), model(x[1:], torch.tensor([999]))]
        )
    assert torch.allclose(batched, alone, atol=1e-5)


def test_gradients_flow_through_the_whole_network():
    model = UNet(base_channels=8, channel_mults=(1, 2), num_res_blocks=1)
    out = model(torch.randn(2, 1, 8, 8), torch.randint(0, 1000, (2,)))
    out.pow(2).mean().backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_default_model_is_small_enough_to_train_locally():
    model = UNet()
    assert count_parameters(model) < 20_000_000
