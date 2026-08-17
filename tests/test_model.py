"""Tests for the scaffolding: the UNet and the timestep embedding.

These exist so that a red test run points at the diffusion code rather than
leaving open the question of whether the network underneath it is sound.
"""

from __future__ import annotations

import math

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


# --------------------------------------------------------------------------- #
# Attention. Added after a review found this block had no coverage at all:
# deleting its residual, dropping the output permute, and setting k = q or
# v = q all left the entire suite green.
# --------------------------------------------------------------------------- #


def test_attention_matches_an_independent_reference():
    """Recompute the block with einsum rather than reshape-and-permute.

    The six lines of index gymnastics in AttentionBlock.forward are the most
    error-prone in the file, and a wrong permute scrambles head, channel and
    spatial layout in a way that still returns the right shape and still
    trains to a plausible loss.
    """
    from model import AttentionBlock

    torch.manual_seed(0)
    block = AttentionBlock(channels=16).eval()
    x = torch.randn(2, 16, 5, 5)

    with torch.no_grad():
        actual = block(x)

        b, c, h, w = x.shape
        heads, head_dim = block.num_heads, 16 // block.num_heads
        qkv = block.qkv(block.norm(x))
        q, k, v = (
            part.reshape(b, heads, head_dim, h * w) for part in qkv.chunk(3, dim=1)
        )
        weights = torch.softmax(
            torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(head_dim), dim=-1
        )
        attended = torch.einsum("bhij,bhdj->bhdi", weights, v).reshape(b, c, h, w)
        expected = x + block.proj(attended)

    assert torch.allclose(actual, expected, atol=1e-5), (
        f"max deviation {(actual - expected).abs().max().item():.3e} from the "
        "einsum reference"
    )


def test_attention_is_equivariant_to_permuting_positions():
    """Attention is set-based: permute the pixels, permute the output.

    Catches a transposed or mis-ordered spatial axis, which no shape check can
    see because H and W are usually equal.
    """
    from model import AttentionBlock

    torch.manual_seed(1)
    block = AttentionBlock(channels=8).eval()
    x = torch.randn(1, 8, 4, 4)
    order = torch.randperm(16)

    with torch.no_grad():
        straight = block(x).flatten(2)[..., order]
        permuted = block(x.flatten(2)[..., order].reshape(1, 8, 4, 4)).flatten(2)

    assert torch.allclose(straight, permuted, atol=1e-5)


def test_attention_actually_mixes_distant_positions():
    """The point of the block: one corner must be able to affect another.

    A block that returned its input, or attended only to itself, would pass
    every shape and finiteness check in this file.
    """
    from model import AttentionBlock

    torch.manual_seed(2)
    block = AttentionBlock(channels=8).eval()
    x = torch.randn(1, 8, 6, 6)
    poked = x.clone()
    poked[0, :, 0, 0] += 5.0

    with torch.no_grad():
        delta = (block(poked) - block(x)).abs()

    assert delta[0, :, 3, 3].max().item() > 1e-3, (
        "poking one corner left a distant position untouched — no mixing"
    )
