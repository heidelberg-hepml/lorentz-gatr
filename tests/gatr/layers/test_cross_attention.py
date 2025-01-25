import pytest
import torch
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from gatr.layers import CrossAttentionConfig, CrossAttention
from gatr.utils.clifford import SlowRandomPinTransform


@pytest.mark.parametrize("block_attention", [True, False])
def test_cross_attention(block_attention):
    """Test cross attention shapes and equivariance."""

    if block_attention:
        attn_mask = BlockDiagonalMask.from_seqlens([31, 29, 40], [3, 7, 21])
        attn_mask = attn_mask.materialize(shape=(100, 31))
        num_batch = 1
        num_kv = 31
        num_q = 100
    else:
        attn_mask = None
        num_batch = 2
        num_kv = 10
        num_q = 7

    config = CrossAttentionConfig(
        in_kv_mv_channels=5,
        out_mv_channels=6,
        in_kv_s_channels=2,
        out_s_channels=4,
        in_q_mv_channels=6,
        in_q_s_channels=6,
        num_heads=5,
        increase_hidden_channels=3,
    )
    layer = CrossAttention(config)

    mv_in = torch.randn(num_batch, num_kv, 5, 16)
    s_in = torch.randn(num_batch, num_kv, 2)

    mv_in_q = torch.randn(num_batch, num_q, 6, 16)
    s_in_q = torch.randn(num_batch, num_q, 6)

    t = SlowRandomPinTransform(spin=True)

    mv_out1, s_out1 = layer.forward(
        mv_in, mv_in_q, s_in, s_in_q, attention_mask=attn_mask
    )
    mv_out1 = t(mv_out1)

    mv_out2, s_out2 = layer.forward(
        t(mv_in), t(mv_in_q), s_in, s_in_q, attention_mask=attn_mask
    )

    assert mv_out1.shape == (num_batch, num_q, 6, 16)
    assert s_out1.shape == (num_batch, num_q, 4)

    torch.testing.assert_close(mv_out1, mv_out2)
    torch.testing.assert_close(s_out1, s_out2)
