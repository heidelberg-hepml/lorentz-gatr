import pytest
import torch

from gatr.primitives import sdp_attention
from tests.helpers import BATCH_DIMS, TOLERANCES, check_pin_equivariance


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_s_channels_out", [11, 1])
@pytest.mark.parametrize("num_s_channels_in", [17, 1])
@pytest.mark.parametrize("num_mv_channels_out", [3, 1])
@pytest.mark.parametrize("num_mv_channels_in", [2, 1])
@pytest.mark.parametrize("num_tokens_out", [5, 1])
@pytest.mark.parametrize("num_tokens_in", [7, 1])
def test_scalar_attention_shape(
    batch_dims,
    num_tokens_in,
    num_tokens_out,
    num_mv_channels_in,
    num_mv_channels_out,
    num_s_channels_in,
    num_s_channels_out,
):
    """Tests that outputs of scalar_attention() have correct shape."""
    # Generate inputs
    q_mv = torch.randn(*batch_dims, num_tokens_out, num_mv_channels_in, 16)
    k_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_in, 16)
    v_mv = torch.randn(*batch_dims, num_tokens_in, num_mv_channels_out, 16)
    q_s = torch.randn(*batch_dims, num_tokens_out, num_s_channels_in)
    k_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_in)
    v_s = torch.randn(*batch_dims, num_tokens_in, num_s_channels_out)

    # Compute attention outputs
    outputs, outputs_scalar = sdp_attention(q_mv, k_mv, v_mv, q_s, k_s, v_s)

    # Check shape of outputs
    assert outputs.shape == (*batch_dims, num_tokens_out, num_mv_channels_out, 16)
    assert outputs_scalar.shape == (*batch_dims, num_tokens_out, num_s_channels_out)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_scalars", [5])
@pytest.mark.parametrize("key_dim", [2])
@pytest.mark.parametrize("item_dim", [3])
def test_scalar_attention_equivariance(batch_dims, key_dim, item_dim, num_scalars):
    """Tests scalar_attention() for Pin equivariance."""
    data_dims = tuple(list(batch_dims) + [item_dim, key_dim])
    queries_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    keys_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    values_scalar = torch.randn(*batch_dims, item_dim, num_scalars)
    kwargs = dict(q_s=queries_scalar, k_s=keys_scalar, v_s=values_scalar)
    check_pin_equivariance(
        sdp_attention, 3, batch_dims=data_dims, fn_kwargs=kwargs, **TOLERANCES
    )
