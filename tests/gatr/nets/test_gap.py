import pytest
import torch

from gatr.layers.mlp.config import MLPConfig
from gatr.nets import GAP
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(None, None, 7), (4, 5, 6)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_blocks", [1])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
@pytest.mark.parametrize("dropout_prob", [None, 0.0, 0.3])
def test_gatr_shape(
    batch_dims,
    num_items,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
    dropout_prob,
):
    """Tests the output shape of GAP."""
    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items, in_s_channels)
    )

    try:
        net = GAP(
            in_mv_channels,
            out_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            mlp=MLPConfig(),
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(inputs, scalars=scalars)

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if in_s_channels is not None:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)


@pytest.mark.parametrize("batch_dims", [(64,)])
@pytest.mark.parametrize(
    "num_items,in_mv_channels,out_mv_channels,hidden_mv_channels", [(8, 3, 4, 6)]
)
@pytest.mark.parametrize("num_blocks", [1])
@pytest.mark.parametrize("in_s_channels,out_s_channels,hidden_s_channels", S_CHANNELS)
def test_gatr_equivariance(
    batch_dims,
    num_items,
    in_mv_channels,
    out_mv_channels,
    hidden_mv_channels,
    num_blocks,
    in_s_channels,
    out_s_channels,
    hidden_s_channels,
):
    """Tests GATr for equivariance."""
    try:
        net = GAP(
            in_mv_channels,
            out_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            mlp=MLPConfig(),
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items, in_s_channels)
    )
    data_dims = tuple(list(batch_dims) + [num_items, in_mv_channels])
    check_pin_equivariance(
        net, 1, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )
