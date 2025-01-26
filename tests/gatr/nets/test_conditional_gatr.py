import pytest
import torch

from gatr.layers import SelfAttentionConfig, CrossAttentionConfig, MLPConfig
from gatr.nets import ConditionalGATr
from tests.helpers import BATCH_DIMS, MILD_TOLERANCES, check_pin_equivariance

S_CHANNELS = [(3, 5, True), (2, 2, False)]


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,in_mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize(
    "in_s_channels,in_s_channels_condition,pos_encoding", S_CHANNELS
)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_shape(
    batch_dims,
    num_items,
    num_items_condition,
    in_mv_channels,
    in_mv_channels_condition,
    hidden_mv_channels,
    out_mv_channels,
    num_heads,
    num_blocks,
    in_s_channels,
    in_s_channels_condition,
    hidden_s_channels,
    out_s_channels,
    pos_encoding,
    multi_query_attention,
    dropout_prob,
):
    """Tests the output shape of ConditionalGATr."""
    inputs = torch.randn(*batch_dims, num_items, in_mv_channels, 16)
    scalars = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items, in_s_channels)
    )
    condition_mv = torch.randn(
        *batch_dims, num_items_condition, in_mv_channels_condition, 16
    )
    condition_s = (
        None
        if in_s_channels is None
        else torch.randn(*batch_dims, num_items_condition, in_s_channels_condition)
    )

    try:
        net = ConditionalGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            condition_mv_channels=in_mv_channels_condition,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            condition_s_channels=in_s_channels_condition,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                pos_encoding=pos_encoding,
                multi_query=multi_query_attention,
            ),
            attention_condition=SelfAttentionConfig(
                num_heads=num_heads,
                pos_encoding=pos_encoding,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    outputs, output_scalars = net(
        inputs,
        scalars=scalars,
        multivectors_condition=condition_mv,
        scalars_condition=condition_s,
    )

    assert outputs.shape == (*batch_dims, num_items, out_mv_channels, 16)
    if in_s_channels is not None:
        assert output_scalars.shape == (*batch_dims, num_items, out_s_channels)


@pytest.mark.parametrize("batch_dims", BATCH_DIMS)
@pytest.mark.parametrize("num_items,num_items_condition", [(2, 2), (2, 9)])
@pytest.mark.parametrize("in_mv_channels,in_mv_channels_condition", [(6, 6), (7, 11)])
@pytest.mark.parametrize("num_blocks,num_heads", [(1, 4)])
@pytest.mark.parametrize(
    "in_s_channels,in_s_channels_condition,pos_encoding", S_CHANNELS
)
@pytest.mark.parametrize("hidden_mv_channels,hidden_s_channels", [(9, 4)])
@pytest.mark.parametrize("out_mv_channels,out_s_channels", [(8, 5)])
@pytest.mark.parametrize("dropout_prob", [None])
@pytest.mark.parametrize("multi_query_attention", [False, True])
def test_conditional_gatr_equivariance(
    batch_dims,
    num_items,
    num_items_condition,
    in_mv_channels,
    in_mv_channels_condition,
    hidden_mv_channels,
    out_mv_channels,
    num_heads,
    num_blocks,
    in_s_channels,
    in_s_channels_condition,
    hidden_s_channels,
    out_s_channels,
    pos_encoding,
    multi_query_attention,
    dropout_prob,
):
    """Tests ConditionalGATr for equivariance."""

    try:
        net = ConditionalGATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            condition_mv_channels=in_mv_channels_condition,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            condition_s_channels=in_s_channels_condition,
            attention=SelfAttentionConfig(
                num_heads=num_heads,
                pos_encoding=pos_encoding,
                multi_query=multi_query_attention,
            ),
            attention_condition=SelfAttentionConfig(
                num_heads=num_heads,
                pos_encoding=pos_encoding,
                multi_query=multi_query_attention,
            ),
            crossattention=CrossAttentionConfig(
                num_heads=num_heads,
                multi_query=multi_query_attention,
            ),
            mlp=MLPConfig(),
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
        )
    except NotImplementedError:
        # Some features require scalar inputs, and failing without them is fine
        return

    scalars = torch.randn(*batch_dims, num_items, in_s_channels)
    scalars_condition = torch.randn(
        *batch_dims, num_items_condition, in_s_channels_condition
    )

    data_dims = [
        tuple(list(batch_dims) + [num_items, in_mv_channels]),
        tuple(list(batch_dims) + [num_items_condition, in_mv_channels_condition]),
    ]
    check_pin_equivariance(
        net, 2, batch_dims=data_dims, fn_kwargs=dict(scalars=scalars), **MILD_TOLERANCES
    )
