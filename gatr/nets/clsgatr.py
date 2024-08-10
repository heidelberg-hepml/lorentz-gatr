"""Equivariant transformer for multivector data."""

from dataclasses import replace
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers.attention.config import SelfAttentionConfig, CrossAttentionConfig
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.clsgatr_block import CLSGATrBlock
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.config import MLPConfig


class CLSGATr(nn.Module):
    """L-CLSGATr network for a data with a single token dimension,
    following the CaiT architecture of https://arxiv.org/pdf/2103.17239.

    It combines `num_sa_blocks` L-GATr self-attention transformer blocks and `num_ca_blocks` L-GATr
    cross-attention transformer blocks, each consisting of geometric self-attention / cross-attention
    layers, a geometric MLP, residual connections, and normalization layers. In addition, there
    are initial and final equivariant linear layers.

    Assumes input has shape `(..., items, in_channels, 16)`, output has shape
    `(..., items, out_channels, 16)`, will create hidden representations with shape
    `(..., items, hidden_channels, 16)`.

    Parameters
    ----------
    in_mv_channels : int
        Number of input multivector channels.
    out_mv_channels : int
        Number of output multivector channels.
    hidden_mv_channels : int
        Number of hidden multivector channels.
    in_s_channels : None or int
        If not None, sets the number of scalar input channels.
    out_s_channels : None or int
        If not None, sets the number of scalar output channels.
    hidden_s_channels : None or int
        If not None, sets the number of scalar hidden channels.
    attention: Dict
        Data for SelfAttentionConfig
    crossattention: Dict
        Data for CrossAttentionConfig
    mlp: Dict
        Data for MLPConfig
    num_sa_blocks : int
        Number of self-attention transformer blocks.
    num_ca_blocks : int
        Number of cross-attention transformer blocks.
    dropout_prob : float or None
        Dropout probability
    double_layernorm : bool
        Whether to use double layer normalization
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: int,
        in_s_channels: Optional[int],
        out_s_channels: Optional[int],
        hidden_s_channels: Optional[int],
        attention: SelfAttentionConfig,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        num_sa_blocks: int = 10,
        num_ca_blocks: int = 2,
        reinsert_mv_channels: Optional[Tuple[int]] = None,
        reinsert_s_channels: Optional[Tuple[int]] = None,
        checkpoint_blocks: bool = False,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
        num_class_tokens: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert num_ca_blocks > 0, (
            f"CaiT requires num_ca_blocks>0 for point cloud aggregation, "
            f"but num_ca_blocks={num_ca_blocks}"
        )
        self.num_class_tokens = num_class_tokens
        self.linear_in = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=hidden_s_channels,
        )
        attention = replace(
            SelfAttentionConfig.cast(attention),
            additional_qk_mv_channels=0
            if reinsert_mv_channels is None
            else len(reinsert_mv_channels),
            additional_qk_s_channels=0
            if reinsert_s_channels is None
            else len(reinsert_s_channels),
        )
        crossattention = replace(
            CrossAttentionConfig.cast(crossattention),
            additional_q_mv_channels=0,
            additional_q_s_channels=0,
            additional_k_mv_channels=0
            if reinsert_mv_channels is None
            else len(reinsert_mv_channels),
            additional_k_s_channels=0
            if reinsert_s_channels is None
            else len(reinsert_s_channels),
        )
        mlp = MLPConfig.cast(mlp)
        self.sa_blocks = nn.ModuleList(
            [
                GATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    attention=attention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    double_layernorm=double_layernorm,
                )
                for _ in range(num_sa_blocks)
            ]
        )
        self.ca_blocks = nn.ModuleList(
            [
                CLSGATrBlock(
                    mv_channels=hidden_mv_channels,
                    s_channels=hidden_s_channels,
                    crossattention=crossattention,
                    mlp=mlp,
                    dropout_prob=dropout_prob,
                    double_layernorm=double_layernorm,
                )
                for _ in range(num_ca_blocks)
            ]
        )
        self.linear_out = EquiLinear(
            hidden_mv_channels,
            out_mv_channels,
            in_s_channels=hidden_s_channels,
            out_s_channels=out_s_channels,
        )
        self.cls_mv = nn.Parameter(
            torch.randn(num_class_tokens, hidden_mv_channels, 16)
        )
        self.cls_s = nn.Parameter(torch.randn(num_class_tokens, hidden_s_channels))
        self._reinsert_s_channels = reinsert_s_channels
        self._reinsert_mv_channels = reinsert_mv_channels
        self._checkpoint_blocks = checkpoint_blocks

    def forward(
        self,
        multivectors: torch.Tensor,
        batchsize: int,
        scalars: Optional[torch.Tensor] = None,
        selfattn_mask: Optional[torch.Tensor] = None,
        crossattn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        """Forward pass of the network.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors.
        scalars : None or torch.Tensor with shape (..., in_s_channels)
            Optional input scalars.
        selfattn_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask
        crossattn_mask: None or torch.Tensor with shape (..., num_items, num_items)
            Optional attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., out_mv_channels, 16)
            Output multivectors.
        outputs_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Channels that will be re-inserted in any query / key computation
        (
            ref_additional_qk_features_mv,
            ref_additional_qk_features_s,
        ) = self._construct_reinserted_channels(multivectors, scalars)

        # Pass through the blocks
        ref_mv, ref_s = self.linear_in(multivectors, scalars=scalars)
        for block in self.sa_blocks:
            if self._checkpoint_blocks:
                ref_mv, ref_s = checkpoint(
                    block,
                    ref_mv,
                    use_reentrant=False,
                    scalars=ref_s,
                    additional_qk_features_mv=ref_additional_qk_features_mv,
                    additional_qk_features_s=ref_additional_qk_features_s,
                    attention_mask=selfattn_mask,
                )
            else:
                ref_mv, ref_s = block(
                    ref_mv,
                    scalars=ref_s,
                    additional_qk_features_mv=ref_additional_qk_features_mv,
                    additional_qk_features_s=ref_additional_qk_features_s,
                    attention_mask=selfattn_mask,
                )
        cls_mv = self.cls_mv.unsqueeze(0).repeat(1, batchsize, 1, 1)
        cls_s = self.cls_s.unsqueeze(0).repeat(1, batchsize, 1)
        for block in self.ca_blocks:
            if self._checkpoint_blocks:
                cls_mv, cls_s = checkpoint(
                    block,
                    multivectors_q=cls_mv,
                    use_reentrant=False,
                    scalars_q=cls_s,
                    multivectors_kv=ref_mv,
                    scalars_kv=ref_s,
                    additional_k_features_mv=ref_additional_qk_features_mv,
                    additional_k_features_s=ref_additional_qk_features_s,
                    attention_mask=crossattn_mask,
                )
            else:
                cls_mv, cls_s = block(
                    multivectors_q=cls_mv,
                    scalars_q=cls_s,
                    multivectors_kv=ref_mv,
                    scalars_kv=ref_s,
                    additional_k_features_mv=ref_additional_qk_features_mv,
                    additional_k_features_s=ref_additional_qk_features_s,
                    attention_mask=crossattn_mask,
                )

        outputs_mv, outputs_s = self.linear_out(cls_mv, scalars=cls_s)

        return outputs_mv, outputs_s

    def _construct_reinserted_channels(self, multivectors, scalars):
        """Constructs input features that will be reinserted in every attention layer."""

        if self._reinsert_mv_channels is None:
            additional_qk_features_mv = None
        else:
            additional_qk_features_mv = multivectors[..., self._reinsert_mv_channels, :]

        if self._reinsert_s_channels is None:
            additional_qk_features_s = None
        else:
            assert scalars is not None
            additional_qk_features_s = scalars[..., self._reinsert_s_channels]

        return additional_qk_features_mv, additional_qk_features_s
