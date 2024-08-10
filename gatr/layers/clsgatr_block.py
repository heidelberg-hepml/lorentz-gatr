from dataclasses import replace
from typing import Optional, Tuple

import torch
from torch import nn

from gatr.layers import CrossAttention, SelfAttentionConfig, CrossAttentionConfig
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.layers.mlp.config import MLPConfig
from gatr.layers.mlp.mlp import GeoMLP


class CLSGATrBlock(nn.Module):
    """Equivariant class-transformer block for L-CLSGATr.

    Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
    cross-attention, and a residual connection. Then the data is processed by a block consisting of
    another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and another
    residual connection.

    Parameters
    ----------
    mv_channels : int
        Number of input and output multivector channels
    s_channels: int
        Number of input and output scalar channels
    crossattention: CrossAttentionConfig
        Cross-attention configuration
    mlp: MLPConfig
        MLP configuration
    dropout_prob : float or None
        Dropout probability
    double_layernorm : bool
        Whether to use double layer normalization
    """

    def __init__(
        self,
        mv_channels: int,
        s_channels: int,
        crossattention: CrossAttentionConfig,
        mlp: MLPConfig,
        dropout_prob: Optional[float] = None,
        double_layernorm: bool = False,
    ) -> None:
        super().__init__()

        # Normalization layer (stateless, so we can use the same layer for both normalization
        # instances)
        self.norm = EquiLayerNorm()
        self.double_layernorm = double_layernorm

        # Cross-attention layer
        crossattention = replace(
            crossattention,
            in_q_mv_channels=mv_channels,
            in_kv_mv_channels=mv_channels,
            out_mv_channels=mv_channels,
            in_q_s_channels=s_channels,
            in_kv_s_channels=s_channels,
            out_s_channels=s_channels,
            output_init="small",
            dropout_prob=dropout_prob,
        )
        self.attention = CrossAttention(crossattention)

        # MLP block
        mlp = replace(
            mlp,
            mv_channels=(mv_channels, 2 * mv_channels, mv_channels),
            s_channels=(s_channels, 2 * s_channels, s_channels),
            dropout_prob=dropout_prob,
        )
        self.mlp = GeoMLP(mlp)

    def forward(
        self,
        multivectors_q: torch.Tensor,
        multivectors_kv: torch.Tensor,
        scalars_q: torch.Tensor = None,
        scalars_kv: torch.Tensor = None,
        additional_q_features_mv=None,
        additional_k_features_mv=None,
        additional_q_features_s=None,
        additional_k_features_s=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the class-transformer block.

        Inputs are first processed by a block consisting of LayerNorm, multi-head geometric
        cross-attention, and a residual connection. Then the data is processed by a block consisting
        of another LayerNorm, an item-wise two-layer geometric MLP with GeLU activations, and
        another residual connection.

        Parameters
        ----------
        multivectors_q : torch.Tensor with shape (..., items, channels, 16)
            Input class multivectors.
        multivectors_kv : torch.Tensor with shape (..., items, channels, 16)
            Input reference multivectors.
        scalars_q : torch.Tensor with shape (..., s_channels)
            Input class scalars.
        scalars_kv : torch.Tensor with shape (..., s_channels)
            Input reference scalars.
        additional_q_features_mv : None or torch.Tensor with shape
            (..., num_items, add_q_mv_channels, 16)
            Additional Q features, multivector part.
        additional_k_features_mv : None or torch.Tensor with shape
            (..., num_items, add_k_mv_channels, 16)
            Additional K features, multivector part.
        additional_q_features_s : None or torch.Tensor with shape
            (..., num_items, add_q_s_channels, 16)
            Additional Q features, scalar part.
        additional_k_features_s : None or torch.Tensor with shape
            (..., num_items, add_k_s_channels, 16)
            Additional K features, scalar part.
        attention_mask: None or torch.Tensor or AttentionBias
            Optional attention mask.

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., items, channels, 16).
            Output multivectors
        outputs_s : torch.Tensor with shape (..., s_channels)
            Output scalars
        """

        # Attention block: pre layer norm
        mv_q, s_q = self.norm(multivectors_q, scalars=scalars_q)

        # Attention block: self attention
        mv, s = self.attention(
            multivectors_q=mv_q,
            scalars_q=s_q,
            multivectors_kv=multivectors_kv,
            scalars_kv=scalars_kv,
            additional_q_features_mv=additional_q_features_mv,
            additional_k_features_mv=additional_k_features_mv,
            additional_q_features_s=additional_q_features_s,
            additional_k_features_s=additional_k_features_s,
            attention_mask=attention_mask,
        )

        # Attention block: post layer norm
        if self.double_layernorm:
            mv, s = self.norm(mv, scalars=s)

        # Attention block: skip connection
        outputs_mv = multivectors_q + mv
        outputs_s = scalars_q + s

        # MLP block: pre layer norm
        mv, s = self.norm(outputs_mv, scalars=outputs_s)

        # MLP block: MLP
        mv, s = self.mlp(mv, scalars=s)

        # MLP block: post layer norm
        if self.double_layernorm:
            mv, s = self.norm(mv, scalars=s)

        # MLP block: skip connection
        outputs_mv = outputs_mv + mv
        outputs_s = outputs_s + s

        return outputs_mv, outputs_s
