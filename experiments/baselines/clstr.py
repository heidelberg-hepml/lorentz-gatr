# Copyright (c) 2023 Qualcomm Technologies, Inc.
# All rights reserved.
"""Baseline transformer."""

from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers import ApplyRotaryPositionalEncoding
from gatr.primitives.attention import scaled_dot_product_attention
from experiments.misc import to_nd
from experiments.baselines.transformer import BaselineTransformerBlock, MultiHeadQKVLinear, \
     MultiQueryQKVLinear, BaselineLayerNorm

class BaselineCrossAttention(nn.Module):
    """Baseline cross-attention layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of input channels.
    hidden_channels : int
        Number of hidden channels = size of query, key, and value.
    num_heads : int
        Number of attention heads.
    pos_encoding : bool
        Whether to apply rotary positional embeddings along the item dimension to the scalar keys
        and queries.
    pos_enc_base : int
        Maximum frequency used in positional encodings. (The minimum frequency is always 1.)
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_enc_base: int = 4096,
        multi_query: bool = False,
    ) -> None:
        super().__init__()

        # Store settings
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        # Linear maps
        qkv_class = MultiQueryQKVLinear if multi_query else MultiHeadQKVLinear
        self.qkv_linear = qkv_class(in_channels, hidden_channels, num_heads)
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        # Optional positional encoding
        if pos_encoding:
            self.pos_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels, item_dim=-2, base=pos_enc_base
            )
        else:
            self.pos_encoding = None

    def forward(
        self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor
            Input data
        attention_mask : None or Tensor or xformers.ops.AttentionBias
            Optional attention mask

        Returns
        -------
        outputs : Tensor
            Outputs
        """
        q, k, v = self.qkv_linear(inputs_q, inputs_kv)  # each: (..., num_heads, num_items, num_channels, 16)

        # Rotary positional encoding
        if self.pos_encoding is not None:
            q = self.pos_encoding(q)
            k = self.pos_encoding(k)

        # Attention layer
        h = self._attend(q, k, v, attention_mask)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        return outputs

    @staticmethod
    def _attend(q, k, v, attention_mask=None):
        """Scaled dot-product attention."""

        # Add batch dimension if needed
        bh_shape = q.shape[:-2]
        q = to_nd(q, 4)
        k = to_nd(k, 4)
        v = to_nd(v, 4)

        # SDPA
        outputs = scaled_dot_product_attention(
            q.contiguous(), k.contiguous(), v.expand_as(k), attn_mask=attention_mask
        )

        # Return batch dimensions to inputs
        outputs = outputs.view(*bh_shape, *outputs.shape[-2:])

        return outputs

class BaselineCrossAttentionBlock(nn.Module):

    def __init__(
        self,
        channels,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        increase_hidden_channels=1,
        multi_query: bool = True,
    ) -> None:
        super().__init__()
        

        self.norm = BaselineLayerNorm()
        
        hidden_channels = channels // num_heads * increase_hidden_channels
        if pos_encoding:
            hidden_channels = (hidden_channels + 1) // 2 * 2
            hidden_channels = max(hidden_channels, 16)

        self.attention = BaselineCrossAttention(
            channels,
            channels,
            hidden_channels,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            pos_enc_base=pos_encoding_base,
            multi_query=multi_query,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
        )

    def forward(self, inputs_q: torch.Tensor, inputs_kv: torch.Tensor):
        q = self.norm(inputs_q)
        kv = self.norm(inputs_kv)
        h = self.attention(q, kv)
        outputs = inputs_q + h

        h = self.norm(outputs)
        h = self.mlp(h)
        outputs = outputs + h

        return outputs

class CLSTr(nn.Module):
    """Baseline CLSTr. Combines a series of self-attention transformer blocks
    and another series of cross-attention transformer blocks.

    Attention masks not implemented here

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hidden_channels : int
        Number of hidden channels.
    num_sa_blocks : int
        Number of self-attention transformer blocks.
    num_ca_blocks : int
        Number of cross-attention transformer blocks.
    num_heads : int
        Number of attention heads.
    increase_hidden_channels : int
        Factor by which the key, query, and value size is increased over the default value of
        hidden_channels / num_heads.
    multi_query : bool
        Use multi-query attention instead of multi-head attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_sa_blocks: int = 5,
        num_ca_blocks: int = 5,
        num_heads: int = 8,
        num_classes: int = 1,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        checkpoint_blocks: bool = False,
        increase_hidden_channels=1,
        multi_query_sa: bool = False,
        multi_query_ca: bool = False,
    ) -> None:
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.sa_blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query_sa,
                )
                for _ in range(num_sa_blocks)
            ]
        )
        self.ca_blocks = nn.ModuleList(
            [
                BaselineCrossAttentionBlock(
                    hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query_ca,
                )
                for _ in range(num_ca_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

        self.class_token = nn.Parameter(torch.randn(1,num_classes,hidden_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        inputs : Tensor with shape (..., num_items, num_channels)
            Input data


        Returns
        -------
        outputs : Tensor with shape (..., num_items, num_channels)
            Outputs
        """
        h = self.linear_in(inputs)
        for block in self.sa_blocks:
            if self.checkpoint_blocks:
                fn = partial(block)
                h = checkpoint(fn, h)
            else:
                h = block(h)

        class_token = self.class_token.expand(h.shape[0], self.class_token.shape[1], self.class_token.shape[2])
        for block in self.ca_blocks:
            if self.checkpoint_blocks:
                fn = partial(block)
                class_token = checkpoint(fn, class_token, h)
            else:
                class_token = block(class_token, h)
        outputs = self.linear_out(class_token)
        return outputs
