from functools import partial
from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.utils.checkpoint import checkpoint

from gatr.layers import ApplyRotaryPositionalEncoding
from gatr.primitives.attention import scaled_dot_product_attention
from experiments.misc import to_nd

from experiments.baselines.transformer import (
    BaselineSelfAttention,
    BaselineLayerNorm,
    BaselineTransformerBlock,
)


class CrossAttention(nn.Module):
    def __init__(
        self,
        in_q_channels: int,
        in_kv_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        multi_query: bool = True,
        dropout_prob: Optional[float] = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels

        self.q_linear = nn.Linear(in_q_channels, hidden_channels * num_heads)
        self.kv_linear = nn.Linear(
            in_kv_channels, 2 * hidden_channels * (1 if multi_query else num_heads)
        )
        self.out_linear = nn.Linear(hidden_channels * num_heads, out_channels)

        if pos_encoding:
            self.q_pos_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels, item_dim=-2, base=pos_encoding_base
            )
            self.k_pos_encoding = ApplyRotaryPositionalEncoding(
                hidden_channels, item_dim=-2, base=pos_encoding_base
            )
        else:
            self.q_pos_encoding = None
            self.k_pos_encoding = None

        self.dropout = nn.Dropout(dropout_prob) if dropout_prob is not None else None

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        q = self.q_linear(q)
        k, v = torch.tensor_split(self.kv_linear(kv), 2, dim=-1)

        if self.q_pos_encoding is not None:
            q = self.q_pos_encoding(q)
            k = self.k_pos_encoding(k)

        # Attention layer
        h = BaselineSelfAttention._attend(q, k, v, attention_mask)

        # Concatenate heads and transform linearly
        h = rearrange(
            h,
            "... num_heads num_items hidden_channels -> ... num_items (num_heads hidden_channels)",
        )
        outputs = self.out_linear(h)  # (..., num_items, out_channels)

        if self.dropout is not None:
            outputs = self.dropout(outputs)

        return outputs


class ConditionalTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        conditional_channels: int,
        num_heads: int,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        increase_hidden_channels=1,
        multi_query: bool = True,
        dropout_prob: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.norm = BaselineLayerNorm()

        hidden_channels = channels // num_heads * increase_hidden_channels
        if pos_encoding:
            hidden_channels = (hidden_channels + 1) // 2 * 2
            hidden_channels = max(hidden_channels, 16)

        self.self_attention = BaselineSelfAttention(
            in_channels=channels,
            out_channels=channels,
            hidden_channels=hidden_channels,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            pos_encoding_base=pos_encoding_base,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.cross_attention = CrossAttention(
            in_q_channels=channels,
            in_kv_channels=conditional_channels,
            hidden_channels=channels,
            out_channels=channels,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
            pos_encoding_base=pos_encoding_base,
            multi_query=multi_query,
            dropout_prob=dropout_prob,
        )

        self.mlp = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
            nn.GELU(),
            nn.Linear(2 * channels, channels),
            nn.Dropout(dropout_prob) if dropout_prob is not None else nn.Identity(),
        )

    def forward(self, inputs, condition, attention_mask=None, crossattention_mask=None):
        # self-attention
        h = self.norm(inputs)
        inputs = self.self_attention(h, attention_mask=attention_mask) + inputs

        # cross-attention
        h = self.norm(inputs)
        condition = self.norm(condition)
        output = (
            self.cross_attention(h, condition, attention_mask=crossattention_mask)
            + inputs
        )

        # mlp
        h = self.norm(output)
        output = self.mlp(h) + output
        return output


class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        condition_in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int = 10,
        num_heads: int = 8,
        pos_encoding: bool = False,
        pos_encoding_base: int = 4096,
        checkpoint_blocks: bool = False,
        increase_hidden_channels=1,
        multi_query: bool = False,
        dropout_prob=None,
    ) -> None:
        super().__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.linear_in = nn.Linear(in_channels, hidden_channels)
        self.condition_linear_in = nn.Linear(condition_in_channels, hidden_channels)

        self.condition_blocks = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    channels=hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                ConditionalTransformerBlock(
                    channels=hidden_channels,
                    conditional_channels=hidden_channels,
                    num_heads=num_heads,
                    pos_encoding=pos_encoding,
                    pos_encoding_base=pos_encoding_base,
                    increase_hidden_channels=increase_hidden_channels,
                    multi_query=multi_query,
                    dropout_prob=dropout_prob,
                )
                for _ in range(num_blocks)
            ]
        )
        self.linear_out = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        condition_attention_mask: Optional[torch.Tensor] = None,
        crossattention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        condition = self.condition_linear_in(condition)
        for block in self.condition_blocks:
            if self.checkpoint_blocks:
                condition = checkpoint(
                    block, inputs=condition, attention_mask=condition_attention_mask
                )
            else:
                condition = block(condition, attention_mask=condition_attention_mask)

        x = self.linear_in(x)
        for block in self.blocks:
            if self.checkpoint_blocks:
                x = checkpoint(
                    block,
                    inputs=x,
                    condition=condition,
                    attention_mask=attention_mask,
                    crossattention_mask=crossattention_mask,
                )
            else:
                x = block(
                    x,
                    condition,
                    attention_mask=attention_mask,
                    crossattention_mask=crossattention_mask,
                )

        return self.linear_out(x)
