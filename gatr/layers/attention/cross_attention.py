"""Cross-attention layer."""

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from gatr.layers.attention.attention import GeometricAttention
from gatr.layers.attention.config import CrossAttentionConfig
from gatr.layers.dropout import GradeDropout
from gatr.layers.linear import EquiLinear


class CrossAttention(nn.Module):
    """Geometric cross-attention layer.

    Constructs queries, keys, and values, computes attention, and projects linearly to outputs.

    Parameters
    ----------
    config : CrossAttentionConfig
        Attention configuration.
    """

    def __init__(
        self,
        config: CrossAttentionConfig,
    ) -> None:
        super().__init__()
        assert (
            not config.multi_query
        ), "multi_query=True not implemented yet for cross-attention"

        if config.pos_encoding:
            raise NotImplementedError(
                "Cross attention is not implemented with positional encoding."
            )

        # Store settings
        self.config = config
        self.q_linear = EquiLinear(
            in_mv_channels=config.in_q_mv_channels + config.additional_q_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_q_s_channels + config.additional_q_s_channels,
            out_s_channels=None
            if config.hidden_s_channels is None
            else config.hidden_s_channels * config.num_heads,
        )

        self.k_linear = EquiLinear(
            in_mv_channels=config.in_kv_mv_channels + config.additional_k_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_kv_s_channels + config.additional_k_s_channels,
            out_s_channels=None
            if config.hidden_s_channels is None
            else config.hidden_s_channels * config.num_heads,
        )
        self.v_linear = EquiLinear(
            in_mv_channels=config.in_kv_mv_channels + config.additional_k_mv_channels,
            out_mv_channels=config.hidden_mv_channels * config.num_heads,
            in_s_channels=config.in_kv_s_channels + config.additional_k_s_channels,
            out_s_channels=None
            if config.hidden_s_channels is None
            else config.hidden_s_channels * config.num_heads,
        )

        # Output projection
        self.out_linear = EquiLinear(
            in_mv_channels=config.hidden_mv_channels * config.num_heads,
            out_mv_channels=config.out_mv_channels,
            in_s_channels=(
                None
                if config.in_q_s_channels is None
                else config.hidden_s_channels * config.num_heads
            ),
            out_s_channels=config.out_s_channels,
            initialization=config.output_init,
        )

        # Attention
        self.attention = GeometricAttention(config)

        # Dropout
        self.dropout: Optional[nn.Module]
        if config.dropout_prob is not None:
            self.dropout = GradeDropout(config.dropout_prob)
        else:
            self.dropout = None

    def forward(
        self,
        multivectors_kv: torch.Tensor,
        multivectors_q: torch.Tensor,
        scalars_kv: Optional[torch.Tensor] = None,
        scalars_q: Optional[torch.Tensor] = None,
        additional_q_features_mv=None,
        additional_k_features_mv=None,
        additional_q_features_s=None,
        additional_k_features_s=None,
        attention_mask=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross attention.

        Parameters
        ----------
        multivectors_kv : torch.Tensor with shape (..., num_items_kv, channels_in, 16)
            Input multivectors for key and value.
        multivectors_q : torch.Tensor with shape (..., num_items_q, channels_in_q, 16)
            Input multivectors for query.
        scalars_kv : None or torch.Tensor with shape (..., num_items_kv, in_scalars)
            Optional input scalars
        scalars_q : None or torch.Tensor with shape (..., num_items_q, in_scalars_q)
            Optional input scalars for query
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
        attention_mask: torch.Tensor with shape (..., num_items_q, num_items_kv) or xformers mask.
            Attention mask

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., num_items_q, channels_out, 16)
            Output multivectors.
        output_scalars : torch.Tensor with shape (..., num_items_q, channels_out, out_scalars)
            Output scalars, if scalars are provided. Otherwise None.
        """
        # Additional inputs
        if additional_q_features_mv is not None:
            multivectors_q = torch.cat(
                (multivectors_q, additional_q_features_mv), dim=-2
            )
        if additional_k_features_mv is not None:
            multivectors_kv = torch.cat(
                (multivectors_kv, additional_k_features_mv), dim=-2
            )
        if additional_q_features_s is not None:
            scalars_q = torch.cat((scalars_q, additional_q_features_s), dim=-1)
        if additional_k_features_s is not None:
            scalars_kv = torch.cat((scalars_kv, additional_k_features_s), dim=-1)

        q_mv, q_s = self.q_linear(
            multivectors_q, scalars_q
        )  # (..., num_items, num_heads*hidden_channels, 16)
        k_mv, k_s = self.k_linear(
            multivectors_kv, scalars_kv
        )  # (..., num_items, num_heads*hidden_channels, 16)
        v_mv, v_s = self.v_linear(
            multivectors_kv, scalars_kv
        )  # (..., num_items, num_heads*hidden_channels, 16)

        # Rearrange to (..., heads, items, channels, 16) shape
        q_mv = rearrange(
            q_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        k_mv = rearrange(
            k_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )
        v_mv = rearrange(
            v_mv,
            "... items (hidden_channels num_heads) x -> ... num_heads items hidden_channels x",
            num_heads=self.config.num_heads,
            hidden_channels=self.config.hidden_mv_channels,
        )

        # Same for scalars
        if q_s is not None:
            q_s = rearrange(
                q_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            k_s = rearrange(
                k_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
            v_s = rearrange(
                v_s,
                "... items (hidden_channels num_heads) -> ... num_heads items hidden_channels",
                num_heads=self.config.num_heads,
                hidden_channels=self.config.hidden_s_channels,
            )
        else:
            q_s, k_s, v_s = None, None, None

        # Attention layer
        h_mv, h_s = self.attention(
            q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask
        )

        h_mv = rearrange(
            h_mv,
            "... n_heads n_items hidden_channels x -> ... n_items (n_heads hidden_channels) x",
        )
        h_s = rearrange(
            h_s,
            "... n_heads n_items hidden_channels -> ... n_items (n_heads hidden_channels)",
        )

        # Transform linearly one more time
        outputs_mv, outputs_s = self.out_linear(h_mv, scalars=h_s)

        # Dropout
        if self.dropout is not None:
            outputs_mv, outputs_s = self.dropout(outputs_mv, outputs_s)

        return outputs_mv, outputs_s
