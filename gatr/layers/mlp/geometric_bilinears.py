"""Pin-equivariant geometric product layer between multivector tensors (torch.nn.Modules)."""

from typing import Optional, Tuple

import torch
from torch import nn

from gatr.layers.linear import EquiLinear
from gatr.primitives import geometric_product
from gatr.layers.layer_norm import EquiLayerNorm

INCLUDE_TENSOR = True


class GeometricBilinear(nn.Module):
    """Geometric bilinear layer.

    Pin-equivariant map between multivector tensors that constructs new geometric features via
    geometric products and the equivariant join (based on a reference vector).

    Parameters
    ----------
    in_mv_channels : int
        Input multivector channels of `x`
    out_mv_channels : int
        Output multivector channels
    hidden_mv_channels : int or None
        Hidden MV channels. If None, uses out_mv_channels.
    in_s_channels : int or None
        Input scalar channels of `x`. If None, no scalars are expected nor returned.
    out_s_channels : int or None
        Output scalar channels. If None, no scalars are expected nor returned.
    """

    def __init__(
        self,
        in_mv_channels: int,
        out_mv_channels: int,
        hidden_mv_channels: Optional[int] = None,
        in_s_channels: Optional[int] = None,
        out_s_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Default options
        if hidden_mv_channels is None:
            hidden_mv_channels = out_mv_channels

        # Linear projections for GP
        self.linear_left = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
        )
        self.linear_right = EquiLinear(
            in_mv_channels,
            hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=None,
            initialization="almost_unit_scalar",
        )

        # Output linear projection
        self.linear_out = EquiLinear(
            hidden_mv_channels, out_mv_channels, in_s_channels, out_s_channels
        )
        self.norm = EquiLayerNorm()

    def forward(
        self,
        multivectors: torch.Tensor,
        scalars: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        multivectors : torch.Tensor with shape (..., in_mv_channels, 16)
            Input multivectors
        scalars : torch.Tensor with shape (..., in_s_channels)
            Input scalars

        Returns
        -------
        outputs_mv : torch.Tensor with shape (..., self.out_mv_channels, 16)
            Output multivectors
        output_s : None or torch.Tensor with shape (..., out_s_channels)
            Output scalars.
        """

        # GP
        left, _ = self.linear_left(multivectors, scalars=scalars)
        right, _ = self.linear_right(multivectors, scalars=scalars)
        gp_outputs = geometric_product(left, right)
        if not INCLUDE_TENSOR:
            gp_outputs[..., 5:11] = 0.0

        # Output linear
        outputs_mv, outputs_s = self.linear_out(gp_outputs, scalars=scalars)

        outputs_mv, outputs_s = self.norm(outputs_mv, outputs_s)
        return outputs_mv, outputs_s
