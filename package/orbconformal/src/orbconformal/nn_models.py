import torch
from torch import distributions
from torch import nn

import sys
sys.path.append("../../../../../pytorch-generative")

from pytorch_generative import nn as pg_nn
from pytorch_generative.models import base
from pytorch_generative.models.autoregressive.pixel_cnn import CausalResidualBlock


class PixelCNNMultiHead(base.AutoregressiveModel):
    """The PixelCNN model, initialized with multiple heads so to accommodate different
    orbs and other variables."""

    def __init__(
        self,
        channels_size, channels_rad,
        in_channels=1,
        out_channels=1,
        n_residual=15,
        residual_channels=128,
        head_channels=32,
        sample_fn=None,
    ):
        """Initializes a new PixelCNN Multi Head instance.

        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            n_residual: The number of residual blocks.
            residual_channels: The number of channels to use in the residual layers.
            head_channels: The number of channels to use in the two 1x1 convolutional
                layers at the head of the network.
            sample_fn: See the base class.
        """
        super().__init__(sample_fn)
        self._input_h1 = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=2 * residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._causal_layers_h1 = nn.ModuleList(
            [
                CausalResidualBlock(n_channels=2 * residual_channels)
                for _ in range(n_residual)
            ]
        )
        self._head_h1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * residual_channels,
                out_channels=head_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            )
        )


        self._input_h2 = pg_nn.CausalConv2d(
            mask_center=True,
            in_channels=in_channels,
            out_channels=2 * residual_channels,
            kernel_size=7,
            padding=3,
        )
        self._causal_layers_h2 = nn.ModuleList(
            [
                CausalResidualBlock(n_channels=2 * residual_channels)
                for _ in range(n_residual)
            ]
        )
        self._head_h2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=2 * residual_channels,
                out_channels=head_channels,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=head_channels, out_channels=out_channels, kernel_size=1
            )
        )

        self.cnn_size_final = nn.Conv2d(in_channels=out_channels + channels_size,
                                        out_channels=out_channels, kernel_size=1)
        self.cnn_rad_final = nn.Conv2d(in_channels=out_channels + channels_rad,
                                        out_channels=out_channels, kernel_size=1)

    def forward(self, size_orb, rad_orb, feats_size, feats_rad):
        # First propagate the first orb
        size_orb = self._input_h1(size_orb)
        for layer in self._causal_layers_h1:
            size_orb = size_orb + layer(size_orb)
        size_orb = self._head_h1(size_orb)

        # Then propagate the second orb
        rad_orb = self._input_h2(rad_orb)
        for layer in self._causal_layers_h2:
            rad_orb = rad_orb + layer(rad_orb)
        rad_orb = self._head_h2(rad_orb)

        # Attach all channels to both orbs
        size_orb_full = torch.cat((size_orb, feats_size), 1)
        rad_orb_full = torch.cat((rad_orb, feats_rad), 1)

        # Run the final convolutions
        size_preds = self.cnn_size_final(size_orb_full)
        rad_preds = self.cnn_rad_final(rad_orb_full)

        return size_preds, rad_preds
