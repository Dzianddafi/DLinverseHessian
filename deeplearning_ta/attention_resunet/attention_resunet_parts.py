import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual convolutional block.

    This block applies two convolutional layers and adds a residual shortcut
    connection from the input to the output.

    Architecture:
        Input
          ├── Conv → BN → ReLU → Conv → BN → ReLU
          └── 1×1 Conv (shortcut)
        Output = ReLU(main_path + shortcut)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of intermediate channels.
            If None, defaults to `out_channels`.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3,
                padding="same", bias=False
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                mid_channels, out_channels, kernel_size=3,
                padding="same", bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, in_channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
            `(batch_size, out_channels, height, width)`.
        """
        x1 = self.conv_block(x)
        x2 = self.shortcut(x)
        x = torch.add(x1, x2)
        x = self.relu(x)
        return x


class AttentionGate(nn.Module):
    """Attention gate for U-Net skip connections.

    This gate suppresses irrelevant encoder features by conditioning them
    on decoder (gating) features.

    Args:
        in_channels_g (int): Number of channels in the gating signal
            (decoder feature map).
        in_channels_x (int): Number of channels in the skip connection
            (encoder feature map).
    """

    def __init__(self, in_channels_g, in_channels_x):
        super().__init__()
        self.conv_x = nn.Conv2d(
            in_channels_x, in_channels_x,
            kernel_size=1, stride=2, bias=False
        )
        self.conv_g = nn.Conv2d(
            in_channels_g, in_channels_x,
            kernel_size=1, stride=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels_x, 1, kernel_size=1,
            padding="same", bias=True
        )
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )

    def forward(self, g, x):
        """Forward pass of the attention gate.

        Args:
            g (torch.Tensor): Gating signal from the decoder.
            x (torch.Tensor): Skip connection feature map from the encoder.

        Returns:
            torch.Tensor: Attention-weighted skip connection.
        """
        x_skip = x

        x = self.conv_x(x)
        g = self.conv_g(g)

        if x.shape[2:] != g.shape[2:]:
            x = F.interpolate(
                x, size=g.shape[2:],
                mode="bilinear", align_corners=True
            )

        x1 = torch.add(x, g)
        x1 = self.relu(x1)
        x1 = self.conv(x1)
        x1 = self.sigmoid(x1)

        x1 = self.up(x1)

        if x_skip.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(
                x1, size=x_skip.shape[2:],
                mode="bilinear", align_corners=True
            )

        x = torch.mul(x_skip, x1)
        return x


class Down_ResUNet(nn.Module):
    """Downsampling block for Residual U-Net.

    Performs max pooling followed by a residual convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x):
        """Forward pass of the downsampling block."""
        return self.maxpool_conv(x)


class Up_ResUNet(nn.Module):
    """Upsampling block for Residual U-Net.

    Upsamples the decoder feature map, concatenates it with the corresponding
    encoder feature map, and applies a residual block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): If True, use bilinear upsampling.
            Otherwise, use transposed convolution. Defaults to True.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = ResidualBlock(
                in_channels, out_channels, in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2,
                kernel_size=2, stride=2
            )
            self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass of the upsampling block.

        Args:
            x1 (torch.Tensor): Decoder feature map to be upsampled.
            x2 (torch.Tensor): Encoder feature map (skip connection).

        Returns:
            torch.Tensor: Fused feature map.
        """
        x1 = self.up(x1)

        # Compute spatial differences for padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution layer.

    Maps feature channels to the desired number of output channels
    using a 1×1 convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

    def forward(self, x):
        """Forward pass of the output layer."""
        return self.conv(x)
