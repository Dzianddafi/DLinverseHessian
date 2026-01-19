import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentBlock(nn.Module):
    """Recurrent convolutional block used in Recurrent U-Net.

    This block applies a convolutional layer followed by a recurrent refinement
    process. The same convolutional block is applied multiple times (`t`
    iterations) to iteratively refine features.

    Unlike residual blocks, this block does not include an explicit shortcut
    connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of intermediate channels used in
            the recurrent convolution. Defaults to `out_channels` if None.
        t (int, optional): Number of recurrent refinement iterations.
            Defaults to 2.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, t=2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.t = t
        self.rec_conv_block = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding="same", bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.input_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)

    def forward(self, x):
        """Forward pass for the recurrent block.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, in_channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
            `(batch_size, mid_channels, height, width)`.
        """
        x = self.input_conv(x)

        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0

        x0 = self.rec_conv_block(x)
        for _ in range(self.t):
            x0 = self.rec_conv_block(x + x0)
        x = x0

        return x


class Down_RecUNet(nn.Module):
    """Downsampling block for Recurrent U-Net.

    This block performs spatial downsampling using max pooling followed by
    a `RecurrentBlock`.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            RecurrentBlock(in_channels, out_channels),
        )

    def forward(self, x):
        """Forward pass for the downsampling block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Downsampled feature map.
        """
        x = self.maxpool_conv(x)
        return x


class Up_RecUNet(nn.Module):
    """Upsampling block for Recurrent U-Net.

    This block upsamples the decoder feature map, concatenates it with the
    corresponding encoder feature map (skip connection), and applies a
    `RecurrentBlock`.

    Args:
        in_channels (int): Number of input channels after concatenation.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): If True, uses bilinear interpolation for
            upsampling. Otherwise, uses transposed convolution.
            Defaults to True.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = RecurrentBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = RecurrentBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass for the upsampling block.

        Args:
            x1 (torch.Tensor): Decoder feature map.
            x2 (torch.Tensor): Encoder feature map used as skip connection.

        Returns:
            torch.Tensor: Upsampled and fused feature map.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2,
             diffY // 2, diffY - diffY // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """Final output convolution layer.

    Applies a 1x1 convolution to map feature channels to the desired number
    of output channels.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass for the output layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv(x)
        return x
