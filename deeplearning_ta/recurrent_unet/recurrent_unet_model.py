from .recurrent_unet_parts import *


class RecurrentUNet(nn.Module):
    """Recurrent U-Net architecture.

    This model is a U-Net variant where standard convolutional blocks are replaced
    with recurrent convolutional blocks. Recurrent refinement allows feature maps
    to be iteratively updated, improving representation capacity without increasing
    network depth.

    Args:
        n_channels (int): Number of input channels.
        n_classes (int): Number of output channels or classes.
        hidden_channels (int): Base number of feature channels used in the network.
        bilinear (bool, optional): If True, uses bilinear interpolation for
            upsampling. Defaults to False.
    """

    def __init__(self, n_channels, n_classes, hidden_channels, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = RecurrentBlock(n_channels, hidden_channels)
        self.down1 = Down_RecUNet(hidden_channels, hidden_channels * 2)
        self.down2 = Down_RecUNet(hidden_channels * 2, hidden_channels * 4)
        self.down3 = Down_RecUNet(hidden_channels * 4, hidden_channels * 8)

        factor = 2 if bilinear else 1
        self.down4 = Down_RecUNet(hidden_channels * 8, hidden_channels * 16 // factor)

        # Decoder
        self.up1 = Up_RecUNet(hidden_channels * 16, hidden_channels * 8 // factor, bilinear)
        self.up2 = Up_RecUNet(hidden_channels * 8, hidden_channels * 4 // factor, bilinear)
        self.up3 = Up_RecUNet(hidden_channels * 4, hidden_channels * 2 // factor, bilinear)
        self.up4 = Up_RecUNet(hidden_channels * 2, hidden_channels, bilinear)

        # Output layer
        self.outc = OutConv(hidden_channels, n_classes)

    def forward(self, x):
        """Forward pass of the Recurrent U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape
                `(batch_size, n_channels, height, width)`.

        Returns:
            torch.Tensor: Output tensor of shape
            `(batch_size, n_classes, height, width)`.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        """Enable gradient checkpointing to reduce GPU memory usage.

        This replaces modules with checkpointed versions, trading additional
        computation for reduced memory consumption during backpropagation.
        """
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
