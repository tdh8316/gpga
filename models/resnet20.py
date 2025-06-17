from torch import Tensor, nn


class ResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv_block(x)
        return nn.functional.relu(out + residual)


class ResNet20(nn.Module):
    """ResNet20 optimized for CIFAR-10 (32x32 images)"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
    ):
        super(ResNet20, self).__init__()

        # ResNet20 has n=3, so 3 blocks in each layer group
        self.layers = nn.Sequential(
            # Initial convolution (no maxpooling for CIFAR-10's small 32x32 images)
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Residual blocks
            *self._build_residual_layers(16, 16, num_blocks=3, stride=1),
            *self._build_residual_layers(16, 32, num_blocks=3, stride=2),
            *self._build_residual_layers(32, 64, num_blocks=3, stride=2),
            # Classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def _build_residual_layers(
        self, in_channels, out_channels, num_blocks, stride
    ) -> list[ResidualBlock]:
        """Create a layer with multiple residual blocks using nn.Sequential"""
        layers: list[ResidualBlock] = []

        # First block may have stride > 1 for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))

        return layers

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the ResNet20 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the ResNet20.
        """
        return self.layers(x)
