from torch import Tensor, nn
import torch


class LeNet5(nn.Module):
    """
    LeNet-5 model for MNIST dataset.
    """

    def __init__(
        self,
        num_classes: int = 10,
    ):
        super(LeNet5, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2),  # C1
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S2
            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # C3
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # S4
            nn.Conv2d(16, 120, kernel_size=5, stride=1),  # C5
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(120, 84),  # F6
            nn.Tanh(),
            nn.Linear(84, num_classes),  # Output layer
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LeNet-5 model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the LeNet-5.
        """
        return self.layers(x)
