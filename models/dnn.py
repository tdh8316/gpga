from torch import Tensor, nn


class DNN(nn.Module):
    """
    A simple DNN model.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 512,
        num_layers: int = 3,
        output_size: int = 10,
    ):
        super(DNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            *[
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            * num_layers,
            nn.Linear(hidden_size, output_size),
        )

        self.layers.apply(self._init_weights)

    def _init_weights(self, layer: nn.Module):
        """
        Initialize weights of the layer.

        Args:
            layer (nn.Module): Layer to initialize.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, a=0.1, nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the MLP model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the MLP.
        """
        return self.layers(x)
