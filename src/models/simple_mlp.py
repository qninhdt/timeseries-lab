import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    """
    A simple MLP for portfolio time-series data.
    Flattens temporal and feature dimensions into a single vector per coin.

    Input:  (B, P, T, D)
    Output: (B, P, 1)
    """

    def __init__(
        self,
        n_features: int,
        channel_list: list = None,
        dropout: float = 0.1,
        max_coins: int = 128,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            channel_list (list): List of hidden dimensions for each layer, e.g., [512, 256, 128]
            dropout (float): Dropout rate
            max_coins (int): Maximum number of coins (for compatibility)
        """
        super().__init__()
        self.n_features = n_features
        self.channel_list = channel_list if channel_list else [512, 256, 128]
        self.dropout = dropout

        # Build MLP layers based on channel_list
        layers = []

        # First layer: LazyLinear to automatically infer input dimension (T*D)
        layers.append(nn.LazyLinear(self.channel_list[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers based on channel_list
        for i in range(len(self.channel_list) - 1):
            layers.append(nn.Linear(self.channel_list[i], self.channel_list[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(self.channel_list[-1], 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, coin_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, P, T, D) - Batch, Portfolio, Time, Features
            coin_ids: Optional coin indices (not used)

        Returns:
            (B, P, 1) - Logits for each coin
        """
        B, P, T, D = x.shape

        # Flatten B and P: (B*P, T, D)
        x_flat = x.reshape(B * P, T, D)

        # Flatten time and features: (B*P, T*D)
        x_flatten = x_flat.reshape(B * P, -1)

        # Pass through MLP
        logits = self.mlp(x_flatten)  # (B*P, 1)

        # Reshape back to (B, P, 1)
        return logits.view(B, P, 1)
