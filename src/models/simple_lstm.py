import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    """
    An LSTM model designed for portfolio data processing.

    Input shape: (B, P, T, D)
        B: Batch Size
        P: Portfolio Size (number of coins)
        T: Time (Lookback Window)
        D: Data (Number of features)

    Output shape: (B, P, 1)
        (Logits for binary 'trade' prediction for each coin)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            hidden_size (int): Size of the LSTM hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate (only applied if num_layers > 1)
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Accepts (Batch, Seq, Feature) input
            dropout=dropout if num_layers > 1 else 0,
        )

        # Linear head to output a single logit per coin
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, P, T, D)
        B, P, T, D = x.shape

        # "Flatten" B and P into a single batch dimension
        # New shape: (B * P, T, D)
        x_flat = x.reshape(B * P, T, D)

        # Pass through LSTM
        # output shape: (B * P, T, hidden_size)
        # h_n shape: (num_layers, B * P, hidden_size)
        # c_n shape: (num_layers, B * P, hidden_size)
        # We only care about the final hidden state
        _, (h_n, c_n) = self.lstm(x_flat)

        # Get the hidden state from the *last* layer
        # h_n[-1] shape: (B * P, hidden_size)
        last_hidden_state = h_n[-1]

        # Pass through the head to get logits
        # logits shape: (B * P, 1)
        logits = self.head(last_hidden_state)

        # Reshape back to (B, P, 1)
        logits_out = logits.reshape(B, P, 1)

        return logits_out
