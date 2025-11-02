import torch
import torch.nn as nn

from layers import ResNet1DBackbone


class CNNLSTM(nn.Module):
    """
    A CNN-LSTM model with ResNet1D backbone for portfolio data processing.

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
        # CNN backbone parameters
        cnn_n_blocks=[2, 2],
        cnn_planes_per_branch=[32, 64],
        cnn_target_planes=[32, 64],
        cnn_num_groups=32,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            hidden_size (int): Size of the LSTM hidden state
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate (only applied if num_layers > 1)
            cnn_n_blocks (list): Number of blocks for each CNN layer
            cnn_planes_per_branch (list): Number of planes per branch for each CNN layer
            cnn_target_planes (list): Target output planes for each CNN layer
            cnn_num_groups (int): Number of groups for GroupNorm
        """
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # CNN backbone
        self.cnn_backbone = ResNet1DBackbone(
            n_features=n_features,
            n_blocks=cnn_n_blocks,
            planes_per_branch=cnn_planes_per_branch,
            target_planes=cnn_target_planes,
            num_groups=cnn_num_groups,
        )

        # LSTM layers - use CNN output dimension directly
        cnn_output_dim = self.cnn_backbone.out_channels
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
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

        # Pass through CNN backbone
        # Input: (B * P, T, D) -> Output: (B * P, T', cnn_output_dim)
        # where T' may be smaller due to striding
        x_cnn = self.cnn_backbone(x_flat)  # (B * P, T', cnn_output_dim)

        # Pass through LSTM directly using CNN features
        # output shape: (B * P, T', hidden_size)
        # We only care about the final hidden state
        _, (h_n, c_n) = self.lstm(x_cnn)

        # Get the hidden state from the *last* layer
        # h_n[-1] shape: (B * P, hidden_size)
        last_hidden_state = h_n[-1]

        # Pass through the head to get logits
        # logits shape: (B * P, 1)
        logits = self.head(last_hidden_state)

        # Reshape back to (B, P, 1)
        logits_out = logits.reshape(B, P, 1)

        return logits_out
