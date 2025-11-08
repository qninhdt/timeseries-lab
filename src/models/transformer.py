import torch
import torch.nn as nn
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (B, T, D)
        Returns:
            Tensor shape (B, T, D) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Transformer model dự đoán 1 giá trị beta với tanh activation.
    Beta trong khoảng [-1, 1]: -1 = sell, 0 = hold, 1 = buy
    """

    def __init__(
        self,
        n_features_1h: int,
        n_features_4h: int = 0,  # Không sử dụng nhưng cần để tương thích với interface
        n_features_1d: int = 0,  # Không sử dụng nhưng cần để tương thích với interface
        max_coins: int = 1,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
    ) -> None:
        """
        Args:
            n_features_1h: Số lượng features cho timeframe 1h
            n_features_4h: Không sử dụng (để tương thích)
            n_features_1d: Không sử dụng (để tương thích)
            max_coins: Số lượng coins tối đa (không sử dụng trong model này)
            d_model: Dimension of model (embedding size)
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super().__init__()

        self.n_features_1h = n_features_1h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # Input projection
        self.input_projection = nn.Linear(n_features_1h, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Regression head: 1 output with tanh
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),  # Single output
            nn.Tanh(),  # Squash to [-1, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: Optional[torch.Tensor] = None,  # Không sử dụng
        x_1d: Optional[torch.Tensor] = None,  # Không sử dụng
        coin_ids: Optional[torch.Tensor] = None,  # Không sử dụng
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass chỉ sử dụng x_1h.

        Args:
            x_1h: Tensor shape (B, T_1h, D_1h) - features từ timeframe 1h
            x_4h: Bỏ qua
            x_1d: Bỏ qua
            coin_ids: Bỏ qua

        Returns:
            beta: Tensor shape (B, 1) - trading signal in [-1, 1]
        """
        # Chỉ sử dụng x_1h
        # x_1h shape: (B, T_1h, D_1h)

        # Project input to d_model dimension
        x = self.input_projection(x_1h)  # (B, T_1h, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)  # (B, T_1h, d_model)

        # Transformer encoding
        transformer_out = self.transformer_encoder(x)  # (B, T_1h, d_model)

        # Take the last timestep output
        last_hidden = transformer_out[:, -1, :]  # (B, d_model)

        # Regression with tanh
        beta = self.regressor(last_hidden)  # (B, 1)

        return beta
