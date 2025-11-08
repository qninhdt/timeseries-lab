import torch
import torch.nn as nn
from typing import Optional


class LSTM(nn.Module):
    """
    LSTM model dự đoán 1 giá trị beta với tanh activation.
    Beta trong khoảng [-1, 1]: -1 = sell, 0 = hold, 1 = buy
    """

    def __init__(
        self,
        n_features_1h: int,
        n_features_4h: int = 0,  # Không sử dụng nhưng cần để tương thích với interface
        n_features_1d: int = 0,  # Không sử dụng nhưng cần để tương thích với interface
        max_coins: int = 1,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        """
        Args:
            n_features_1h: Số lượng features cho timeframe 1h
            n_features_4h: Không sử dụng (để tương thích)
            n_features_1d: Không sử dụng (để tương thích)
            max_coins: Số lượng coins tối đa (không sử dụng trong model này)
            hidden_size: Kích thước hidden state của LSTM
            num_layers: Số lớp LSTM
            dropout: Dropout rate
            bidirectional: Có sử dụng bidirectional LSTM hay không
        """
        super().__init__()

        self.n_features_1h = n_features_1h
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=n_features_1h,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Tính số chiều output của LSTM
        lstm_output_size = hidden_size * (2 if bidirectional else 1)

        # Regression head: 1 output with tanh
        self.regressor = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),  # Single output
            nn.Tanh(),  # Squash to [-1, 1]
        )

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
        lstm_out, _ = self.lstm(x_1h)  # (B, T_1h, hidden_size * num_directions)

        # Lấy output tại timestep cuối cùng
        last_hidden = lstm_out[:, -1, :]  # (B, hidden_size * num_directions)

        # Regression with tanh
        beta = self.regressor(last_hidden)  # (B, 1)

        return beta
