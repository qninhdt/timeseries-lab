import torch
import torch.nn as nn


class SimpleTransformer(nn.Module):
    """
    A Transformer model designed for portfolio data processing.

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
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
        """
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # CLS token - learnable embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Input projection to d_model dimension
        self.input_projection = nn.Linear(n_features, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Accepts (Batch, Seq, Feature) input
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Linear head to output a single logit per coin
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, P, T, D)
        B, P, T, D = x.shape

        # "Flatten" B and P into a single batch dimension
        # New shape: (B * P, T, D)
        x_flat = x.reshape(B * P, T, D)

        # Project to d_model dimension
        # output shape: (B * P, T, d_model)
        x_proj = self.input_projection(x_flat)

        # Add positional encoding
        # output shape: (B * P, T, d_model)
        x_pos = self.pos_encoder(x_proj)

        # Add CLS token at the beginning of the sequence
        # cls_token shape: (1, 1, d_model) -> (B * P, 1, d_model)
        cls_tokens = self.cls_token.expand(B * P, -1, -1)
        # Concatenate: output shape: (B * P, T + 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x_pos], dim=1)

        # Pass through transformer encoder
        # output shape: (B * P, T + 1, d_model)
        x_transformed = self.transformer_encoder(x_with_cls)

        # Take the output from the CLS token (first position)
        # output shape: (B * P, d_model)
        cls_output = x_transformed[:, 0, :]

        # Pass through the head to get logits
        # logits shape: (B * P, 1)
        logits = self.head(cls_output)

        # Reshape back to (B, P, 1)
        logits_out = logits.reshape(B, P, 1)

        return logits_out


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Args:
            d_model (int): Dimension of the model
            dropout (float): Dropout rate
            max_len (int): Maximum sequence length
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)
