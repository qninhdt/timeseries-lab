import torch
import torch.nn as nn

from layers import ResNet1DBackbone


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
            torch.arange(0, d_model, 2)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
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


class CNNTransformer(nn.Module):
    """
    A CNN-Transformer model with ResNet1D backbone for portfolio data processing.

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
        # CNN backbone parameters
        cnn_n_blocks=[2, 2],
        cnn_planes_per_branch=[32, 64],
        cnn_target_planes=[32, 64],
        cnn_num_groups=32,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            d_model (int): Dimension of the model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
            cnn_n_blocks (list): Number of blocks for each CNN layer
            cnn_planes_per_branch (list): Number of planes per branch for each CNN layer
            cnn_target_planes (list): Target output planes for each CNN layer
            cnn_num_groups (int): Number of groups for GroupNorm
        """
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        # CNN backbone
        self.cnn_backbone = ResNet1DBackbone(
            n_features=n_features,
            n_blocks=cnn_n_blocks,
            planes_per_branch=cnn_planes_per_branch,
            target_planes=cnn_target_planes,
            num_groups=cnn_num_groups,
        )

        # Project CNN output to d_model dimension
        cnn_output_dim = self.cnn_backbone.out_channels
        self.cnn_to_transformer = nn.Linear(cnn_output_dim, d_model)

        # CLS token - learnable embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

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

        # Pass through CNN backbone
        # Input: (B * P, T, D) -> Output: (B * P, T', cnn_output_dim)
        # where T' may be smaller due to striding
        x_cnn = self.cnn_backbone(x_flat)  # (B * P, T', cnn_output_dim)

        # Project to d_model dimension
        # output shape: (B * P, T', d_model)
        x_proj = self.cnn_to_transformer(x_cnn)

        # Add positional encoding
        # output shape: (B * P, T', d_model)
        x_pos = self.pos_encoder(x_proj)

        # Add CLS token at the beginning of the sequence
        # cls_token shape: (1, 1, d_model) -> (B * P, 1, d_model)
        cls_tokens = self.cls_token.expand(B * P, -1, -1)
        # Concatenate: output shape: (B * P, T' + 1, d_model)
        x_with_cls = torch.cat([cls_tokens, x_pos], dim=1)

        # Pass through transformer encoder
        # output shape: (B * P, T' + 1, d_model)
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

