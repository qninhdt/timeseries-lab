import torch
import torch.nn as nn

from layers.resnet_backbone import ResNet1DBackbone


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


class CNNTransformer(nn.Module):
    """
    A hybrid CNN-Transformer model for portfolio data processing.

    Architecture:
    1. ResNet1D backbone for feature extraction (CNN)
    2. Transformer encoder for temporal modeling
    3. Classification head

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
        n_blocks: list = [2, 2],
        planes_per_branch: list = [32, 64],
        target_planes: list = [32, 64],
        num_groups: int = 32,
        use_coin_embedding: bool = False,
        max_coins: int = 128,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            d_model (int): Dimension of the transformer model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
            n_blocks (list): Number of blocks in each ResNet layer
            planes_per_branch (list): Number of planes per branch in ResNet
            target_planes (list): Target planes for each ResNet layer
            num_groups (int): Number of groups for GroupNorm
            use_coin_embedding (bool): Whether to use coin-specific embeddings
            max_coins (int): Maximum number of coins (for coin embedding)
        """
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.use_coin_embedding = use_coin_embedding

        # CNN backbone (ResNet1D)
        self.cnn_backbone = ResNet1DBackbone(
            n_features=n_features,
            n_blocks=n_blocks,
            planes_per_branch=planes_per_branch,
            target_planes=target_planes,
            num_groups=num_groups,
        )

        # Get the output channels from CNN backbone
        cnn_out_channels = self.cnn_backbone.out_channels

        # CLS token - learnable embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Coin-specific embeddings (optional)
        if use_coin_embedding:
            self.coin_embedding = nn.Embedding(max_coins, d_model)
        else:
            self.coin_embedding = None

        # Project CNN output to d_model dimension
        self.cnn_projection = nn.Linear(cnn_out_channels, d_model)

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

    def forward(self, x: torch.Tensor, coin_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, P, T, D)
            coin_ids: Coin indices of shape (B, P) - optional

        Returns:
            logits: Output tensor of shape (B, P, 1)
        """
        # x shape: (B, P, T, D)
        B, P, T, D = x.shape

        # "Flatten" B and P into a single batch dimension
        # New shape: (B * P, T, D)
        x_flat = x.reshape(B * P, T, D)

        # Pass through CNN backbone for feature extraction
        # Input: (B * P, T, D) -> Output: (B * P, T', D')
        # where T' is the downsampled time dimension
        cnn_features = self.cnn_backbone(x_flat)

        # Project CNN features to d_model dimension
        # output shape: (B * P, T', d_model)
        x_proj = self.cnn_projection(cnn_features)

        # Add coin-specific embedding if enabled
        if self.use_coin_embedding and coin_ids is not None:
            # coin_ids shape: (B, P) -> (B * P,)
            coin_ids_flat = coin_ids.reshape(B * P)
            # Get coin embeddings: (B * P, d_model)
            coin_emb = self.coin_embedding(coin_ids_flat)
            # Add to all timesteps: (B * P, d_model) -> (B * P, 1, d_model) -> broadcast
            x_proj = x_proj + coin_emb.unsqueeze(1)

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
