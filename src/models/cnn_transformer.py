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
    A hybrid CNN-Transformer model for multi-timeframe portfolio data processing.

    Architecture:
    1. Input projections for 3 timeframes (1h, 4h, 1d) to base_dim
    2. Shared ResNet1D backbone for feature extraction (CNN)
    3. Projection to d_model dimension
    4. Positional encoding + Timeframe embeddings (like BERT segment embeddings)
    5. Concatenate all timeframes: [h1_tokens, h4_tokens, d1_tokens]
    6. Add special tokens: [CLS, COIN_EMB, ...]
    7. Single shared Transformer encoder for temporal modeling
    8. Classification head using CLS token output

    Input shapes:
        h1: (B, P, T_1h, D_1h) - 1h timeframe features
        h4: (B, P, T_4h, D_4h) - 4h timeframe features
        d1: (B, P, T_1d, D_1d) - 1d timeframe features
        coin_ids: (B, P) - coin indices

    Output shape: (B, P, 1)
        (Logits for binary 'trade' prediction for each coin)
    """

    def __init__(
        self,
        n_features_1h: int,
        n_features_4h: int,
        n_features_1d: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        base_dim: int = 32,
        n_blocks: list = [2, 2],
        planes_per_branch: list = [32, 64],
        target_planes: list = [32, 64],
        num_groups: int = 32,
        max_coins: int = 128,
        **kwargs,
    ):
        """
        Args:
            n_features_1h (int): Number of features for 1h timeframe
            n_features_4h (int): Number of features for 4h timeframe
            n_features_1d (int): Number of features for 1d timeframe
            d_model (int): Dimension of the transformer model
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            dropout (float): Dropout rate
            base_dim (int): Base dimension for input projection before CNN
            n_blocks (list): Number of blocks in each ResNet layer
            planes_per_branch (list): Number of planes per branch in ResNet
            target_planes (list): Target planes for each ResNet layer
            num_groups (int): Number of groups for GroupNorm
            max_coins (int): Maximum number of coins (for coin embedding)
        """
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.base_dim = base_dim

        # Input projections to base_dim (before CNN backbone)
        self.h1_input_proj = nn.Linear(n_features_1h, base_dim)
        self.h4_input_proj = nn.Linear(n_features_4h, base_dim)
        self.d1_input_proj = nn.Linear(n_features_1d, base_dim)

        # Shared CNN backbone (processes base_dim features)
        self.cnn_backbone = ResNet1DBackbone(
            n_features=base_dim,
            n_blocks=n_blocks,
            planes_per_branch=planes_per_branch,
            target_planes=target_planes,
            num_groups=num_groups,
        )

        # Get the output channels from CNN backbone
        cnn_out_channels = self.cnn_backbone.out_channels

        # Project CNN output to d_model dimension
        self.cnn_projection = nn.Linear(cnn_out_channels, d_model)

        # CLS token - learnable embedding (first token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Coin embedding (second token)
        self.coin_embedding = nn.Embedding(max_coins, d_model)

        # Timeframe embeddings (like BERT segment embeddings)
        # 3 timeframes: 1h, 4h, 1d
        self.timeframe_embedding = nn.Embedding(3, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # Transformer encoder (single shared encoder)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_layers
        )

        # Linear head to output a single logit per coin
        self.head = nn.Linear(d_model, 1)

    def forward(
        self,
        h1: torch.Tensor,
        h4: torch.Tensor = None,
        d1: torch.Tensor = None,
        coin_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with multi-timeframe support.

        Args:
            h1: Input tensor of shape (B, P, T_1h, D_1h) - 1h timeframe
            h4: Input tensor of shape (B, P, T_4h, D_4h) - 4h timeframe
            d1: Input tensor of shape (B, P, T_1d, D_1d) - 1d timeframe
            coin_ids: Coin indices of shape (B, P) - required for coin embedding

        Returns:
            logits: Output tensor of shape (B, P, 1)
        """
        if h4 is None or d1 is None:
            raise ValueError("h4 and d1 must be provided for CNNTransformer")
        if coin_ids is None:
            raise ValueError("coin_ids must be provided for CNNTransformer")

        B, P, T_1h, D_1h = h1.shape
        _, _, T_4h, D_4h = h4.shape
        _, _, T_1d, D_1d = d1.shape

        # Flatten B and P into a single batch dimension
        h1_flat = h1.reshape(B * P, T_1h, D_1h)
        h4_flat = h4.reshape(B * P, T_4h, D_4h)
        d1_flat = d1.reshape(B * P, T_1d, D_1d)

        # Project to base_dim before CNN
        h1_base = self.h1_input_proj(h1_flat)  # (B*P, T_1h, base_dim)
        h4_base = self.h4_input_proj(h4_flat)  # (B*P, T_4h, base_dim)
        d1_base = self.d1_input_proj(d1_flat)  # (B*P, T_1d, base_dim)

        # Pass through shared CNN backbone
        h1_cnn = self.cnn_backbone(h1_base)  # (B*P, T_1h', cnn_out_channels)
        h4_cnn = self.cnn_backbone(h4_base)  # (B*P, T_4h', cnn_out_channels)
        d1_cnn = self.cnn_backbone(d1_base)  # (B*P, T_1d', cnn_out_channels)

        # Project CNN output to d_model
        h1_proj = self.cnn_projection(h1_cnn)  # (B*P, T_1h', d_model)
        h4_proj = self.cnn_projection(h4_cnn)  # (B*P, T_4h', d_model)
        d1_proj = self.cnn_projection(d1_cnn)  # (B*P, T_1d', d_model)

        # Add positional encoding to each timeframe
        h1_pos = self.pos_encoder(h1_proj)  # (B*P, T_1h', d_model)
        h4_pos = self.pos_encoder(h4_proj)  # (B*P, T_4h', d_model)
        d1_pos = self.pos_encoder(d1_proj)  # (B*P, T_1d', d_model)

        # Get actual sequence lengths after CNN downsampling
        T_1h_down = h1_pos.shape[1]
        T_4h_down = h4_pos.shape[1]
        T_1d_down = d1_pos.shape[1]

        # Add timeframe embeddings (like BERT segment embeddings)
        # Timeframe IDs: 0=1h, 1=4h, 2=1d
        tf_emb_1h = self.timeframe_embedding(
            torch.zeros(B * P, T_1h_down, dtype=torch.long, device=h1.device)
        )  # (B*P, T_1h', d_model)
        tf_emb_4h = self.timeframe_embedding(
            torch.ones(B * P, T_4h_down, dtype=torch.long, device=h4.device)
        )  # (B*P, T_4h', d_model)
        tf_emb_1d = self.timeframe_embedding(
            torch.full((B * P, T_1d_down), 2, dtype=torch.long, device=d1.device)
        )  # (B*P, T_1d', d_model)

        h1_with_tf = h1_pos + tf_emb_1h
        h4_with_tf = h4_pos + tf_emb_4h
        d1_with_tf = d1_pos + tf_emb_1d

        # Concatenate all timeframes: [h1, h4, d1]
        # Shape: (B*P, T_1h' + T_4h' + T_1d', d_model)
        all_tokens = torch.cat([h1_with_tf, h4_with_tf, d1_with_tf], dim=1)

        # Get coin embeddings
        coin_ids_flat = coin_ids.reshape(B * P)  # (B*P,)
        coin_emb = self.coin_embedding(coin_ids_flat)  # (B*P, d_model)
        coin_token = coin_emb.unsqueeze(1)  # (B*P, 1, d_model)

        # Get CLS token
        cls_tokens = self.cls_token.expand(B * P, -1, -1)  # (B*P, 1, d_model)

        # Concatenate: [CLS, COIN_EMB, h1_tokens, h4_tokens, d1_tokens]
        # Shape: (B*P, 1 + 1 + T_1h' + T_4h' + T_1d', d_model)
        x_with_special = torch.cat([cls_tokens, coin_token, all_tokens], dim=1)

        # Pass through transformer encoder
        x_transformed = self.transformer_encoder(x_with_special)

        # Take the output from the CLS token (first position)
        cls_output = x_transformed[:, 0, :]  # (B*P, d_model)

        # Pass through the head to get logits
        logits = self.head(cls_output)  # (B*P, 1)

        # Reshape back to (B, P, 1)
        logits_out = logits.reshape(B, P, 1)

        return logits_out
