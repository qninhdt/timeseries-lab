import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """Standard positional embedding for transformers."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class DataEmbedding_inverted(nn.Module):
    """
    Inverted embedding: projects time dimension to d_model.
    Used in iTransformer where each variate is treated as a token.
    """

    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: [Batch, Time, Variate] -> permute to [Batch, Variate, Time]
        x = x.permute(0, 2, 1)
        # Project time dimension to d_model: [Batch, Variate, d_model]
        x = self.value_embedding(x)
        return self.dropout(x)


class FullAttention(nn.Module):
    """Standard scaled dot-product attention."""

    def __init__(
        self,
        mask_flag: bool = True,
        scale=None,
        attention_dropout: float = 0.1,
        output_attention: bool = False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag and attn_mask is not None:
            scores.masked_fill_(attn_mask.mask, -float("inf"))

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    """Multi-head attention layer with projections."""

    def __init__(
        self,
        attention,
        d_model: int,
        n_heads: int,
        d_keys: int = None,
        d_values: int = None,
    ):
        super().__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class EncoderLayer(nn.Module):
    """Transformer encoder layer with attention and feed-forward network."""

    def __init__(
        self,
        attention,
        d_model: int,
        d_ff: int = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, attn_layers, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class iTransformer(nn.Module):
    """
    iTransformer adapted for portfolio classification task.
    Paper: https://arxiv.org/abs/2310.06625

    Key idea: Treats each variate (feature) as a token instead of each time step.
    This is the "inverted" approach - attention operates on the variate dimension.

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
        seq_len: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = False,
        max_coins: int = 128,
    ):
        """
        Args:
            n_features (int): Number of features (D) - number of variates
            seq_len (int): Sequence length (T) - lookback window
            d_model (int): Dimension of the model
            n_heads (int): Number of attention heads
            e_layers (int): Number of encoder layers
            d_ff (int): Dimension of feedforward network
            dropout (float): Dropout rate
            activation (str): Activation function ('relu' or 'gelu')
            use_norm (bool): Whether to use normalization (Non-stationary Transformer)
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.use_norm = use_norm

        # Inverted embedding: projects time dimension to d_model
        # Each feature becomes a token with d_model dimensions
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout)

        # Encoder: attention operates on variates (features)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # Classification head
        # For each variate, project d_model to 1 (logit)
        self.act = F.gelu
        self.dropout_head = nn.Dropout(dropout)
        # Project from d_model * n_features to 1 (single logit per sample)
        self.projection = nn.Linear(d_model * n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.

        Args:
            x: Input tensor of shape (B, P, T, D)

        Returns:
            logits: Output tensor of shape (B, P, 1)
        """
        # x shape: (B, P, T, D)
        B, P, T, D = x.shape

        # Flatten B and P into a single batch dimension
        # New shape: (B * P, T, D)
        x_flat = x.reshape(B * P, T, D)

        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_flat.mean(1, keepdim=True).detach()
            x_flat = x_flat - means
            stdev = torch.sqrt(
                torch.var(x_flat, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_flat /= stdev

        # Embedding: (B * P, T, D) -> (B * P, D, d_model)
        # Each of the D features becomes a token
        enc_out = self.enc_embedding(x_flat)

        # Encoder: (B * P, D, d_model) -> (B * P, D, d_model)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Activation and dropout
        output = self.act(enc_out)  # (B * P, D, d_model)
        output = self.dropout_head(output)

        # Flatten to (B * P, D * d_model)
        output = output.reshape(output.shape[0], -1)

        # Project to single logit: (B * P, D * d_model) -> (B * P, 1)
        output = self.projection(output)

        # Reshape back to (B, P, 1)
        logits_out = output.reshape(B, P, 1)

        return logits_out
