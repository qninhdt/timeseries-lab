import torch
import torch.nn as nn

from layers.rms_norm import RMSNorm
from layers.resnet_backbone import ResNet1DBackbone


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True, bias=False
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm_self = RMSNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True, bias=False
        )
        self.dropout_self = nn.Dropout(dropout)
        self.norm_ffn = RMSNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout_ffn = nn.Dropout(dropout)
        self.silu = nn.SiLU(inplace=True)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        x = query
        x_norm = self.norm1(x)
        attn_output, _ = self.cross_attn(x_norm, memory, memory)
        x = x + self.dropout1(attn_output)
        x_norm = self.norm_self(x)
        self_attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self.dropout_self(self_attn_output)
        x_norm = self.norm_ffn(x)
        ff_output = self.linear2(self.dropout(self.silu(self.linear1(x_norm))))
        x = x + self.dropout_ffn(ff_output)
        return x


class StackedCrossAttention(nn.Module):
    def __init__(
        self,
        n_cross_layers: int,
        d_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(d_model, n_head, dim_feedforward, dropout)
                for _ in range(n_cross_layers)
            ]
        )

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            query = layer(query, memory)
        return query


class TransformerEncoderLayerWithRMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        batch_first: bool = True,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first, bias=False
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        is_causal=False,
        **kwargs,
    ):
        x = src
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm,
            x_norm,
            x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
        )
        x = x + self.dropout1(attn_output)
        x_norm = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_output)
        return x


class MultiTFTransformer(nn.Module):
    """
    Multi-timeframe transformer with cross-attention.

    Input shapes:
        h1: (B, T_1h, D_1h) or (B, P, T_1h, D_1h) - 1h timeframe features
        h4: (B, T_4h, D_4h) or (B, P, T_4h, D_4h) - 4h timeframe features
        d1: (B, T_1d, D_1d) or (B, P, T_1d, D_1d) - 1d timeframe features
        coin_ids: (B,) or (B, P) - coin indices

    Output shape:
        (B, n_classes) if P=1 (standard case)
        (B, P, n_classes) if P>1 (portfolio case)
    """

    def __init__(
        self,
        n_features_1h: int,
        n_features_4h: int,
        n_features_1d: int,
        n_classes: int = 3,
        base_dim: int = 32,
        d_model: int = 256,
        n_head: int = 8,
        n_context_encoder_layers: int = 3,
        n_cross_layers: int = 2,
        dim_feedforward: int = 1024,
        max_len: int = 500,
        dropout: float = 0.1,
        context_embed_dim: int = 64,
        max_coins: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.base_dim = base_dim
        self.max_coins = max_coins
        # Each coin has its own cls token: (max_coins, 1, d_model)
        self.cls_token = nn.Parameter(torch.randn(max_coins, 1, d_model))
        # self.context_embed_dim = context_embed_dim
        # self.h1_context_embed = nn.Parameter(torch.randn(1, context_embed_dim))
        # self.h4_context_embed = nn.Parameter(torch.randn(1, context_embed_dim))
        # self.d1_context_embed = nn.Parameter(torch.randn(1, context_embed_dim))
        self.h1_input_proj = nn.Linear(n_features_1h, base_dim)
        self.h4_input_proj = nn.Linear(n_features_4h, base_dim)
        self.d1_input_proj = nn.Linear(n_features_1d, base_dim)
        self.cnn_backbone = ResNet1DBackbone(
            n_features=base_dim,
            n_blocks=[1, 1],
            planes_per_branch=[32, 64],
            target_planes=[32, 64],
            # context_embed_dim=context_embed_dim,
            dropout=dropout,
        )
        self.h1_embed = nn.Linear(self.cnn_backbone.out_channels, d_model)
        self.h4_embed = nn.Linear(self.cnn_backbone.out_channels, d_model)
        self.d1_embed = nn.Linear(self.cnn_backbone.out_channels, d_model)
        self.reduced_max_len = (max_len // 4) + 1
        self.pos_embed = nn.Embedding(self.reduced_max_len, d_model)
        self.register_buffer("positional_ids", torch.arange(self.reduced_max_len))
        self.embed_dropout = nn.Dropout(dropout)

        def _create_encoder_layer():
            return TransformerEncoderLayerWithRMSNorm(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )

        # Separate encoders for each timeframe
        self.h1_encoder = nn.TransformerEncoder(
            _create_encoder_layer(), num_layers=n_context_encoder_layers
        )
        self.h4_encoder = nn.TransformerEncoder(
            _create_encoder_layer(), num_layers=n_context_encoder_layers
        )
        self.d1_encoder = nn.TransformerEncoder(
            _create_encoder_layer(), num_layers=n_context_encoder_layers
        )
        # Separate cross attention for each cross-attention step
        # Note: cross_attn_h1 removed, using h1_reps directly
        self.cross_attn_h4 = StackedCrossAttention(
            n_cross_layers=n_cross_layers,
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.cross_attn_d1 = StackedCrossAttention(
            n_cross_layers=n_cross_layers,
            d_model=d_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def _add_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        pos_ids = self.positional_ids[:seq_len].unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.pos_embed(pos_ids)
        combined_embed = x + pos_embedding
        return self.embed_dropout(combined_embed)

    def forward(
        self,
        h1: torch.Tensor,
        h4: torch.Tensor = None,
        d1: torch.Tensor = None,
        coin_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        if h4 is None or d1 is None:
            raise ValueError("h4 and d1 must be provided for MultiTFTransformer")
        if coin_ids is None:
            raise ValueError(
                "coin_ids must be provided for MultiTFTransformer with coin-specific cls tokens"
            )

        # Handle input shapes: (B, T, D) -> treat as (B, 1, T, D) internally
        if h1.dim() == 3:
            B, T_1h, D_1h = h1.shape
            _, T_4h, D_4h = h4.shape
            _, T_1d, D_1d = d1.shape
            P = 1
            h1 = h1.unsqueeze(1)  # (B, 1, T_1h, D_1h)
            h4 = h4.unsqueeze(1)  # (B, 1, T_4h, D_4h)
            d1 = d1.unsqueeze(1)  # (B, 1, T_1d, D_1d)
            coin_ids = coin_ids.unsqueeze(1)  # (B, 1)
        else:
            B, P, T_1h, D_1h = h1.shape
            _, _, T_4h, D_4h = h4.shape
            _, _, T_1d, D_1d = d1.shape

        h1_flat = h1.reshape(B * P, T_1h, D_1h)
        h4_flat = h4.reshape(B * P, T_4h, D_4h)
        d1_flat = d1.reshape(B * P, T_1d, D_1d)
        batch_size_flat = B * P

        # Get coin-specific cls tokens using coin_ids
        # Flatten coin_ids: (B, P) -> (B*P,)
        coin_ids_flat = coin_ids.reshape(-1)  # (B*P,)
        # Index into cls_token: (B*P,) -> (B*P, 1, d_model)
        cls_tokens = self.cls_token[coin_ids_flat]  # (B*P, 1, d_model)
        h1_base = self.h1_input_proj(h1_flat)
        h4_base = self.h4_input_proj(h4_flat)
        d1_base = self.d1_input_proj(d1_flat)

        # Expand context embeddings for batch
        # h1_context = self.h1_context_embed.expand(batch_size_flat, -1)
        # h4_context = self.h4_context_embed.expand(batch_size_flat, -1)
        # d1_context = self.d1_context_embed.expand(batch_size_flat, -1)

        h1_feats = self.cnn_backbone(h1_base)
        h4_feats = self.cnn_backbone(h4_base)
        d1_feats = self.cnn_backbone(d1_base)
        h1_embed = self.h1_embed(h1_feats)
        h4_embed = self.h4_embed(h4_feats)
        d1_embed = self.d1_embed(d1_feats)
        h1_embed_pos = self._add_positional_encoding(h1_embed)
        h4_embed_pos = self._add_positional_encoding(h4_embed)
        d1_embed_pos = self._add_positional_encoding(d1_embed)

        # Add cls token to h1 before encoding
        h1_with_cls = torch.cat(
            [cls_tokens, h1_embed_pos], dim=1
        )  # (B*P, 1+T_1h, d_model)

        # Encode each timeframe with separate encoders
        h1_reps = self.h1_encoder(h1_with_cls).contiguous()  # (B*P, 1+T_1h, d_model)
        h4_reps = self.h4_encoder(h4_embed_pos).contiguous()
        d1_reps = self.d1_encoder(d1_embed_pos).contiguous()

        # Cross attention: start from h1_reps (with cls_token) directly
        # Bỏ cross_attn_h1, sử dụng h1_reps trực tiếp
        context_h4 = self.cross_attn_h4(query=h1_reps, memory=h4_reps)
        context_d1 = self.cross_attn_d1(query=context_h4, memory=d1_reps)

        # Extract cls token from h1_reps (first position)
        h1_cls = h1_reps[:, 0, :]  # (B*P, d_model)
        context_d1_cls = context_d1[:, 0, :]  # (B*P, d_model)
        fused_reps = torch.cat([context_d1_cls, h1_cls], dim=1)
        logits = self.classifier(fused_reps)  # (B*P, n_classes)
        n_classes = self.classifier[-1].out_features

        # Reshape based on P dimension
        if P == 1:
            # Standard case: return (B, n_classes)
            logits = logits.reshape(B, n_classes)
        else:
            # Portfolio case: return (B, P, n_classes)
            logits = logits.reshape(B, P, n_classes)
        return logits
