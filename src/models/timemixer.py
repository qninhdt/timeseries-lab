import torch
import torch.nn as nn
import torch.nn.functional as F


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class DFT_series_decomp(nn.Module):
    """
    DFT-based series decomposition block
    """

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(
        self, seq_len: int, down_sampling_window: int, down_sampling_layers: int
    ):
        super().__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(
        self, seq_len: int, down_sampling_window: int, down_sampling_layers: int
    ):
        super().__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window**i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window**i),
                        seq_len // (down_sampling_window**i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """
    Past Decomposable Mixing block
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        d_ff: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        decomp_method: str,
        moving_avg: int,
        top_k: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if decomp_method == "moving_avg":
            self.decomposition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomposition = DFT_series_decomp(top_k)
        else:
            raise ValueError(f"Unknown decomposition method: {decomp_method}")

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        # Mixing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Normalize(nn.Module):
    """Normalization layer with optional affine parameters."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Unknown mode: {mode}")
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev + self.mean
        return x


class TimeMixer(nn.Module):
    """
    TimeMixer adapted for portfolio classification task.
    Paper: https://openreview.net/forum?id=7oLshfEIC2

    Key idea: Decomposes time series into seasonal and trend components,
    then mixes them at multiple scales for better representation learning.

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
        d_model: int = 16,
        d_ff: int = 32,
        e_layers: int = 2,
        down_sampling_layers: int = 3,
        down_sampling_window: int = 2,
        down_sampling_method: str = "avg",
        decomp_method: str = "moving_avg",
        moving_avg: int = 25,
        top_k: int = 5,
        dropout: float = 0.1,
        use_norm: bool = True,
        max_coins: int = 128,
    ):
        """
        Args:
            n_features (int): Number of features (D)
            seq_len (int): Sequence length (T) - lookback window
            d_model (int): Dimension of the model
            d_ff (int): Dimension of feedforward network
            e_layers (int): Number of encoder layers (PDM blocks)
            down_sampling_layers (int): Number of downsampling layers
            down_sampling_window (int): Window size for downsampling
            down_sampling_method (str): Downsampling method ('avg', 'max', 'conv')
            decomp_method (str): Decomposition method ('moving_avg', 'dft_decomp')
            moving_avg (int): Kernel size for moving average decomposition
            top_k (int): Top-K frequencies for DFT decomposition
            dropout (float): Dropout rate
            use_norm (bool): Whether to use normalization
        """
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.down_sampling_layers_count = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.use_norm = use_norm

        # Embedding layer - simple linear projection
        self.enc_embedding = nn.Linear(n_features, d_model)

        # Past Decomposable Mixing blocks
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    seq_len=seq_len,
                    d_model=d_model,
                    d_ff=d_ff,
                    down_sampling_window=down_sampling_window,
                    down_sampling_layers=down_sampling_layers,
                    decomp_method=decomp_method,
                    moving_avg=moving_avg,
                    top_k=top_k,
                    dropout=dropout,
                )
                for _ in range(e_layers)
            ]
        )

        # Normalization layers for each scale
        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(n_features, affine=True)
                for _ in range(down_sampling_layers + 1)
            ]
        )

        # Classification head
        self.act = F.gelu
        self.dropout_head = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, 1)

    def _multi_scale_process_inputs(self, x_enc):
        """
        Create multi-scale representations by downsampling.
        """
        if self.down_sampling_method == "max":
            down_pool = torch.nn.MaxPool1d(
                self.down_sampling_window, return_indices=False
            )
        elif self.down_sampling_method == "avg":
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            padding = 1 if torch.__version__ >= "1.5.0" else 2
            down_pool = nn.Conv1d(
                in_channels=self.n_features,
                out_channels=self.n_features,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
            )
        else:
            return [x_enc]

        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))

        for i in range(self.down_sampling_layers_count):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

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

        # Multi-scale processing
        x_enc = self._multi_scale_process_inputs(x_flat)

        # Normalize and embed each scale
        x_list = []
        for i, x_scale in enumerate(x_enc):
            # Normalize
            if self.use_norm:
                x_scale = self.normalize_layers[i](x_scale, "norm")

            # Embed: (B * P, T_i, D) -> (B * P, T_i, d_model)
            x_embedded = self.enc_embedding(x_scale)
            x_list.append(x_embedded)

        # Pass through PDM blocks
        for pdm_block in self.pdm_blocks:
            x_list = pdm_block(x_list)

        # Use the finest scale output for classification
        enc_out = x_list[0]  # (B * P, T, d_model)

        # Activation and dropout
        output = self.act(enc_out)
        output = self.dropout_head(output)

        # Flatten: (B * P, T, d_model) -> (B * P, T * d_model)
        output = output.reshape(output.shape[0], -1)

        # Project to single logit: (B * P, T * d_model) -> (B * P, 1)
        output = self.projection(output)

        # Reshape back to (B, P, 1)
        logits_out = output.reshape(B, P, 1)

        return logits_out
