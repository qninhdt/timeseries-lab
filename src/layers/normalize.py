import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class Normalize(nn.Module):
    def __init__(
        self,
        num_features_1h: int,
        num_features_4h: int,
        num_features_1d: int,
        norm_types_1h: List[int],
        norm_types_4h: List[int],
        norm_types_1d: List[int],
        close_idx_1h: int,
        eps: float = 1e-5,
        affine: bool = False,
    ):
        """
        Lớp chuẩn hóa đa khung thời gian (Multi-timeframe normalization) đã được sửa lỗi.

        Quy tắc chuẩn hóa:
        - norm_type 0: Không chuẩn hóa (mean=0, std=1)
        - norm_type 1: Chuẩn hóa bằng mean/std của chính nó (tính riêng cho từng timeframe)
        - norm_type 2: Chuẩn hóa bằng mean/std của 'close_1h'

        :param num_features_1h: Số lượng features 1h
        :param num_features_4h: Số lượng features 4h
        :param num_features_1d: Số lượng features 1d
        :param norm_types_1h: Danh sách loại norm (0, 1, 2) cho features 1h
        :param norm_types_4h: Danh sách loại norm (0, 1, 2) cho features 4h
        :param norm_types_1d: Danh sách loại norm (0, 1, 2) cho features 1d
        :param close_idx_1h: Index của feature 'close' trong 1h
        :param eps: Epsilon để đảm bảo ổn định số học
        :param affine: Bật tham số affine (learnable weight/bias)
        """
        super().__init__()
        self.num_features_1h = num_features_1h
        self.num_features_4h = num_features_4h
        self.num_features_1d = num_features_1d

        self.norm_types_1h = norm_types_1h
        self.norm_types_4h = norm_types_4h
        self.norm_types_1d = norm_types_1d

        self.close_idx_1h = close_idx_1h
        self.eps = eps
        self.affine = affine

        assert len(norm_types_1h) == num_features_1h, "norm_types_1h length mismatch"
        assert len(norm_types_4h) == num_features_4h, "norm_types_4h length mismatch"
        assert len(norm_types_1d) == num_features_1d, "norm_types_1d length mismatch"
        assert close_idx_1h >= 0, "close_idx_1h must be valid"

        # Khởi tạo các tham số affine riêng biệt cho mỗi timeframe
        if self.affine:
            self.affine_weight_1h = nn.Parameter(torch.ones(num_features_1h))
            self.affine_bias_1h = nn.Parameter(torch.zeros(num_features_1h))
            self.affine_weight_4h = nn.Parameter(torch.ones(num_features_4h))
            self.affine_bias_4h = nn.Parameter(torch.zeros(num_features_4h))
            self.affine_weight_1d = nn.Parameter(torch.ones(num_features_1d))
            self.affine_bias_1d = nn.Parameter(torch.zeros(num_features_1d))

        # Statistics storage (sẽ được gán trong _get_statistics)
        self.register_buffer("mean_1h", None, persistent=False)
        self.register_buffer("stdev_1h", None, persistent=False)
        self.register_buffer("mean_4h", None, persistent=False)
        self.register_buffer("stdev_4h", None, persistent=False)
        self.register_buffer("mean_1d", None, persistent=False)
        self.register_buffer("stdev_1d", None, persistent=False)

    def forward(
        self,
        x_1h: torch.Tensor,
        x_4h: torch.Tensor,
        x_1d: torch.Tensor,
        mode: str = "norm",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Chuẩn hóa cả ba khung thời gian.

        Args:
            x_1h: (B, T_1h, D_1h)
            x_4h: (B, T_4h, D_4h)
            x_1d: (B, T_1d, D_1d)
            mode: "norm"

        Returns:
            Tuple of (x_1h_norm, x_4h_norm, x_1d_norm)
        """
        if mode != "norm":
            raise ValueError(f"Only mode='norm' is supported, got {mode}")

        # 1. Tính toán statistics (mean/std) cho batch hiện tại
        self._get_statistics(x_1h, x_4h, x_1d)

        # 2. Áp dụng normalization
        x_1h_norm = self._normalize(x_1h, self.mean_1h, self.stdev_1h, "1h")
        x_4h_norm = self._normalize(x_4h, self.mean_4h, self.stdev_4h, "4h")
        x_1d_norm = self._normalize(x_1d, self.mean_1d, self.stdev_1d, "1d")

        return x_1h_norm, x_4h_norm, x_1d_norm

    def _get_statistics(
        self, x_1h: torch.Tensor, x_4h: torch.Tensor, x_1d: torch.Tensor
    ) -> None:
        """
        Tính toán mean và std cho từng feature dựa trên norm_type.
        Hàm này gán giá trị cho self.mean_1h, self.stdev_1h, ...
        """
        B_1h, T_1h, D_1h = x_1h.shape
        B_4h, T_4h, D_4h = x_4h.shape
        B_1d, T_1d, D_1d = x_1d.shape

        # --- 1. Lấy 1h Close Stats (dùng cho norm_type=2) ---
        # (B, T, 1)
        close_1h_feat = x_1h[:, :, self.close_idx_1h : self.close_idx_1h + 1]
        # (B, 1, 1)
        close_1h_mean = torch.mean(close_1h_feat, dim=1, keepdim=True)
        close_1h_std = torch.sqrt(
            torch.var(close_1h_feat, dim=1, keepdim=True, unbiased=False) + self.eps
        )

        # --- 2. Xử lý 1h Features ---
        mean_1h = torch.empty(B_1h, 1, D_1h, device=x_1h.device, dtype=x_1h.dtype)
        stdev_1h = torch.empty(B_1h, 1, D_1h, device=x_1h.device, dtype=x_1h.dtype)

        for i in range(D_1h):
            nt = self.norm_types_1h[i]
            if nt == 0:
                # Type 0: Không norm (mean=0, std=1)
                mean_1h[:, :, i : i + 1] = 0.0
                stdev_1h[:, :, i : i + 1] = 1.0
            elif nt == 1:
                # Type 1: Dùng stats của chính nó
                feat = x_1h[:, :, i : i + 1]
                mean_1h[:, :, i : i + 1] = torch.mean(feat, dim=1, keepdim=True)
                stdev_1h[:, :, i : i + 1] = torch.sqrt(
                    torch.var(feat, dim=1, keepdim=True, unbiased=False) + self.eps
                )
            elif nt == 2:
                # Type 2: Dùng stats của close_1h
                mean_1h[:, :, i : i + 1] = close_1h_mean
                stdev_1h[:, :, i : i + 1] = close_1h_std

        self.mean_1h = mean_1h
        self.stdev_1h = stdev_1h

        # --- 3. Xử lý 4h Features ---
        mean_4h = torch.empty(B_4h, 1, D_4h, device=x_4h.device, dtype=x_4h.dtype)
        stdev_4h = torch.empty(B_4h, 1, D_4h, device=x_4h.device, dtype=x_4h.dtype)

        for i in range(D_4h):
            nt = self.norm_types_4h[i]
            if nt == 0:
                mean_4h[:, :, i : i + 1] = 0.0
                stdev_4h[:, :, i : i + 1] = 1.0
            elif nt == 1:
                feat = x_4h[:, :, i : i + 1]
                mean_4h[:, :, i : i + 1] = torch.mean(feat, dim=1, keepdim=True)
                stdev_4h[:, :, i : i + 1] = torch.sqrt(
                    torch.var(feat, dim=1, keepdim=True, unbiased=False) + self.eps
                )
            elif nt == 2:
                # Vẫn dùng stats của close_1h
                mean_4h[:, :, i : i + 1] = close_1h_mean
                stdev_4h[:, :, i : i + 1] = close_1h_std

        self.mean_4h = mean_4h
        self.stdev_4h = stdev_4h

        # --- 4. Xử lý 1d Features ---
        mean_1d = torch.empty(B_1d, 1, D_1d, device=x_1d.device, dtype=x_1d.dtype)
        stdev_1d = torch.empty(B_1d, 1, D_1d, device=x_1d.device, dtype=x_1d.dtype)

        for i in range(D_1d):
            nt = self.norm_types_1d[i]
            if nt == 0:
                mean_1d[:, :, i : i + 1] = 0.0
                stdev_1d[:, :, i : i + 1] = 1.0
            elif nt == 1:
                feat = x_1d[:, :, i : i + 1]
                mean_1d[:, :, i : i + 1] = torch.mean(feat, dim=1, keepdim=True)
                stdev_1d[:, :, i : i + 1] = torch.sqrt(
                    torch.var(feat, dim=1, keepdim=True, unbiased=False) + self.eps
                )
            elif nt == 2:
                # Vẫn dùng stats của close_1h
                mean_1d[:, :, i : i + 1] = close_1h_mean
                stdev_1d[:, :, i : i + 1] = close_1h_std

        self.mean_1d = mean_1d
        self.stdev_1d = stdev_1d

    def _normalize(
        self, x: torch.Tensor, mean: torch.Tensor, stdev: torch.Tensor, timeframe: str
    ) -> torch.Tensor:
        """
        Áp dụng (x - mean) / std và affine transform.
        Đối với norm_type=0, mean=0 và stdev=1, nên (x-0)/1 = x (không đổi).
        """
        x_norm = (x - mean) / stdev

        if self.affine:
            D = x.shape[-1]  # Số features thực tế (D_1h, D_4h, hoặc D_1d)
            if timeframe == "1h":
                weight = self.affine_weight_1h
                bias = self.affine_bias_1h
            elif timeframe == "4h":
                weight = self.affine_weight_4h
                bias = self.affine_bias_4h
            elif timeframe == "1d":
                weight = self.affine_weight_1d
                bias = self.affine_bias_1d
            else:
                raise ValueError(f"Unknown timeframe: {timeframe}")

            # Đảm bảo weight/bias khớp với D
            # (Điều này đã được đảm bảo trong __init__)
            x_norm = x_norm * weight[:D] + bias[:D]

        return torch.clamp(x_norm, -10.0, 10.0)
