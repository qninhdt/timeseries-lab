import torch
import talib
import numpy as np
import pandas as pd
import pandas_ta as pta
import bottleneck as bn
import warnings
import json
import gc

from pathlib import Path
from numba import jit
from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from typing import Optional, Dict, Any, List, Tuple

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# --- Constants ---
warnings.simplefilter("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None

# ========================================================================
# 1. STANDALONE HELPER FUNCTIONS (Numba, Config)
# ========================================================================


@jit(nopython=True)
def _calculate_labels_numba(
    close_prices: np.ndarray,
    atr_values: np.ndarray,
    barrier_atr_multiplier: float,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numba-accelerated triple-barrier labeling. (Không thay đổi)
    """
    n = len(close_prices)
    labels_trade = np.zeros(n, dtype=np.bool_)
    labels_dir = np.zeros(n, dtype=np.bool_)

    for i in range(n - 1):
        entry_price = close_prices[i]
        atr = atr_values[i]
        if np.isnan(atr) or atr <= 0:
            continue

        barrier_distance = atr * barrier_atr_multiplier
        upper_barrier = entry_price + barrier_distance
        lower_barrier = entry_price - barrier_distance
        end_idx = min(i + 1 + horizon, n)

        for j in range(i + 1, end_idx):
            if close_prices[j] >= upper_barrier:
                labels_trade[i] = True
                labels_dir[i] = True
                break
            elif close_prices[j] <= lower_barrier:
                labels_trade[i] = True
                labels_dir[i] = False
                break
    return labels_trade, labels_dir


# ========================================================================
# 2. DATASET (Minimalist)
# ========================================================================


class CryptoPortfolioDataset(Dataset):
    """
    Minimalist Dataset. (Không thay đổi)
    """

    def __init__(self, sample_indices: np.ndarray):
        self.sample_indices = sample_indices

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> int:
        return self.sample_indices[idx]


# ========================================================================
# 3. DATAMODULE (Orchestrator & Data Holder)
# ========================================================================


class CryptoDataModule(LightningDataModule):
    """
    DataModule đã được tái cấu trúc để hỗ trợ đa khung thời gian (MTF).
    """

    def __init__(
        self,
        data_dir: str = "./data/crypto-1200",
        end_date: str = "2025-09-29",
        validation_start_date: str = "2025-06-01",
        candle_length: int = 40000,
        lookback_window_1h: int = 64,
        lookback_window_4h: int = 16,
        lookback_window_1d: int = 7,
        portfolio_size: int = 64,
        barrier_atr_multiplier: float = 2.0,
        barrier_horizon: int = 4,
        batch_size: int = 4,
        num_workers: int = 4,
        max_coins: int = -1,
        validation_coins: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            "data_dir",
            "end_date",
            "validation_start_date",
            "candle_length",
            "lookback_window_1h",
            "lookback_window_4h",
            "lookback_window_1d",
            "portfolio_size",
            "barrier_atr_multiplier",
            "barrier_horizon",
            "batch_size",
            "num_workers",
            "max_coins",
            "validation_coins",
        )

        # --- Paths and Dates ---
        self.data_dir = Path(self.hparams.data_dir)
        self.end_date = pd.to_datetime(self.hparams.end_date, utc=True)
        self.validation_start_date = pd.to_datetime(
            self.hparams.validation_start_date, utc=True
        )

        # --- Feature Config ---
        self.feature_names: Dict[str, List[str]] = {}
        self.n_features: Dict[str, int] = {}
        # <<< MỚI: Thêm norm_types và close_idx_1h
        self.norm_types: Dict[str, List[int]] = {}
        self.close_idx_1h: int = -1

        # --- Internal State ---
        self.coins: List[str] = []
        self.master_timestamps: np.ndarray = None
        self.date_to_idx_map: Dict[pd.Timestamp, int] = {}
        self.train_indices: np.ndarray = None
        self.val_indices: np.ndarray = None
        self.val_coin_indices: np.ndarray = None
        self.coin_baselines: Dict[str, float] = {}

        # --- Optimized Data Storage ---
        self.all_labels_trade_aligned: np.ndarray = None
        self.all_labels_dir_aligned: np.ndarray = None
        self.mask_aligned: np.ndarray = None

        self.features_per_coin: Dict[str, List[np.ndarray]] = {
            "1h": [],
            "4h": [],
            "1d": [],
        }
        self.master_to_local_map_aligned: Dict[str, np.ndarray] = {}

        self._mask_1h_raw: np.ndarray = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Main setup method.
        """
        if self.mask_aligned is not None:
            print("Data already set up.")
            return

        print(f"--- Starting setup (stage: {stage}) ---")

        # --- 0. Load Metadata ---
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        self.coins = sorted(
            metadata["coins"].keys(),
            key=lambda x: metadata["coins"][x]["duration_days"],
            reverse=True,
        )
        if self.hparams.max_coins > 0:
            self.coins = self.coins[: self.hparams.max_coins]
        n_coins = len(self.coins)
        print(f"Loaded and limited to {n_coins} coins.")

        # --- 1. Create Master Timestamps ---
        self.master_timestamps = pd.date_range(
            end=self.end_date,
            periods=self.hparams.candle_length,
            freq="h",
            tz="UTC",
        ).values
        self.date_to_idx_map = {
            date: i for i, date in enumerate(self.master_timestamps)
        }
        n_timestamps = len(self.master_timestamps)

        # --- 2. Initialize Empty Data Holders ---
        self.all_labels_trade_aligned = np.full(
            (n_coins, n_timestamps), False, dtype=np.bool_
        )
        self.all_labels_dir_aligned = np.full(
            (n_coins, n_timestamps), False, dtype=np.bool_
        )
        self._mask_1h_raw = np.full((n_coins, n_timestamps), False, dtype=np.bool_)

        self.features_per_coin = {
            "1h": [None] * n_coins,
            "4h": [None] * n_coins,
            "1d": [None] * n_coins,
        }
        self.master_to_local_map_aligned = {
            "1h": np.full((n_coins, n_timestamps), -1, dtype=np.int32),
            "4h": np.full((n_coins, n_timestamps), -1, dtype=np.int32),
            "1d": np.full((n_coins, n_timestamps), -1, dtype=np.int32),
        }
        print("Initialized empty data arrays for 1h, 4h, 1d.")

        self.master_df = pd.DataFrame(
            {"date": self.master_timestamps, "master_idx": np.arange(n_timestamps)}
        )

        # --- 3. Run sequential processing ---
        print(f"Starting sequential processing for {n_coins} coins...")

        jobs = list(enumerate(self.coins))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(f"Processing {n_coins} coins", total=n_coins)

            for i, coin_info in enumerate(jobs):
                is_first_coin = i == 0
                self._process_and_fill_coin(coin_info, is_first_coin=is_first_coin)
                progress.update(task, advance=1)

        print("Sequential processing complete.")
        del self.master_df
        gc.collect()

        # --- 4. Post-processing (Splitting, Masking) ---
        self._calculate_validity_mask()
        self._find_samples_and_split()
        self._find_validation_coins()
        self._calculate_coin_baselines()

        print(f"--- Setup complete ---")
        print(f"  Features 1h: {self.n_features['1h']}")
        print(f"  Features 4h: {self.n_features['4h']}")
        print(f"  Features 1d: {self.n_features['1d']}")
        print(f"  Close_1h index: {self.close_idx_1h}")
        print(f"  Total valid train samples: {len(self.train_indices):,}")
        print(f"  Total valid val samples: {len(self.val_indices):,}")

    # --- Processing Functions (Called Sequentially) ---

    def _load_single_coin_data(self, coin: str) -> Optional[pd.DataFrame]:
        """
        Loads and filters Parquet data for a single coin. (Không thay đổi)
        """
        filename = f"{coin.lower()}.parquet"
        filepath = self.data_dir / "data" / filename
        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= self.end_date]

        if df.empty:
            return None

        df = df.drop_duplicates(subset=["date"]).sort_values("date")

        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    # <<< MỚI: Cấu hình norm được chuyển vào đây
    def _calculate_features(
        self, df_in: pd.DataFrame, timeframe: str
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[int]]:
        """
        Tính toán các chỉ báo kỹ thuật VÀ loại chuẩn hóa (norm_type) cho chúng.
        Labels (nhãn) CHỈ được tính cho khung 1h.
        """
        df = df_in.copy()
        open_p = df["open"].values
        high_p = df["high"].values
        low_p = df["low"].values
        close_p = df["close"].values
        volume = df["volume"].values

        cols = {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        }

        # --- Định nghĩa loại chuẩn hóa (Normalization) ---
        # Type 0: Không chuẩn hóa (đã chuẩn hóa hoặc không cần)
        # Type 1: Dùng mean/std của chính nó
        # Type 2: Dùng mean/std của close_h1 (áp dụng cho tất cả các feature giá/volume)

        # Đây là các feature dùng mean/std của close_h1 (type 2)
        price_feature_names = [
            "open",
            "high",
            "low",
            "close",
            "sma_50",
            "sma_200",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "atr",
            "sar",
            "macd",  # macd là chênh lệch giá, nên norm theo giá
            "macd_signal",  # Tương tự
        ]

        # Các feature này là (type 1), sẽ dùng mean/std của riêng nó
        own_stat_features = [
            "volume",  # Volume có range quá lớn, nên dùng std của riêng nó
            "obv",  # Tương tự volume
        ]

        # Các feature này là (type 0), không cần chuẩn hóa
        no_norm_features = [
            "temporal_sin",
            "temporal_cos",
            "log_return",
            "rsi",
            "cci",
            "mfi",
            "roc",
            "cmf",
            "candle_range",
            "candle_body_pct",
            "candle_wick_pct",
        ]

        # --- Tính toán Features ---

        # BBands cho tất cả timeframes
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_p, timeperiod=20)
        cols["bb_upper"] = bb_upper.astype(np.float32)
        cols["bb_middle"] = bb_middle.astype(np.float32)
        cols["bb_lower"] = bb_lower.astype(np.float32)

        cols["sma_50"] = talib.SMA(close_p, timeperiod=50).astype(np.float32)
        cols["sma_200"] = talib.SMA(close_p, timeperiod=200).astype(np.float32)

        # --- Chỉ tính các chỉ báo cho 1h (không tính cho h4, d1) ---
        if timeframe == "1h":
            cols["log_return"] = np.log(close_p / (np.roll(close_p, 1) + 1e-8)).astype(
                np.float32
            )
            cols["sar"] = talib.SAR(high_p, low_p).astype(np.float32)

            cols["rsi"] = (talib.RSI(close_p, timeperiod=14) / 50.0).astype(
                np.float32
            ) - 1.0
            cci_raw = talib.CCI(high_p, low_p, close_p, timeperiod=14)
            cols["cci"] = (np.clip(cci_raw, -200, 200) / 200.0).astype(np.float32)
            cols["mfi"] = (
                talib.MFI(high_p, low_p, close_p, volume, timeperiod=14) / 50.0
            ).astype(np.float32) - 1.0
            roc_raw = talib.ROC(close_p, timeperiod=10)
            cols["roc"] = (np.clip(roc_raw, -20, 20) / 20.0).astype(np.float32)

            cols["cmf"] = pta.cmf(
                df["high"], df["low"], df["close"], df["volume"], length=20
            ).values.astype(np.float32)
            macd, macd_signal, _ = talib.MACD(close_p)
            cols["macd"] = macd.astype(np.float32)
            cols["macd_signal"] = macd_signal.astype(np.float32)

            cols["obv"] = talib.OBV(close_p, volume).astype(np.float32)
            cols["atr"] = talib.ATR(high_p, low_p, close_p, timeperiod=14).astype(
                np.float32
            )

            cols["candle_range"] = ((high_p - low_p) / (open_p + 1e-8)).astype(
                np.float32
            )
            cols["candle_body_pct"] = (
                np.abs(close_p - open_p) / (high_p - low_p + 1e-8)
            ).astype(np.float32)
            cols["candle_wick_pct"] = (
                (high_p - np.maximum(open_p, close_p)) / (high_p - low_p + 1e-8)
            ).astype(np.float32)

            cols["temporal_sin"] = np.sin(
                2 * np.pi * df["date"].dt.hour / 24.0
            ).values.astype(np.float32)
            cols["temporal_cos"] = np.cos(
                2 * np.pi * df["date"].dt.hour / 24.0
            ).values.astype(np.float32)

            # Chỉ 1h mới tính labels
            labels_trade, labels_dir = _calculate_labels_numba(
                close_p,
                cols["atr"],
                self.hparams.barrier_atr_multiplier,
                self.hparams.barrier_horizon,
            )
            cols["label_trade"] = labels_trade
            cols["label_dir"] = labels_dir

        # --- Tạo feature_names và norm_types ---
        feature_names = [
            k for k in cols.keys() if k not in ["date", "label_trade", "label_dir"]
        ]

        # <<< MỚI: Tạo danh sách norm_types
        norm_types = []
        for name in feature_names:
            if name in no_norm_features:
                norm_types.append(0)
            elif name in price_feature_names:
                norm_types.append(2)
            elif name in own_stat_features:
                norm_types.append(1)
            else:
                # Mặc định, nếu không được định nghĩa, dùng type 1 (norm riêng)
                # Điều này an toàn hơn là không norm, nhưng cũng có thể gây warning
                warnings.warn(
                    f"Feature '{name}' in timeframe '{timeframe}' not assigned a norm type, defaulting to 1 (own stats)."
                )
                norm_types.append(1)

        df_out = pd.DataFrame(cols, index=df["date"])
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

        feature_df = df_out[feature_names]
        feature_df = feature_df.ffill().bfill().fillna(0)
        features_array = feature_df.values.astype(np.float32)

        if timeframe == "1h":
            return df_out, features_array, feature_names, norm_types
        else:
            return df_out, features_array, feature_names, norm_types

    def _process_coin_numpy_optimized(self, df_raw: pd.DataFrame) -> Dict[str, Tuple]:
        """
        Tính toán đặc trưng và nhãn cho 3 khung thời gian 1h, 4h, 1d.
        """
        results = {}

        # 1. Xử lý 1H (Khung thời gian cơ sở)
        # <<< MỚI: Nhận 4 giá trị trả về
        (
            df_1h,
            features_1h,
            feature_names_1h,
            norm_types_1h,
        ) = self._calculate_features(df_raw, "1h")
        results["1h"] = (
            df_1h.index.values,  # dates
            features_1h,
            df_1h["label_trade"].values.astype(np.bool_),
            df_1h["label_dir"].values.astype(np.bool_),
            feature_names_1h,
            norm_types_1h,  # <<< MỚI
        )

        # 2. Resample và xử lý 4H
        df_raw_indexed = df_raw.set_index("date")
        agg_rules = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }

        df_4h_raw = (
            df_raw_indexed.resample("4h", closed="left", label="right")
            .agg(agg_rules)
            .dropna()
            .reset_index()
        )
        if not df_4h_raw.empty:
            (
                df_4h,
                features_4h,
                feature_names_4h,
                norm_types_4h,  # <<< MỚI
            ) = self._calculate_features(df_4h_raw, "4h")
            results["4h"] = (
                df_4h.index.values,
                features_4h,
                feature_names_4h,
                norm_types_4h,  # <<< MỚI
            )

        # 3. Resample và xử lý 1D
        df_1d_raw = (
            df_raw_indexed.resample("1d", closed="left", label="right")
            .agg(agg_rules)
            .dropna()
            .reset_index()
        )
        if not df_1d_raw.empty:
            (
                df_1d,
                features_1d,
                feature_names_1d,
                norm_types_1d,  # <<< MỚI
            ) = self._calculate_features(df_1d_raw, "1d")
            results["1d"] = (
                df_1d.index.values,
                features_1d,
                feature_names_1d,
                norm_types_1d,  # <<< MỚI
            )

        return results

    def _process_and_fill_coin(
        self, coin_info: Tuple[int, str], is_first_coin: bool = False
    ):
        """
        Hàm "pipeline" tuần tự chính cho một coin.
        Tải, xử lý 3 khung thời gian, và ghi kết quả vào các mảng chung.
        """
        coin_idx, coin = coin_info

        df_raw = self._load_single_coin_data(coin)
        if df_raw is None:
            return

        results = self._process_coin_numpy_optimized(df_raw)

        # --- Xử lý 1H (Giống như cũ) ---
        if "1h" not in results:
            return

        (
            dates_1h,
            features_array_1h,
            labels_trade_array,
            labels_dir_array,
            feature_names_1h,
            norm_types_1h,  # <<< MỚI
        ) = results["1h"]

        # <<< MỚI: Lưu trữ cấu hình norm nếu là coin đầu tiên
        if is_first_coin:
            self.feature_names["1h"] = feature_names_1h
            self.n_features["1h"] = len(feature_names_1h)
            self.norm_types["1h"] = norm_types_1h
            try:
                self.close_idx_1h = feature_names_1h.index("close")
            except ValueError:
                raise ValueError("Feature 'close' not found in 1h feature list.")

        master_indices_1h = []
        df_indices_1h = []
        for i, date in enumerate(dates_1h):
            if date in self.date_to_idx_map:
                master_indices_1h.append(self.date_to_idx_map[date])
                df_indices_1h.append(i)

        if not master_indices_1h:
            return

        valid_features_1h = features_array_1h[df_indices_1h].astype(np.float32)
        local_indices_for_map_1h = np.arange(len(df_indices_1h), dtype=np.int32)

        self.all_labels_trade_aligned[coin_idx, master_indices_1h] = labels_trade_array[
            df_indices_1h
        ]
        self.all_labels_dir_aligned[coin_idx, master_indices_1h] = labels_dir_array[
            df_indices_1h
        ]
        self._mask_1h_raw[coin_idx, master_indices_1h] = True
        self.features_per_coin["1h"][coin_idx] = valid_features_1h
        self.master_to_local_map_aligned["1h"][
            coin_idx, master_indices_1h
        ] = local_indices_for_map_1h

        # --- Xử lý 4H (Logic căn chỉnh mới) ---
        if "4h" in results:
            (
                dates_4h,
                features_array_4h,
                feature_names_4h,
                norm_types_4h,  # <<< MỚI
            ) = results["4h"]
            if is_first_coin:
                self.feature_names["4h"] = feature_names_4h
                self.n_features["4h"] = len(feature_names_4h)
                self.norm_types["4h"] = norm_types_4h  # <<< MỚI

            df_4h_local = pd.DataFrame(
                {
                    "date": dates_4h,
                    "local_idx_4h": np.arange(len(dates_4h), dtype=np.int32),
                }
            )

            aligned_4h = pd.merge_asof(
                self.master_df,
                df_4h_local,
                on="date",
                direction="backward",
            )

            valid_aligned_4h = aligned_4h.dropna()
            master_indices_4h = valid_aligned_4h["master_idx"].values
            local_indices_4h = valid_aligned_4h["local_idx_4h"].values.astype(np.int32)

            self.master_to_local_map_aligned["4h"][
                coin_idx, master_indices_4h
            ] = local_indices_4h
            self.features_per_coin["4h"][coin_idx] = features_array_4h.astype(
                np.float32
            )

        # --- Xử lý 1D (Logic căn chỉnh mới) ---
        if "1d" in results:
            (
                dates_1d,
                features_array_1d,
                feature_names_1d,
                norm_types_1d,  # <<< MỚI
            ) = results["1d"]
            if is_first_coin:
                self.feature_names["1d"] = feature_names_1d
                self.n_features["1d"] = len(feature_names_1d)
                self.norm_types["1d"] = norm_types_1d  # <<< MỚI

            df_1d_local = pd.DataFrame(
                {
                    "date": dates_1d,
                    "local_idx_1d": np.arange(len(dates_1d), dtype=np.int32),
                }
            )

            aligned_1d = pd.merge_asof(
                self.master_df,
                df_1d_local,
                on="date",
                direction="backward",
            )

            valid_aligned_1d = aligned_1d.dropna()
            master_indices_1d = valid_aligned_1d["master_idx"].values
            local_indices_1d = valid_aligned_1d["local_idx_1d"].values.astype(np.int32)

            self.master_to_local_map_aligned["1d"][
                coin_idx, master_indices_1d
            ] = local_indices_1d
            self.features_per_coin["1d"][coin_idx] = features_array_1d.astype(
                np.float32
            )

    # --- Post-Processing Functions ---

    def _calculate_validity_mask(self):
        """
        Cập nhật `mask_aligned` để chỉ bao gồm các mốc thời gian
        có đủ lookback cho CẢ BA khung thời gian. (Không thay đổi)
        """
        print("Calculating final validity mask across 1h, 4h, 1d...")

        # 1. Kiểm tra 1H
        T_1h = self.hparams.lookback_window_1h
        mask_1h_raw = self._mask_1h_raw
        history_counts_1h = bn.move_sum(
            mask_1h_raw.astype(np.int8),
            window=T_1h,
            axis=1,
            min_count=T_1h,
        )
        valid_lookback_1h = history_counts_1h == T_1h
        final_mask_1h = mask_1h_raw & valid_lookback_1h

        # 2. Kiểm tra 4H
        T_4h = self.hparams.lookback_window_4h
        map_4h = self.master_to_local_map_aligned["4h"]
        final_mask_4h = map_4h >= (T_4h - 1)

        # 3. Kiểm tra 1D
        T_1d = self.hparams.lookback_window_1d
        map_1d = self.master_to_local_map_aligned["1d"]
        final_mask_1d = map_1d >= (T_1d - 1)

        # Mask cuối cùng là AND của cả ba
        self.mask_aligned = final_mask_1h & final_mask_4h & final_mask_1d

        del self._mask_1h_raw
        gc.collect()

    def _calculate_coin_baselines(self):
        """Calculates the positive trade rate (baseline) for each coin. (Không thay đổi)"""
        print("Calculating coin baselines from training data...")
        if self.train_indices is None or len(self.train_indices) == 0:
            self.coin_baselines = {coin: 0.0 for coin in self.coins}
            return

        n_coins = len(self.coins)
        n_timestamps = len(self.master_timestamps)

        unique_train_indices = np.unique(self.train_indices)
        train_mask_full = np.zeros(n_timestamps, dtype=bool)
        train_mask_full[unique_train_indices] = True

        valid_train_mask = self.mask_aligned & train_mask_full[None, :]

        n_positive_trades = np.sum(
            self.all_labels_trade_aligned & valid_train_mask, axis=1
        )
        n_total_valid_samples = np.sum(valid_train_mask, axis=1)

        baselines = np.divide(
            n_positive_trades,
            n_total_valid_samples,
            out=np.full(n_coins, np.nan),
            where=n_total_valid_samples > 0,
        )

        self.coin_baselines = {
            self.coins[i]: (baselines[i] if not np.isnan(baselines[i]) else 0.0)
            for i in range(n_coins)
        }

    def _find_samples_and_split(self):
        """
        Finds all valid timestamp indices (>= P coins)
        and splits them into train/val sets. (Không thay đổi logic)
        """
        valid_coins_per_timestamp = np.sum(self.mask_aligned, axis=0)
        valid_sample_mask = valid_coins_per_timestamp >= self.hparams.portfolio_size
        all_valid_sample_indices = np.where(valid_sample_mask)[0]

        if len(all_valid_sample_indices) == 0:
            raise ValueError(
                f"No timestamps found with at least {self.hparams.portfolio_size} valid coins "
                f"(with 1h, 4h, 1d lookbacks)."
            )

        val_start_idx = np.searchsorted(
            self.master_timestamps,
            self.validation_start_date.to_datetime64(),
            side="left",
        )
        val_mask = all_valid_sample_indices >= val_start_idx
        unique_train_indices = all_valid_sample_indices[~val_mask]
        self.val_indices = all_valid_sample_indices[val_mask]

        if len(unique_train_indices) == 0:
            raise ValueError(
                "No training samples found. Adjust `validation_start_date`."
            )
        if len(self.val_indices) == 0:
            raise ValueError(
                "No validation samples found. Adjust `validation_start_date`."
            )

        n_valid_coins_at_train_indices = valid_coins_per_timestamp[unique_train_indices]
        n_repeats = np.ceil(
            n_valid_coins_at_train_indices / self.hparams.portfolio_size
        ).astype(np.intp)
        n_repeats = np.maximum(n_repeats, 1)
        self.train_indices = np.repeat(unique_train_indices, n_repeats)

    # --- Validation Coin Selection ---
    # (Toàn bộ logic này không cần thay đổi)

    def _find_validation_coins(self):
        """Orchestrates the selection of a fixed validation coin portfolio."""
        P = self.hparams.portfolio_size
        user_coins = self.hparams.validation_coins
        if user_coins is not None:
            selected_indices = self._get_user_provided_val_coins(user_coins)
            selected_indices = self._filter_val_coins_by_availability(selected_indices)
            selected_indices = self._fill_or_truncate_val_coins(selected_indices, P)
        else:
            selected_indices = self._get_auto_selected_val_coins(P)
        self.val_coin_indices = np.sort(np.array(selected_indices))
        self.val_coin_names = [self.coins[i] for i in self.val_coin_indices]
        if len(self.val_coin_indices) < P:
            warnings.warn(
                f"Could only find {len(self.val_coin_indices)} coins for validation portfolio (P={P})."
            )
        print(
            f"Final validation coins (P={len(self.val_coin_indices)}): {self.val_coin_names}"
        )

    def _get_user_provided_val_coins(self, user_coins: List[str]) -> List[int]:
        coins_set = set(self.coins)
        coin_to_idx = {coin: idx for idx, coin in enumerate(self.coins)}
        existing_coins = [coin for coin in user_coins if coin in coins_set]
        missing_coins = [coin for coin in user_coins if coin not in coins_set]
        if missing_coins:
            warnings.warn(f"Missing coins, ignored: {missing_coins}", UserWarning)
        selected_indices = []
        seen = set()
        for coin in existing_coins:
            idx = coin_to_idx[coin]
            if idx not in seen:
                seen.add(idx)
                selected_indices.append(idx)
        return selected_indices

    def _filter_val_coins_by_availability(self, coin_indices: List[int]) -> List[int]:
        if len(self.val_indices) == 0:
            return coin_indices
        valid_indices = []
        for coin_idx in coin_indices:
            if np.all(self.mask_aligned[coin_idx, self.val_indices]):
                valid_indices.append(coin_idx)
            else:
                warnings.warn(
                    f"Coin '{self.coins[coin_idx]}' has missing data in val range, ignored.",
                    UserWarning,
                )
        return valid_indices

    def _fill_or_truncate_val_coins(
        self, selected_indices: List[int], P: int
    ) -> List[int]:
        if len(selected_indices) < P:
            candidate_scores = {}
            for coin_idx in range(len(self.coins)):
                if coin_idx in selected_indices:
                    continue
                if len(self.val_indices) > 0:
                    if not np.all(self.mask_aligned[coin_idx, self.val_indices]):
                        continue
                    score = np.sum(self.mask_aligned[coin_idx, self.val_indices])
                else:
                    score = np.sum(self.mask_aligned[coin_idx, :])
                candidate_scores[coin_idx] = score
            sorted_candidates = sorted(
                candidate_scores.items(), key=lambda x: x[1], reverse=True
            )
            for coin_idx, _ in sorted_candidates:
                if len(selected_indices) >= P:
                    break
                selected_indices.append(coin_idx)
        elif len(selected_indices) > P:
            coin_scores = []
            for coin_idx in selected_indices:
                score = (
                    np.sum(self.mask_aligned[coin_idx, self.val_indices])
                    if len(self.val_indices) > 0
                    else np.sum(self.mask_aligned[coin_idx, :])
                )
                coin_scores.append((coin_idx, score))
            coin_scores.sort(key=lambda x: x[1], reverse=True)
            selected_indices = [idx for idx, _ in coin_scores[:P]]
            warnings.warn(f"Too many val coins, kept top {P}", UserWarning)
        return selected_indices

    def _get_auto_selected_val_coins(self, P: int) -> List[int]:
        valid_samples_per_coin = np.sum(self.mask_aligned, axis=1)
        top_p_coin_indices = np.argsort(valid_samples_per_coin)[-P:]
        return list(top_p_coin_indices)

    # --- Collate Function & DataLoaders ---

    def _create_collator(self, is_train: bool):
        """
        Factory tạo ra collate_fn cho 3 khung thời gian. (Không thay đổi)
        """
        T_1h = self.hparams.lookback_window_1h
        T_4h = self.hparams.lookback_window_4h
        T_1d = self.hparams.lookback_window_1d
        F_1h = self.n_features["1h"]
        F_4h = self.n_features["4h"]
        F_1d = self.n_features["1d"]
        P = self.hparams.portfolio_size

        def collate_fn(batch_timestamp_indices: List[int]) -> Dict[str, torch.Tensor]:
            B = len(batch_timestamp_indices)

            batch_features_1h = np.empty((B, P, T_1h, F_1h), dtype=np.float32)
            batch_features_4h = np.empty((B, P, T_4h, F_4h), dtype=np.float32)
            batch_features_1d = np.empty((B, P, T_1d, F_1d), dtype=np.float32)

            list_labels_trade = []
            list_labels_dir = []
            list_coin_ids = []

            for i, ts_idx in enumerate(batch_timestamp_indices):
                if is_train:
                    valid_coin_indices = np.where(self.mask_aligned[:, ts_idx])[0]
                    coin_indices = np.random.choice(
                        valid_coin_indices, P, replace=False
                    )
                    coin_indices.sort()
                else:
                    coin_indices = self.val_coin_indices

                list_labels_trade.append(
                    self.all_labels_trade_aligned[coin_indices, ts_idx]
                )
                list_labels_dir.append(
                    self.all_labels_dir_aligned[coin_indices, ts_idx]
                )
                list_coin_ids.append(coin_indices)

                for p_idx, coin_idx in enumerate(coin_indices):
                    # --- 1. Lấy dữ liệu 1H ---
                    master_idx_range_1h = np.arange(ts_idx - T_1h + 1, ts_idx + 1)
                    local_indices_1h = self.master_to_local_map_aligned["1h"][
                        coin_idx, master_idx_range_1h
                    ]
                    features_slice_1h = self.features_per_coin["1h"][coin_idx][
                        local_indices_1h, :
                    ]
                    batch_features_1h[i, p_idx, :, :] = features_slice_1h

                    # --- 2. Lấy dữ liệu 4H ---
                    current_local_idx_4h = self.master_to_local_map_aligned["4h"][
                        coin_idx, ts_idx
                    ]
                    local_indices_4h = np.arange(
                        current_local_idx_4h - T_4h + 1, current_local_idx_4h + 1
                    )
                    features_slice_4h = self.features_per_coin["4h"][coin_idx][
                        local_indices_4h, :
                    ]
                    batch_features_4h[i, p_idx, :, :] = features_slice_4h

                    # --- 3. Lấy dữ liệu 1D ---
                    current_local_idx_1d = self.master_to_local_map_aligned["1d"][
                        coin_idx, ts_idx
                    ]
                    local_indices_1d = np.arange(
                        current_local_idx_1d - T_1d + 1, current_local_idx_1d + 1
                    )
                    features_slice_1d = self.features_per_coin["1d"][coin_idx][
                        local_indices_1d, :
                    ]
                    batch_features_1d[i, p_idx, :, :] = features_slice_1d

            return {
                "features_1h": torch.from_numpy(batch_features_1h),
                "features_4h": torch.from_numpy(batch_features_4h),
                "features_1d": torch.from_numpy(batch_features_1d),
                "labels_trade": torch.from_numpy(np.stack(list_labels_trade)),
                "labels_dir": torch.from_numpy(np.stack(list_labels_dir)),
                "coin_ids": torch.from_numpy(np.stack(list_coin_ids)).long(),
            }

        return collate_fn

    def train_dataloader(self) -> DataLoader:
        dataset = CryptoPortfolioDataset(self.train_indices)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._create_collator(is_train=True),
        )

    def val_dataloader(self) -> DataLoader:
        dataset = CryptoPortfolioDataset(self.val_indices)
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._create_collator(is_train=False),
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Test dataloader is not implemented.")
