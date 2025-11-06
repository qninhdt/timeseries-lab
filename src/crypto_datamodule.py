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
) -> np.ndarray:
    """
    Numba-accelerated triple-barrier labeling.
    Returns multiclass labels: 0=hold, 1=buy, 2=sell
    """
    n = len(close_prices)
    label_multi = np.zeros(n, dtype=np.int64)

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
                label_multi[i] = 1  # buy
                break
            elif close_prices[j] <= lower_barrier:
                label_multi[i] = 2  # sell
                break
    return label_multi


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
        barrier_atr_multiplier: float = 2.0,
        barrier_horizon: int = 4,
        batch_size: int = 4,
        num_workers: int = 4,
        max_coins: int = -1,
        validation_coins: Optional[List[str]] = None,
        val_exchange: str = "binance",
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
            "barrier_atr_multiplier",
            "barrier_horizon",
            "batch_size",
            "num_workers",
            "max_coins",
            "validation_coins",
            "val_exchange",
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
        self.norm_types: Dict[str, List[int]] = {}
        self.close_idx_1h: int = -1

        # --- Internal State ---
        self.coins: List[str] = []
        self.coin_source_exchange: Dict[str, str] = {}
        self.master_timestamps: np.ndarray = None
        self.date_to_idx_map: Dict[pd.Timestamp, int] = {}
        self.train_indices: np.ndarray = None
        self.val_indices: np.ndarray = None
        self.train_pairs: np.ndarray = None
        self.val_pairs: np.ndarray = None
        self.coin_baselines: Dict[str, float] = {}

        # --- Optimized Data Storage ---
        self.all_labels_trade_aligned: np.ndarray = None
        self.all_labels_dir_aligned: np.ndarray = None
        # <<< MỚI: (giữ) Mảng lưu trữ nhãn 3-class (0, 1, 2)
        # Lưu ý: pipeline huấn luyện dùng nhị phân trade/dir; multi có thể giữ cho mục đích khác
        self.all_labels_multi_aligned: np.ndarray = None
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
        # ... (Không thay đổi)
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        requested_val_coins = self.hparams.validation_coins or []
        requested_val_coins = list(dict.fromkeys(requested_val_coins))
        coins_all = list(metadata["coins"].keys())
        val_existing = [c for c in requested_val_coins if c in coins_all]
        val_missing = [c for c in requested_val_coins if c not in coins_all]

        rest = [c for c in coins_all if c not in set(val_existing)]
        rest_sorted = sorted(
            rest, key=lambda c: metadata["coins"][c]["duration_days"], reverse=True
        )

        ordered = val_existing + rest_sorted
        if self.hparams.max_coins > 0:
            self.coins = ordered[: self.hparams.max_coins]
        else:
            self.coins = ordered
        n_coins = len(self.coins)
        print(f"Loaded and limited to {n_coins} coins.")

        self.coin_source_exchange = {
            coin: metadata["coins"][coin].get("source_exchange", "")
            for coin in self.coins
        }

        # --- 1. Create Master Timestamps ---
        # ... (Không thay đổi)
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
        # <<< MỚI: Khởi tạo mảng cho nhãn 3-class (kiểu int, default=0 (hold))
        self.all_labels_multi_aligned = np.full(
            (n_coins, n_timestamps), 0, dtype=np.int64
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
        # ... (Không thay đổi)
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

        # ... (Logic báo cáo validation coins không đổi)
        requested_val = list(dict.fromkeys(self.hparams.validation_coins or []))
        if requested_val and self.val_pairs is not None and len(self.val_pairs) > 0:
            selected_coin_indices = sorted(set(self.val_pairs[:, 0].tolist()))
            selected_coin_names = [self.coins[i] for i in selected_coin_indices]
            included = [c for c in requested_val if c in set(selected_coin_names)]
            excluded = [c for c in requested_val if c not in set(selected_coin_names)]
            print(f"Validation coins used ({len(included)}): {included}")
            if excluded:
                print(f"Validation coins excluded ({len(excluded)}): {excluded}")

        # <<< MỚI: Log label distribution cho train và validation
        self._log_label_distribution()

        # <<< QUAN TRỌNG: Tính toán baselines (dựa trên nhãn trade nhị phân)
        # Hàm này vẫn quan trọng vì TraderLitModule.setup() sử dụng nó
        # để tính 'coin_pos_weights' (ngay cả khi loss 3-class không dùng chúng).
        self._calculate_coin_baselines()

        print(f"--- Setup complete ---")
        print(f"  Features 1h: {self.n_features['1h']}")
        print(f"  Features 4h: {self.n_features['4h']}")
        print(f"  Features 1d: {self.n_features['1d']}")
        print(f"  Close_1h index: {self.close_idx_1h}")
        print(
            f"  Total valid train samples: {len(self.train_pairs) if self.train_pairs is not None else 0:,}"
        )
        print(
            f"  Total valid val samples: {len(self.val_pairs) if self.val_pairs is not None else 0:,}"
        )

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
        # (Không thay đổi)
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
            "macd",
            "macd_signal",
        ]
        own_stat_features = ["volume", "obv"]
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
        # (Không thay đổi)
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

            # Tính nhãn 3-class trực tiếp bằng numba
            label_multi = _calculate_labels_numba(
                close_p,
                cols["atr"],
                self.hparams.barrier_atr_multiplier,
                self.hparams.barrier_horizon,
            )
            cols["label_multi"] = label_multi
            # Suy ra nhị phân từ 3-class (phục vụ baseline & metrics)
            cols["label_trade"] = label_multi != 0
            cols["label_dir"] = label_multi == 1

        # --- Tạo feature_names và norm_types ---
        feature_names = [
            k
            for k in cols.keys()
            # <<< MỚI: Thêm 'label_multi' vào danh sách loại trừ
            if k not in ["date", "label_trade", "label_dir", "label_multi"]
        ]

        # Logic tạo norm_types (Không thay đổi)
        norm_types = []
        for name in feature_names:
            if name in no_norm_features:
                norm_types.append(0)
            elif name in price_feature_names:
                norm_types.append(2)
            elif name in own_stat_features:
                norm_types.append(1)
            else:
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
        (
            df_1h,
            features_1h,
            feature_names_1h,
            norm_types_1h,
        ) = self._calculate_features(df_raw, "1h")

        # <<< THAY ĐỔI: Trả về cả 3 loại nhãn
        results["1h"] = (
            df_1h.index.values,  # dates
            features_1h,
            df_1h["label_trade"].values.astype(np.bool_),  # Cho baseline
            df_1h["label_dir"].values.astype(np.bool_),
            df_1h["label_multi"].values.astype(np.int64),  # Nhãn 3-class
            feature_names_1h,
            norm_types_1h,
        )

        # 2. Resample và xử lý 4H
        # (Không thay đổi)
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
                norm_types_4h,
            ) = self._calculate_features(df_4h_raw, "4h")
            results["4h"] = (
                df_4h.index.values,
                features_4h,
                feature_names_4h,
                norm_types_4h,
            )

        # 3. Resample và xử lý 1D
        # (Không thay đổi)
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
                norm_types_1d,
            ) = self._calculate_features(df_1d_raw, "1d")
            results["1d"] = (
                df_1d.index.values,
                features_1d,
                feature_names_1d,
                norm_types_1d,
            )

        return results

    def _process_and_fill_coin(
        self, coin_info: Tuple[int, str], is_first_coin: bool = False
    ):
        """
        Hàm "pipeline" tuần tự chính cho một coin.
        """
        coin_idx, coin = coin_info

        df_raw = self._load_single_coin_data(coin)
        if df_raw is None:
            return

        results = self._process_coin_numpy_optimized(df_raw)

        # --- Xử lý 1H ---
        if "1h" not in results:
            return

        # <<< THAY ĐỔI: Unpack 3 loại nhãn
        (
            dates_1h,
            features_array_1h,
            labels_trade_array,  # Nhãn nhị phân
            labels_dir_array,
            labels_multi_array,  # Nhãn 3-class
            feature_names_1h,
            norm_types_1h,
        ) = results["1h"]

        # Logic is_first_coin (Không thay đổi)
        if is_first_coin:
            self.feature_names["1h"] = feature_names_1h
            self.n_features["1h"] = len(feature_names_1h)
            self.norm_types["1h"] = norm_types_1h
            try:
                self.close_idx_1h = feature_names_1h.index("close")
            except ValueError:
                raise ValueError("Feature 'close' not found in 1h feature list.")

        # Logic căn chỉnh master_indices (Không thay đổi)
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

        # <<< THAY ĐỔI: Lưu trữ TẤT CẢ các nhãn
        # Lưu nhãn nhị phân (để tính baseline)
        self.all_labels_trade_aligned[coin_idx, master_indices_1h] = labels_trade_array[
            df_indices_1h
        ]
        self.all_labels_dir_aligned[coin_idx, master_indices_1h] = labels_dir_array[
            df_indices_1h
        ]
        # Lưu nhãn 3-class (cho model training)
        self.all_labels_multi_aligned[coin_idx, master_indices_1h] = labels_multi_array[
            df_indices_1h
        ]

        self._mask_1h_raw[coin_idx, master_indices_1h] = True
        self.features_per_coin["1h"][coin_idx] = valid_features_1h
        self.master_to_local_map_aligned["1h"][
            coin_idx, master_indices_1h
        ] = local_indices_for_map_1h

        # --- Xử lý 4H (Logic căn chỉnh mới) ---
        # (Không thay đổi)
        if "4h" in results:
            (
                dates_4h,
                features_array_4h,
                feature_names_4h,
                norm_types_4h,
            ) = results["4h"]
            if is_first_coin:
                self.feature_names["4h"] = feature_names_4h
                self.n_features["4h"] = len(feature_names_4h)
                self.norm_types["4h"] = norm_types_4h
            df_4h_local = pd.DataFrame(
                {
                    "date": dates_4h,
                    "local_idx_4h": np.arange(len(dates_4h), dtype=np.int32),
                }
            )
            aligned_4h = pd.merge_asof(
                self.master_df, df_4h_local, on="date", direction="backward"
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
        # (Không thay đổi)
        if "1d" in results:
            (
                dates_1d,
                features_array_1d,
                feature_names_1d,
                norm_types_1d,
            ) = results["1d"]
            if is_first_coin:
                self.feature_names["1d"] = feature_names_1d
                self.n_features["1d"] = len(feature_names_1d)
                self.norm_types["1d"] = norm_types_1d
            df_1d_local = pd.DataFrame(
                {
                    "date": dates_1d,
                    "local_idx_1d": np.arange(len(dates_1d), dtype=np.int32),
                }
            )
            aligned_1d = pd.merge_asof(
                self.master_df, df_1d_local, on="date", direction="backward"
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
        (Không thay đổi)
        """
        print("Calculating final validity mask across 1h, 4h, 1d...")
        T_1h = self.hparams.lookback_window_1h
        mask_1h_raw = self._mask_1h_raw
        history_counts_1h = bn.move_sum(
            mask_1h_raw.astype(np.int8), window=T_1h, axis=1, min_count=T_1h
        )
        valid_lookback_1h = history_counts_1h == T_1h
        final_mask_1h = mask_1h_raw & valid_lookback_1h
        T_4h = self.hparams.lookback_window_4h
        map_4h = self.master_to_local_map_aligned["4h"]
        final_mask_4h = map_4h >= (T_4h - 1)
        T_1d = self.hparams.lookback_window_1d
        map_1d = self.master_to_local_map_aligned["1d"]
        final_mask_1d = map_1d >= (T_1d - 1)
        self.mask_aligned = final_mask_1h & final_mask_4h & final_mask_1d
        del self._mask_1h_raw
        gc.collect()

    def _log_label_distribution(self):
        """
        Log phân phối nhãn 3-class (hold/buy/sell) cho train và validation sets.
        """
        print("\n=== Label Distribution ===")

        # Train distribution
        if self.train_pairs is not None and len(self.train_pairs) > 0:
            train_coin_idxs = self.train_pairs[:, 0]
            train_ts_idxs = self.train_pairs[:, 1]
            train_labels = self.all_labels_multi_aligned[train_coin_idxs, train_ts_idxs]

            train_total = len(train_labels)
            train_hold_count = np.sum(train_labels == 0)
            train_buy_count = np.sum(train_labels == 1)
            train_sell_count = np.sum(train_labels == 2)

            train_hold_pct = (
                (train_hold_count / train_total) * 100.0 if train_total > 0 else 0.0
            )
            train_buy_pct = (
                (train_buy_count / train_total) * 100.0 if train_total > 0 else 0.0
            )
            train_sell_pct = (
                (train_sell_count / train_total) * 100.0 if train_total > 0 else 0.0
            )

            print(f"Train Set:")
            print(f"  Hold: {train_hold_count:,} ({train_hold_pct:.2f}%)")
            print(f"  Buy:  {train_buy_count:,} ({train_buy_pct:.2f}%)")
            print(f"  Sell: {train_sell_count:,} ({train_sell_pct:.2f}%)")
            print(f"  Total: {train_total:,}")
        else:
            print("Train Set: No samples")

        # Validation distribution
        if self.val_pairs is not None and len(self.val_pairs) > 0:
            val_coin_idxs = self.val_pairs[:, 0]
            val_ts_idxs = self.val_pairs[:, 1]
            val_labels = self.all_labels_multi_aligned[val_coin_idxs, val_ts_idxs]

            val_total = len(val_labels)
            val_hold_count = np.sum(val_labels == 0)
            val_buy_count = np.sum(val_labels == 1)
            val_sell_count = np.sum(val_labels == 2)

            val_hold_pct = (
                (val_hold_count / val_total) * 100.0 if val_total > 0 else 0.0
            )
            val_buy_pct = (val_buy_count / val_total) * 100.0 if val_total > 0 else 0.0
            val_sell_pct = (
                (val_sell_count / val_total) * 100.0 if val_total > 0 else 0.0
            )

            print(f"Validation Set:")
            print(f"  Hold: {val_hold_count:,} ({val_hold_pct:.2f}%)")
            print(f"  Buy:  {val_buy_count:,} ({val_buy_pct:.2f}%)")
            print(f"  Sell: {val_sell_count:,} ({val_sell_pct:.2f}%)")
            print(f"  Total: {val_total:,}")
        else:
            print("Validation Set: No samples")

        print("=" * 35)

    def _calculate_coin_baselines(self):
        """
        (Không thay đổi)
        Calculates the positive trade rate (baseline) for each coin using per-coin training pairs.
        Sử dụng 'all_labels_trade_aligned' (nhãn nhị phân).
        """
        print("Calculating coin baselines from training data...")
        n_coins = len(self.coins)
        if self.val_pairs is None and self.train_pairs is None:
            self.coin_baselines = {coin: 0.0 for coin in self.coins}
            return

        positives_per_coin = np.zeros(n_coins, dtype=np.int64)
        totals_per_coin = np.zeros(n_coins, dtype=np.int64)

        if self.train_pairs is not None and len(self.train_pairs) > 0:
            coin_idxs = self.train_pairs[:, 0]
            ts_idxs = self.train_pairs[:, 1]
            # <<< SỬ DỤNG NHÃN NHỊ PHÂN (trade)
            labels = self.all_labels_trade_aligned[coin_idxs, ts_idxs]
            for c, y in zip(coin_idxs, labels):
                totals_per_coin[c] += 1
                if y:
                    positives_per_coin[c] += 1

        baselines = np.divide(
            positives_per_coin,
            totals_per_coin,
            out=np.zeros_like(positives_per_coin, dtype=float),
            where=totals_per_coin > 0,
        )
        self.coin_baselines = {
            self.coins[i]: float(baselines[i]) for i in range(n_coins)
        }

    def _find_samples_and_split(self):
        """
        (Không thay đổi)
        """
        coin_idxs, ts_idxs = np.where(self.mask_aligned)
        if len(ts_idxs) == 0:
            raise ValueError(
                "No valid (coin, timestamp) samples found with required lookbacks."
            )
        val_start_idx = np.searchsorted(
            self.master_timestamps,
            self.validation_start_date.to_datetime64(),
            side="left",
        )
        is_val = ts_idxs >= val_start_idx
        train_pairs = np.stack([coin_idxs[~is_val], ts_idxs[~is_val]], axis=1)
        val_pairs = np.stack([coin_idxs[is_val], ts_idxs[is_val]], axis=1)
        if len(val_pairs) > 0 and self.hparams.validation_coins:
            requested = list(dict.fromkeys(self.hparams.validation_coins))
            coin_to_idx = {c: i for i, c in enumerate(self.coins)}
            allowed_coin_indices = []
            missing = []
            for c in requested:
                if c in coin_to_idx:
                    allowed_coin_indices.append(coin_to_idx[c])
                else:
                    missing.append(c)
            if allowed_coin_indices:
                allowed_set = set(allowed_coin_indices)
                mask = np.array(
                    [pair[0] in allowed_set for pair in val_pairs], dtype=bool
                )
                val_pairs = val_pairs[mask]
            else:
                warnings.warn(
                    "No valid coins from validation_coins found among selected coins. Validation set will be empty.",
                    UserWarning,
                )
        if len(train_pairs) == 0:
            raise ValueError(
                "No training samples found. Adjust `validation_start_date`."
            )
        if len(val_pairs) == 0:
            raise ValueError(
                "No validation samples found. Adjust `validation_start_date`."
            )
        self.train_pairs = train_pairs.astype(np.int32)
        self.val_pairs = val_pairs.astype(np.int32)

    # --- Collate Function & DataLoaders ---

    def _create_collator(self, is_train: bool):
        """
        Factory tạo ra collate_fn cho 3 khung thời gian.
        """
        T_1h = self.hparams.lookback_window_1h
        T_4h = self.hparams.lookback_window_4h
        T_1d = self.hparams.lookback_window_1d
        F_1h = self.n_features["1h"]
        F_4h = self.n_features["4h"]
        F_1d = self.n_features["1d"]

        OFFSETS_1H = np.arange(-T_1h + 1, 1, dtype=np.int32)
        OFFSETS_4H = np.arange(-T_4h + 1, 1, dtype=np.int32)
        OFFSETS_1D = np.arange(-T_1d + 1, 1, dtype=np.int32)

        def collate_fn(batch_pairs: List[Tuple[int, int]]) -> Dict[str, torch.Tensor]:
            B = len(batch_pairs)

            batch_features_1h = np.empty((B, T_1h, F_1h), dtype=np.float32)
            batch_features_4h = np.empty((B, T_4h, F_4h), dtype=np.float32)
            batch_features_1d = np.empty((B, T_1d, F_1d), dtype=np.float32)

            batch_pairs_np = np.asarray(batch_pairs, dtype=np.int32)
            batch_coin_idxs = batch_pairs_np[:, 0]
            batch_ts_idxs = batch_pairs_np[:, 1]

            # <<< THAY ĐỔI: Lấy nhãn 3-class (0, 1, 2)
            labels_multi = self.all_labels_multi_aligned[batch_coin_idxs, batch_ts_idxs]
            coin_ids = batch_coin_idxs

            current_local_idxs_4h = self.master_to_local_map_aligned["4h"][
                batch_coin_idxs, batch_ts_idxs
            ]
            current_local_idxs_1d = self.master_to_local_map_aligned["1d"][
                batch_coin_idxs, batch_ts_idxs
            ]

            # Vòng for trích xuất features (Không thay đổi)
            for i in range(B):
                coin_idx = batch_coin_idxs[i]
                ts_idx = batch_ts_idxs[i]
                master_idx_range_1h = ts_idx + OFFSETS_1H
                local_indices_1h = self.master_to_local_map_aligned["1h"][
                    coin_idx, master_idx_range_1h
                ]
                batch_features_1h[i, :, :] = self.features_per_coin["1h"][coin_idx][
                    local_indices_1h, :
                ]
                local_indices_4h = current_local_idxs_4h[i] + OFFSETS_4H
                batch_features_4h[i, :, :] = self.features_per_coin["4h"][coin_idx][
                    local_indices_4h, :
                ]
                local_indices_1d = current_local_idxs_1d[i] + OFFSETS_1D
                batch_features_1d[i, :, :] = self.features_per_coin["1d"][coin_idx][
                    local_indices_1d, :
                ]

            # 4. Convert to Tensor
            labels_multi = self.all_labels_multi_aligned[batch_coin_idxs, batch_ts_idxs]

            return {
                "features_1h": torch.from_numpy(batch_features_1h),
                "features_4h": torch.from_numpy(batch_features_4h),
                "features_1d": torch.from_numpy(batch_features_1d),
                "labels": torch.from_numpy(labels_multi).long(),
                "coin_ids": torch.from_numpy(coin_ids).long(),
            }

        return collate_fn

    def train_dataloader(self) -> DataLoader:
        # (Không thay đổi)
        dataset = CryptoPortfolioDataset(self.train_pairs)
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
        # (Không thay đổi)
        dataset = CryptoPortfolioDataset(self.val_pairs)
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
