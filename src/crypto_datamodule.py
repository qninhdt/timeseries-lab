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


# <<< MỚI: FEATURE_CONFIG không còn được dùng làm biến toàn cục
# vì các đặc trưng giờ đây phụ thuộc vào khung thời gian.


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
        # <<< MỚI: Thêm 3 tham số lookback window
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
        # <<< MỚI: Đổi tên lookback_window -> lookback_window_1h
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
        # <<< MỚI: Sẽ được điền động
        self.feature_names: Dict[str, List[str]] = {}
        self.n_features: Dict[str, int] = {}

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
        self.mask_aligned: np.ndarray = (
            None  # Đây sẽ là mask *cuối cùng* sau khi tính toán
        )

        # <<< MỚI: Lưu trữ 3 bộ đặc trưng và 3 bản đồ căn chỉnh
        self.features_per_coin: Dict[str, List[np.ndarray]] = {
            "1h": [],
            "4h": [],
            "1d": [],
        }
        self.master_to_local_map_aligned: Dict[str, np.ndarray] = {}

        # <<< MỚI: Cần một mask 1h tạm thời trước khi tính toán lookback
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
        # <<< MỚI: _mask_1h_raw lưu trữ tính hợp lệ *chỉ* của 1h
        self._mask_1h_raw = np.full((n_coins, n_timestamps), False, dtype=np.bool_)

        # <<< MỚI: Khởi tạo bộ lưu trữ cho 3 khung thời gian
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

        # <<< MỚI: DataFrame chính để căn chỉnh
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
                # <<< MỚI: Cập nhật self.n_features trong lần chạy đầu tiên
                is_first_coin = i == 0
                self._process_and_fill_coin(coin_info, is_first_coin=is_first_coin)
                progress.update(task, advance=1)

        print("Sequential processing complete.")
        del self.master_df  # Không cần nữa, giải phóng bộ nhớ
        gc.collect()

        # --- 4. Post-processing (Splitting, Masking) ---
        self._calculate_validity_mask()  # <<< MỚI: Logic này đã thay đổi hoàn toàn
        self._find_samples_and_split()
        self._find_validation_coins()
        self._calculate_coin_baselines()

        print(f"--- Setup complete ---")
        print(f"  Features 1h: {self.n_features['1h']} | {self.feature_names['1h']}")
        print(f"  Features 4h: {self.n_features['4h']} | {self.feature_names['4h']}")
        print(f"  Features 1d: {self.n_features['1d']} | {self.feature_names['1d']}")
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

        # <<< MỚI: Đảm bảo 'date' là duy nhất và được sắp xếp
        df = df.drop_duplicates(subset=["date"]).sort_values("date")

        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    # <<< MỚI: Hàm trợ giúp để tính toán các đặc trưng
    def _calculate_features(
        self, df_in: pd.DataFrame, timeframe: str
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Tính toán các chỉ báo kỹ thuật cho một DataFrame.
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

        # BBands cho tất cả timeframes (h4, d1 cần)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_p, timeperiod=20)
        cols["bb_upper"] = bb_upper.astype(np.float32)
        cols["bb_middle"] = bb_middle.astype(np.float32)
        cols["bb_lower"] = bb_lower.astype(np.float32)

        # --- Chỉ tính các chỉ báo cho 1h (không tính cho h4, d1) ---
        if timeframe == "1h":
            cols["log_return"] = np.log(close_p / (np.roll(close_p, 1) + 1e-8)).astype(
                np.float32
            )
            cols["sar"] = talib.SAR(high_p, low_p).astype(np.float32)

            # Tạm thời comment các chỉ báo sau
            # cols["adx"] = (talib.ADX(high_p, low_p, close_p, timeperiod=14) / 50.0).astype(
            #     np.float32
            # ) - 1.0
            cols["rsi"] = (talib.RSI(close_p, timeperiod=14) / 50.0).astype(
                np.float32
            ) - 1.0
            # Tạm thời comment stoch
            # stoch_k, stoch_d = talib.STOCH(high_p, low_p, close_p)
            # cols["stoch_k"] = (stoch_k / 50.0).astype(np.float32) - 1.0
            # cols["stoch_d"] = (stoch_d / 50.0).astype(np.float32) - 1.0
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

            # Tạm thời comment ema20, ema50
            # cols["ema_20"] = talib.EMA(close_p, timeperiod=20).astype(np.float32)
            # cols["ema_50"] = talib.EMA(close_p, timeperiod=50).astype(np.float32)
            cols["sma_20"] = talib.SMA(close_p, timeperiod=20).astype(np.float32)
            cols["sma_50"] = talib.SMA(close_p, timeperiod=50).astype(np.float32)
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

            # --- Đặc trưng riêng của 1h ---
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

        feature_names = [
            k for k in cols.keys() if k not in ["date", "label_trade", "label_dir"]
        ]
        df_out = pd.DataFrame(cols, index=df["date"])
        df_out.replace([np.inf, -np.inf], np.nan, inplace=True)

        feature_df = df_out[feature_names]
        feature_df = feature_df.ffill().bfill().fillna(0)
        features_array = feature_df.values.astype(np.float32)

        if timeframe == "1h":
            return df_out, features_array, feature_names
        else:
            return df_out, features_array, feature_names

    # <<< MỚI: Hàm này được viết lại hoàn toàn
    def _process_coin_numpy_optimized(self, df_raw: pd.DataFrame) -> Dict[str, Tuple]:
        """
        Tính toán đặc trưng và nhãn cho 3 khung thời gian 1h, 4h, 1d.
        """
        results = {}

        # 1. Xử lý 1H (Khung thời gian cơ sở)
        df_1h, features_1h, feature_names_1h = self._calculate_features(df_raw, "1h")
        results["1h"] = (
            df_1h.index.values,  # dates
            features_1h,
            df_1h["label_trade"].values.astype(np.bool_),
            df_1h["label_dir"].values.astype(np.bool_),
            feature_names_1h,
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
            df_4h, features_4h, feature_names_4h = self._calculate_features(
                df_4h_raw, "4h"
            )
            results["4h"] = (df_4h.index.values, features_4h, feature_names_4h)  # dates

        # 3. Resample và xử lý 1D
        df_1d_raw = (
            df_raw_indexed.resample("1d", closed="left", label="right")
            .agg(agg_rules)
            .dropna()
            .reset_index()
        )
        if not df_1d_raw.empty:
            df_1d, features_1d, feature_names_1d = self._calculate_features(
                df_1d_raw, "1d"
            )
            results["1d"] = (df_1d.index.values, features_1d, feature_names_1d)  # dates

        return results

    # <<< MỚI: Hàm này được viết lại hoàn toàn để căn chỉnh 3 khung thời gian
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

        # results chứa {'1h': (...), '4h': (...), '1d': (...)}
        results = self._process_coin_numpy_optimized(df_raw)

        # --- Xử lý 1H (Giống như cũ) ---
        if "1h" not in results:
            return  # Bỏ qua nếu không có dữ liệu 1h

        (
            dates_1h,
            features_array_1h,
            labels_trade_array,
            labels_dir_array,
            feature_names_1h,
        ) = results["1h"]

        if is_first_coin:
            self.feature_names["1h"] = feature_names_1h
            self.n_features["1h"] = len(feature_names_1h)

        master_indices_1h = []
        df_indices_1h = []
        for i, date in enumerate(dates_1h):
            if date in self.date_to_idx_map:
                master_indices_1h.append(self.date_to_idx_map[date])
                df_indices_1h.append(i)

        if not master_indices_1h:
            return  # Bỏ qua nếu không có mốc thời gian 1h nào khớp

        valid_features_1h = features_array_1h[df_indices_1h].astype(np.float32)
        local_indices_for_map_1h = np.arange(len(df_indices_1h), dtype=np.int32)

        # Ghi trực tiếp
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
            dates_4h, features_array_4h, feature_names_4h = results["4h"]
            if is_first_coin:
                self.feature_names["4h"] = feature_names_4h
                self.n_features["4h"] = len(feature_names_4h)

            # Tạo DF 4h cục bộ để merge
            df_4h_local = pd.DataFrame(
                {
                    "date": dates_4h,
                    "local_idx_4h": np.arange(len(dates_4h), dtype=np.int32),
                }
            )

            # Merge_asof để tìm nến 4h *đã đóng* gần nhất cho *mọi* mốc 1h
            aligned_4h = pd.merge_asof(
                self.master_df,
                df_4h_local,
                on="date",
                direction="backward",
            )

            # Điền vào bản đồ
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
            dates_1d, features_array_1d, feature_names_1d = results["1d"]
            if is_first_coin:
                self.feature_names["1d"] = feature_names_1d
                self.n_features["1d"] = len(feature_names_1d)

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

    # <<< MỚI: Viết lại hoàn toàn để kiểm tra 3 lookback
    def _calculate_validity_mask(self):
        """
        Cập nhật `mask_aligned` để chỉ bao gồm các mốc thời gian
        có đủ lookback cho CẢ BA khung thời gian.
        """
        print("Calculating final validity mask across 1h, 4h, 1d...")

        # 1. Kiểm tra 1H: Phải có dữ liệu 1h (mask_1h_raw) VÀ đủ lookback 1h
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

        # 2. Kiểm tra 4H: Phải map tới một chỉ số 4h VÀ chỉ số đó >= (T_4h - 1)
        T_4h = self.hparams.lookback_window_4h
        map_4h = self.master_to_local_map_aligned["4h"]
        final_mask_4h = map_4h >= (T_4h - 1)

        # 3. Kiểm tra 1D: Phải map tới một chỉ số 1d VÀ chỉ số đó >= (T_1d - 1)
        T_1d = self.hparams.lookback_window_1d
        map_1d = self.master_to_local_map_aligned["1d"]
        final_mask_1d = map_1d >= (T_1d - 1)

        # Mask cuối cùng là AND của cả ba
        self.mask_aligned = final_mask_1h & final_mask_4h & final_mask_1d

        # Giải phóng bộ nhớ mask 1h thô
        del self._mask_1h_raw
        gc.collect()

    def _calculate_coin_baselines(self):
        """Calculates the positive trade rate (baseline) for each coin."""
        # Logic này vẫn đúng vì nó dựa trên self.mask_aligned (đã được cập nhật)
        # và self.train_indices
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
        # self.mask_aligned giờ đã là mask tổng hợp 1h+4h+1d
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

        # Logic lặp lại (repeat) vẫn giữ nguyên
        n_valid_coins_at_train_indices = valid_coins_per_timestamp[unique_train_indices]
        n_repeats = np.ceil(
            n_valid_coins_at_train_indices / self.hparams.portfolio_size
        ).astype(np.intp)
        n_repeats = np.maximum(n_repeats, 1)
        self.train_indices = np.repeat(unique_train_indices, n_repeats)

    # --- Validation Coin Selection ---
    # (Toàn bộ logic này không cần thay đổi vì nó hoạt động trên self.mask_aligned)

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
        """Parses the user-provided list of coin names into indices."""
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
        """Filters a list of coin indices, keeping only those valid across all val timestamps."""
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
        """Ensures the list of indices has exactly P coins."""
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
        """Auto-selects the top P coins based on data availability."""
        valid_samples_per_coin = np.sum(self.mask_aligned, axis=1)
        top_p_coin_indices = np.argsort(valid_samples_per_coin)[-P:]
        return list(top_p_coin_indices)

    # --- Collate Function & DataLoaders ---

    # <<< MỚI: Viết lại hoàn toàn để lắp ráp 3 tensor đặc trưng
    def _create_collator(self, is_train: bool):
        """
        Factory tạo ra collate_fn cho 3 khung thời gian.
        """
        # Lấy kích thước từ hparams và state
        T_1h = self.hparams.lookback_window_1h
        T_4h = self.hparams.lookback_window_4h
        T_1d = self.hparams.lookback_window_1d
        F_1h = self.n_features["1h"]
        F_4h = self.n_features["4h"]
        F_1d = self.n_features["1d"]
        P = self.hparams.portfolio_size

        def collate_fn(batch_timestamp_indices: List[int]) -> Dict[str, torch.Tensor]:
            B = len(batch_timestamp_indices)

            # Khởi tạo các mảng trống cho 3 khung thời gian
            batch_features_1h = np.empty((B, P, T_1h, F_1h), dtype=np.float32)
            batch_features_4h = np.empty((B, P, T_4h, F_4h), dtype=np.float32)
            batch_features_1d = np.empty((B, P, T_1d, F_1d), dtype=np.float32)

            list_labels_trade = []
            list_labels_dir = []
            list_coin_ids = []

            for i, ts_idx in enumerate(batch_timestamp_indices):
                if is_train:
                    # mask_aligned đã là mask tổng hợp, nên logic này vẫn đúng
                    valid_coin_indices = np.where(self.mask_aligned[:, ts_idx])[0]
                    coin_indices = np.random.choice(
                        valid_coin_indices, P, replace=False
                    )
                    coin_indices.sort()
                else:
                    coin_indices = self.val_coin_indices

                # Nhãn và coin_ids (như cũ, vì chúng dựa trên 1h)
                list_labels_trade.append(
                    self.all_labels_trade_aligned[coin_indices, ts_idx]
                )
                list_labels_dir.append(
                    self.all_labels_dir_aligned[coin_indices, ts_idx]
                )
                list_coin_ids.append(coin_indices)

                # Lắp ráp đặc trưng cho từng coin trong danh mục
                for p_idx, coin_idx in enumerate(coin_indices):
                    # --- 1. Lấy dữ liệu 1H ---
                    # Tạo dải chỉ số 1h trong không gian master
                    master_idx_range_1h = np.arange(ts_idx - T_1h + 1, ts_idx + 1)
                    # Map về chỉ số 1h cục bộ
                    local_indices_1h = self.master_to_local_map_aligned["1h"][
                        coin_idx, master_idx_range_1h
                    ]
                    # Cắt lát
                    features_slice_1h = self.features_per_coin["1h"][coin_idx][
                        local_indices_1h, :
                    ]
                    batch_features_1h[i, p_idx, :, :] = features_slice_1h

                    # --- 2. Lấy dữ liệu 4H ---
                    # Lấy chỉ số 4h cục bộ *hiện tại* (đã đóng)
                    current_local_idx_4h = self.master_to_local_map_aligned["4h"][
                        coin_idx, ts_idx
                    ]
                    # Tạo dải chỉ số 4h trong không gian *cục bộ 4h*
                    local_indices_4h = np.arange(
                        current_local_idx_4h - T_4h + 1, current_local_idx_4h + 1
                    )
                    # Cắt lát
                    features_slice_4h = self.features_per_coin["4h"][coin_idx][
                        local_indices_4h, :
                    ]
                    batch_features_4h[i, p_idx, :, :] = features_slice_4h

                    # --- 3. Lấy dữ liệu 1D ---
                    # Lấy chỉ số 1d cục bộ *hiện tại* (đã đóng)
                    current_local_idx_1d = self.master_to_local_map_aligned["1d"][
                        coin_idx, ts_idx
                    ]
                    # Tạo dải chỉ số 1d trong không gian *cục bộ 1d*
                    local_indices_1d = np.arange(
                        current_local_idx_1d - T_1d + 1, current_local_idx_1d + 1
                    )
                    # Cắt lát
                    features_slice_1d = self.features_per_coin["1d"][coin_idx][
                        local_indices_1d, :
                    ]
                    batch_features_1d[i, p_idx, :, :] = features_slice_1d

            # <<< MỚI: Trả về 3 tensor đặc trưng
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
            collate_fn=self._create_collator(is_train=True),  # <<< MỚI
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
            collate_fn=self._create_collator(is_train=False),  # <<< MỚI
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Test dataloader is not implemented.")
