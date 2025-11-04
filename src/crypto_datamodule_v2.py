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

# (Đã xóa joblib)
# Thêm lại rich.progress
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
    Numba-accelerated triple-barrier labeling.
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


FEATURE_CONFIG = {
    "open": {"norm_type": 2},
    "high": {"norm_type": 2},
    "low": {"norm_type": 2},
    "close": {"norm_type": 2},
    "sar": {"norm_type": 2},
    "bb_upper": {"norm_type": 2},
    "bb_lower": {"norm_type": 2},
    "atr": {"norm_type": 1},
    "macd": {"norm_type": 1},
    "macd_signal": {"norm_type": 1},
    "volume": {"norm_type": 1},
    "obv": {"norm_type": 1},
    "rsi": {"norm_type": 0},
    "stoch_k": {"norm_type": 0},
    "stoch_d": {"norm_type": 0},
    "cci": {"norm_type": 0},
    "mfi": {"norm_type": 0},
    "adx": {"norm_type": 0},
    "cmf": {"norm_type": 0},
    "roc": {"norm_type": 0},
    "sma_20": {"norm_type": 2},
    "sma_50": {"norm_type": 2},
    "ema_20": {"norm_type": 2},
    "ema_50": {"norm_type": 2},
    "candle_range": {"norm_type": 0},
    "candle_body_pct": {"norm_type": 0},
    "candle_wick_pct": {"norm_type": 0},
    "log_return": {"norm_type": 0},
    "temporal_sin": {"norm_type": 0},
    "temporal_cos": {"norm_type": 0},
}


# ========================================================================
# 2. DATASET (Minimalist)
# ========================================================================


class CryptoPortfolioDataset(Dataset):
    """
    Minimalist Dataset.
    Returns a single integer (timestamp index).
    The collate_fn handles all batch assembly logic.
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


class CryptoDataModuleV2(LightningDataModule):
    """
    Refactored DataModule using a sequential (single-thread) setup loop.
    This is the most memory-stable option, albeit the slowest to setup.
    """

    def __init__(
        self,
        data_dir: str = "./data/crypto-1200",
        end_date: str = "2025-09-29",
        validation_start_date: str = "2025-06-01",
        candle_length: int = 40000,
        lookback_window: int = 64,
        portfolio_size: int = 64,
        barrier_atr_multiplier: float = 2.0,
        barrier_horizon: int = 4,
        batch_size: int = 4,
        num_workers: int = 4,
        max_coins: int = -1,
        validation_coins: Optional[List[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- Paths and Dates ---
        self.data_dir = Path(self.hparams.data_dir)
        self.end_date = pd.to_datetime(self.hparams.end_date, utc=True)
        self.validation_start_date = pd.to_datetime(
            self.hparams.validation_start_date, utc=True
        )

        # --- Feature Config ---
        self.feature_names = list(FEATURE_CONFIG.keys())

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
        self.master_to_local_map_aligned: np.ndarray = None
        self.features_per_coin: List[np.ndarray] = []

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Main setup method.
        Runs a sequential loop to process coins one-by-one.
        """
        if self.master_to_local_map_aligned is not None:
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
        self.mask_aligned = np.full((n_coins, n_timestamps), False, dtype=np.bool_)
        self.master_to_local_map_aligned = np.full(
            (n_coins, n_timestamps), -1, dtype=np.int32
        )
        self.features_per_coin = [None] * n_coins
        print("Initialized empty data arrays.")

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

            for coin_info in jobs:
                # Call the processing function directly
                self._process_and_fill_coin(coin_info)
                progress.update(task, advance=1)

        print("Sequential processing complete.")
        gc.collect()

        # --- 4. Post-processing (Splitting, Masking) ---
        self._calculate_validity_mask()
        self._find_samples_and_split()
        self._find_validation_coins()
        self._calculate_coin_baselines()

        print(f"--- Setup complete ---")
        print(f"  Total valid train samples: {len(self.train_indices):,}")
        print(f"  Total valid val samples: {len(self.val_indices):,}")

    # --- Processing Functions (Called Sequentially) ---

    def _load_single_coin_data(self, coin: str) -> Optional[pd.DataFrame]:
        """
        Loads and filters Parquet data for a single coin.
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

        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    def _process_coin_numpy_optimized(
        self, df_raw: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates all features (TA-lib) and labels (Numba) for a single coin's DataFrame.
        """
        open_p = df_raw["open"].values
        high_p = df_raw["high"].values
        low_p = df_raw["low"].values
        close_p = df_raw["close"].values
        volume = df_raw["volume"].values
        dates = df_raw["date"].values

        cols = {
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close_p,
            "volume": volume,
        }

        cols["log_return"] = np.log(close_p / (np.roll(close_p, 1) + 1e-8)).astype(
            np.float32
        )
        cols["sar"] = talib.SAR(high_p, low_p).astype(np.float32)
        bb_upper, _, bb_lower = talib.BBANDS(close_p, timeperiod=20)
        cols["bb_upper"] = bb_upper.astype(np.float32)
        cols["bb_lower"] = bb_lower.astype(np.float32)

        cols["adx"] = (talib.ADX(high_p, low_p, close_p, timeperiod=14) / 50.0).astype(
            np.float32
        ) - 1.0
        cols["rsi"] = (talib.RSI(close_p, timeperiod=14) / 50.0).astype(
            np.float32
        ) - 1.0
        stoch_k, stoch_d = talib.STOCH(high_p, low_p, close_p)
        cols["stoch_k"] = (stoch_k / 50.0).astype(np.float32) - 1.0
        cols["stoch_d"] = (stoch_d / 50.0).astype(np.float32) - 1.0
        cci_raw = talib.CCI(high_p, low_p, close_p, timeperiod=14)
        cols["cci"] = (np.clip(cci_raw, -200, 200) / 200.0).astype(np.float32)
        cols["mfi"] = (
            talib.MFI(high_p, low_p, close_p, volume, timeperiod=14) / 50.0
        ).astype(np.float32) - 1.0
        roc_raw = talib.ROC(close_p, timeperiod=10)
        cols["roc"] = (np.clip(roc_raw, -20, 20) / 20.0).astype(np.float32)

        cols["cmf"] = pta.cmf(
            df_raw["high"], df_raw["low"], df_raw["close"], df_raw["volume"], length=20
        ).values.astype(np.float32)

        macd, macd_signal, _ = talib.MACD(close_p)
        cols["macd"] = macd.astype(np.float32)
        cols["macd_signal"] = macd_signal.astype(np.float32)

        cols["ema_20"] = talib.EMA(close_p, timeperiod=20).astype(np.float32)
        cols["ema_50"] = talib.EMA(close_p, timeperiod=50).astype(np.float32)
        cols["sma_20"] = talib.SMA(close_p, timeperiod=20).astype(np.float32)
        cols["sma_50"] = talib.SMA(close_p, timeperiod=50).astype(np.float32)
        cols["obv"] = talib.OBV(close_p, volume).astype(np.float32)
        cols["atr"] = talib.ATR(high_p, low_p, close_p, timeperiod=14).astype(
            np.float32
        )

        cols["candle_range"] = ((high_p - low_p) / (open_p + 1e-8)).astype(np.float32)
        cols["candle_body_pct"] = (
            np.abs(close_p - open_p) / (high_p - low_p + 1e-8)
        ).astype(np.float32)
        cols["candle_wick_pct"] = (
            (high_p - np.maximum(open_p, close_p)) / (high_p - low_p + 1e-8)
        ).astype(np.float32)

        cols["temporal_sin"] = np.sin(
            2 * np.pi * df_raw["date"].dt.hour / 24.0
        ).values.astype(np.float32)
        cols["temporal_cos"] = np.cos(
            2 * np.pi * df_raw["date"].dt.hour / 24.0
        ).values.astype(np.float32)

        labels_trade, labels_dir = _calculate_labels_numba(
            close_p,
            cols["atr"],
            self.hparams.barrier_atr_multiplier,
            self.hparams.barrier_horizon,
        )
        cols["label_trade"] = labels_trade
        cols["label_dir"] = labels_dir

        df = pd.DataFrame(cols, index=dates)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        feature_df = df[self.feature_names]
        feature_df = feature_df.ffill().bfill().fillna(0)

        features_array = feature_df.values.astype(np.float32)
        labels_trade_array = df["label_trade"].values.astype(np.bool_)
        labels_dir_array = df["label_dir"].values.astype(np.bool_)

        return (dates, features_array, labels_trade_array, labels_dir_array)

    def _process_and_fill_coin(self, coin_info: Tuple[int, str]):
        """
        The main sequential "pipeline" function for a single coin.
        It loads, processes, and writes results directly to class arrays.
        """
        coin_idx, coin = coin_info

        try:
            df_raw = self._load_single_coin_data(coin)
            if df_raw is None:
                return

            (
                dates,
                features_array,
                labels_trade_array,
                labels_dir_array,
            ) = self._process_coin_numpy_optimized(df_raw)

            master_indices = []
            df_indices = []
            for i, date in enumerate(dates):
                if date in self.date_to_idx_map:
                    master_indices.append(self.date_to_idx_map[date])
                    df_indices.append(i)
            if not master_indices:
                return

            valid_features = features_array[df_indices].astype(np.float32)
            local_indices_for_map = np.arange(len(df_indices), dtype=np.int32)

            # Direct write
            self.all_labels_trade_aligned[coin_idx, master_indices] = (
                labels_trade_array[df_indices]
            )
            self.all_labels_dir_aligned[coin_idx, master_indices] = labels_dir_array[
                df_indices
            ]
            self.mask_aligned[coin_idx, master_indices] = True
            self.features_per_coin[coin_idx] = valid_features
            self.master_to_local_map_aligned[coin_idx, master_indices] = (
                local_indices_for_map
            )

        except Exception as e:
            print(f"[Warning] Error processing {coin} (idx {coin_idx}): {e}")

    # --- Post-Processing Functions ---

    def _calculate_validity_mask(self):
        """
        Updates the `mask_aligned` to only include timesteps
        that have a full lookback window.
        """
        history_mask = self.mask_aligned  # True where data exists
        history_counts = bn.move_sum(
            history_mask.astype(np.int8),
            window=self.hparams.lookback_window,
            axis=1,
            min_count=self.hparams.lookback_window,
        )
        valid_lookback = history_counts == self.hparams.lookback_window

        self.mask_aligned = history_mask & valid_lookback

    def _calculate_coin_baselines(self):
        """Calculates the positive trade rate (baseline) for each coin."""
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
        and splits them into train/val sets.
        """
        valid_coins_per_timestamp = np.sum(self.mask_aligned, axis=0)
        valid_sample_mask = valid_coins_per_timestamp >= self.hparams.portfolio_size
        all_valid_sample_indices = np.where(valid_sample_mask)[0]

        if len(all_valid_sample_indices) == 0:
            raise ValueError(
                f"No timestamps found with at least {self.hparams.portfolio_size} valid coins."
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

    # --- Validation Coin Selection (Refactored) ---

    def _find_validation_coins(self):
        """
        Orchestrates the selection of a fixed validation coin portfolio.
        """
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

    def _create_collator(self, is_train: bool):
        """
        Factory that creates the collate_fn.
        """
        n_features = len(self.feature_names)

        def collate_fn(batch_timestamp_indices: List[int]) -> Dict[str, torch.Tensor]:
            B = len(batch_timestamp_indices)
            T = self.hparams.lookback_window
            P = self.hparams.portfolio_size
            F = n_features
            batch_features = np.empty((B, P, T, F), dtype=np.float32)
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

                master_idx_range = np.arange(ts_idx - T + 1, ts_idx + 1)

                for p_idx, coin_idx in enumerate(coin_indices):
                    try:
                        local_indices = self.master_to_local_map_aligned[
                            coin_idx, master_idx_range
                        ]
                        features_slice = self.features_per_coin[coin_idx][
                            local_indices, :
                        ]
                        batch_features[i, p_idx, :, :] = features_slice
                    except (IndexError, TypeError):
                        batch_features[i, p_idx, :, :] = 0.0

                list_labels_trade.append(
                    self.all_labels_trade_aligned[coin_indices, ts_idx]
                )
                list_labels_dir.append(
                    self.all_labels_dir_aligned[coin_indices, ts_idx]
                )
                list_coin_ids.append(coin_indices)

            return {
                "features": torch.from_numpy(batch_features),
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
