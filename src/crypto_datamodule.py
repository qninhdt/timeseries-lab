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

# --- Numba Accelerated Functions ---


@jit(nopython=True)
def _calculate_labels_numba(
    close_prices: np.ndarray, barrier_up: float, barrier_down: float, horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the trade and direction labels using the triple barrier method.
    """
    n = len(close_prices)
    labels_trade = np.zeros(n, dtype=np.bool_)
    labels_dir = np.zeros(n, dtype=np.bool_)

    for i in range(n - 1):
        entry_price = close_prices[i]
        upper_barrier = entry_price * (1 + barrier_up)
        lower_barrier = entry_price * (1 - barrier_down)

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


# --- Dataset Class (Minimalist) ---

FEATURE_CONFIG = {
    "open": {"norm": "close"},
    "high": {"norm": "close"},
    "low": {"norm": "close"},
    "close": {"norm": "close"},
    "sar": {"norm": "close"},
    "vwap": {"norm": "close"},
    "bb_upper": {"norm": "close"},
    "bb_middle": {"norm": "close"},
    "bb_lower": {"norm": "close"},
    "atr": {"norm": "atr"},
    "macd": {"norm": "macd"},
    "macd_signal": {"norm": "macd"},
    # "macd_hist": {"norm": "macd"},
    "volume": {"norm": "volume"},
    "obv": {"norm": "obv"},
    "rsi": {"norm": None},
    "stoch_k": {"norm": None},
    "stoch_d": {"norm": None},
    "cci": {"norm": None},
    "mfi": {"norm": None},
    "adx": {"norm": None},
    "cmf": {"norm": None},
    "roc": {"norm": None},
    "sma_20": {"norm": "close"},
    "sma_50": {"norm": "close"},
    # "ema_20": {"norm": "close"},
    # "ema_50": {"norm": "close"},
    # "candle_range": {"norm": None},
    # "candle_body_pct": {"norm": None},
    # "candle_wick_pct": {"norm": None},
    # "log_return": {"norm": None},
    # "temporal_sin": {"norm": None},
    # "temporal_cos": {"norm": None},
}


class CryptoPortfolioDataset(Dataset):
    """
    A minimalist Dataset.
    It only holds the indices of valid samples for a specific split (train/val).
    """

    def __init__(self, sample_indices: np.ndarray):
        self.sample_indices = sample_indices

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> int:
        return self.sample_indices[idx]


# --- Data Module Class (Main Logic) ---


class CryptoDataModule(LightningDataModule):
    """
    Manages all aspects of data loading, processing, and batching.
    Optimized to process and align data in a streaming fashion to
    avoid memory peaks.
    """

    def __init__(
        self,
        data_dir: str = "./data/crypto-700",
        end_date: str = "2025-09-29",
        validation_start_date: str = "2025-06-01",
        candle_length: int = 40000,
        lookback_window: int = 64,
        portfolio_size: int = 64,
        barrier_up: float = 0.02,
        barrier_down: float = 0.02,
        barrier_horizon: int = 4,
        batch_size: int = 4,
        num_workers: int = 4,
        max_coins: int = -1,
    ):
        """
        Initializes the DataModule.

        Args:
            data_dir (str): Path to the merged data directory.
            end_date (str): The final date to include in the dataset (YYYY-MM-DD).
            validation_start_date (str): The date to split train and val sets.
            candle_length (int): Fixed number of hours to create in the master
                                 timestamp index, ending at `end_date`.
            lookback_window (int): Number of historical candles (T) for the model.
            portfolio_size (int): Number of coins (P) in each portfolio sample.
            barrier_up (float): Upper barrier for label generation (+%).
            barrier_down (float): Lower barrier for label generation (-%).
            barrier_horizon (int): Lookahead period for label generation (in hours).
            batch_size (int): Number of samples per batch (B).
            num_workers (int): Number of parallel workers for data loading.
            max_coins (int): Maximum number of coins to process (-1 means all coins).
        """
        super().__init__()
        self.save_hyperparameters()

        # --- Paths and Dates ---
        self.data_dir = Path(data_dir)
        self.end_date = pd.to_datetime(end_date, utc=True)
        self.validation_start_date = pd.to_datetime(validation_start_date, utc=True)

        # --- Feature & Label Config ---
        self._setup_feature_config()

        # --- Internal State ---
        self.coins = []
        self.master_timestamps = None
        self.date_to_idx_map = {}
        self.train_indices = None
        self.val_indices = None

        # --- Aligned Data (Large numpy arrays) ---
        self.all_features_aligned = None
        self.all_stats_aligned = None
        self.all_labels_trade_aligned = None
        self.all_labels_dir_aligned = None
        self.mask_aligned = None
        self.val_coin_indices = None

    def _setup_feature_config(self):
        """
        Defines the feature configuration and creates all necessary
        mappings for normalization and statistics calculation.
        """
        self.feature_config = FEATURE_CONFIG

        self.feature_names = list(self.feature_config.keys())

        stats_needed = set()
        self.normalization_map = {}
        self.clip_only_indices = []

        for i, (name, config) in enumerate(self.feature_config.items()):
            norm_strategy = config["norm"]
            if norm_strategy is None:
                self.clip_only_indices.append(i)
            else:
                stats_needed.add(norm_strategy)
                self.normalization_map[i] = norm_strategy

        self.stats_source_names = sorted(list(stats_needed))
        self.stats_to_calc = {name: name for name in self.stats_source_names}
        self.stats_cols = [f"stats_{name}_mean" for name in self.stats_source_names] + [
            f"stats_{name}_std" for name in self.stats_source_names
        ]

        self.stats_map = {}
        n_stats_sources = len(self.stats_source_names)
        for i, name in enumerate(self.stats_source_names):
            self.stats_map[name] = (i, i + n_stats_sources)

        self.clip_only_indices = np.array(self.clip_only_indices, dtype=np.intp)

    def prepare_data(self):
        """
        Loads metadata. This runs on only one process.
        """
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            self.coins = list(metadata["coins"].keys())
            print(f"Loaded {len(self.coins)} coins from metadata.")

    def setup(self, stage: Optional[str] = None):
        """
        The main data processing pipeline. This runs on every GPU process.
        1. Generate master timestamp index.
        2. Initialize large, empty, shared arrays.
        3. Load and process all coins in parallel, filling arrays directly.
        4. Calculate the validity mask.
        5. Determine sample indices and split train/val.
        6. Identify fixed coins for validation.
        """
        if self.all_features_aligned is not None:
            print("Data already set up.")
            return

        print(f"--- Starting setup for stage: {stage} ---")

        # Apply max_coins limit
        if self.hparams.max_coins > 0 and len(self.coins) > self.hparams.max_coins:
            self.coins = self.coins[: self.hparams.max_coins]
            print(
                f"Limited to {len(self.coins)} coins (max_coins={self.hparams.max_coins})"
            )

        # --- 1. Generate Master Timestamps ---
        print(
            f"Generating {self.hparams.candle_length} master timestamps ending at {self.end_date}..."
        )
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
        n_coins = len(self.coins)
        n_features = len(self.feature_names)
        n_stats = len(self.stats_cols)
        print(f"Master timestamps created (Total: {n_timestamps})")

        # --- 2. Initialize large aligned arrays (fill with NaN) ---
        self.all_features_aligned = np.full(
            (n_coins, n_timestamps, n_features), np.nan, dtype=np.float32
        )
        self.all_stats_aligned = np.full(
            (n_coins, n_timestamps, n_stats), np.nan, dtype=np.float32
        )
        self.all_labels_trade_aligned = np.full(
            (n_coins, n_timestamps), False, dtype=np.bool_
        )
        self.all_labels_dir_aligned = np.full(
            (n_coins, n_timestamps), False, dtype=np.bool_
        )

        # --- 3. Process all coins in parallel and fill arrays ---
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            transient=True,
        )

        with progress:
            task_process = progress.add_task(
                f"Processing {n_coins} coins", total=n_coins
            )

            # Process coins sequentially (no multi-threading)
            for coin_idx, coin in enumerate(self.coins):
                self._process_and_fill(coin_idx, coin)
                progress.update(task_process, advance=1)

        print("Data processing and alignment complete.")
        gc.collect()

        # --- 4. Calculate validity mask ---
        self._calculate_validity_mask()

        # --- 5. Determine sample indices and split ---
        self._find_samples_and_split()

        # --- 6. Identify validation coins ---
        self._find_validation_coins()

        print(f"--- Setup complete ---")
        print(f"  Total valid train samples: {len(self.train_indices):,}")
        print(f"  Total valid val samples: {len(self.val_indices):,}")
        print(
            f"  Validation coins (P={self.hparams.portfolio_size}): {self.val_coin_names}"
        )

        # Print date ranges for train and validation
        if len(self.train_indices) > 0:
            train_start_idx = np.min(self.train_indices)
            train_end_idx = np.max(self.train_indices)
            train_start_date = pd.to_datetime(self.master_timestamps[train_start_idx])
            train_end_date = pd.to_datetime(self.master_timestamps[train_end_idx])
            train_days = (train_end_date - train_start_date).days
            print(
                f"  Train date range: {train_start_date} to {train_end_date} ({train_days} days)"
            )

        if len(self.val_indices) > 0:
            val_start_idx = np.min(self.val_indices)
            val_end_idx = np.max(self.val_indices)
            val_start_date = pd.to_datetime(self.master_timestamps[val_start_idx])
            val_end_date = pd.to_datetime(self.master_timestamps[val_end_idx])
            val_days = (val_end_date - val_start_date).days
            print(
                f"  Val date range: {val_start_date} to {val_end_date} ({val_days} days)"
            )

    def _load_single_coin_data(self, coin: str) -> Optional[pd.DataFrame]:
        """Loads 1h data, truncates to end_date."""
        filename = f"{coin.lower()}.parquet"
        filepath = self.data_dir / "data" / filename

        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath)
        df["date"] = pd.to_datetime(df["date"])

        # Filter by end date
        df = df[df["date"] <= self.end_date]

        if df.empty:
            return None

        return df[["date", "open", "high", "low", "close", "volume"]].copy()

    def _process_and_fill(self, coin_idx: int, coin: str):
        """
        Loads data, calls the optimized processing worker,
        and fills the shared numpy arrays directly.
        """
        try:
            df_raw = self._load_single_coin_data(coin)
            if df_raw is None:
                return

            (
                dates,
                features_array,
                stats_array,
                labels_trade_array,
                labels_dir_array,
            ) = self._process_coin_numpy_optimized(df_raw)

            master_indices = [
                self.date_to_idx_map[date]
                for date in dates
                if date in self.date_to_idx_map
            ]
            df_indices = [
                i for i, date in enumerate(dates) if date in self.date_to_idx_map
            ]

            if not master_indices:
                return

            self.all_features_aligned[coin_idx, master_indices, :] = features_array[
                df_indices
            ]
            self.all_stats_aligned[coin_idx, master_indices, :] = stats_array[
                df_indices
            ]
            self.all_labels_trade_aligned[coin_idx, master_indices] = (
                labels_trade_array[df_indices]
            )
            self.all_labels_dir_aligned[coin_idx, master_indices] = labels_dir_array[
                df_indices
            ]

        except Exception as e:
            print(f"Error processing {coin} (idx {coin_idx}): {e}")
            return

    def _process_coin_numpy_optimized(
        self, df_raw: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates all features, labels, and stats using a NumPy-first
        approach to minimize GIL contention.
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

        # cols["log_return"] = np.log(close_p / (np.roll(close_p, 1) + 1e-8))
        cols["sar"] = talib.SAR(high_p, low_p)
        cols["bb_upper"], cols["bb_middle"], cols["bb_lower"] = talib.BBANDS(
            close_p, timeperiod=20
        )
        cols["adx"] = (talib.ADX(high_p, low_p, close_p, timeperiod=14) / 50.0) - 1.0
        cols["rsi"] = (talib.RSI(close_p, timeperiod=14) / 50.0) - 1.0
        stoch_k, stoch_d = talib.STOCH(high_p, low_p, close_p)
        cols["stoch_k"] = (stoch_k / 50.0) - 1.0
        cols["stoch_d"] = (stoch_d / 50.0) - 1.0
        cols["cci"] = (
            np.clip(talib.CCI(high_p, low_p, close_p, timeperiod=14), -200, 200) / 200.0
        )
        cols["mfi"] = (
            talib.MFI(high_p, low_p, close_p, volume, timeperiod=14) / 50.0
        ) - 1.0
        cols["roc"] = np.clip(talib.ROC(close_p, timeperiod=10), -20, 20) / 20.0

        df_temp = df_raw.set_index("date")
        cols["cmf"] = pta.cmf(
            df_temp["high"],
            df_temp["low"],
            df_temp["close"],
            df_temp["volume"],
            length=20,
        ).values
        vwap = pta.vwap(
            df_temp["high"], df_temp["low"], df_temp["close"], df_temp["volume"]
        )
        cols["vwap"] = vwap.values if vwap is not None else 0

        cols["macd"], cols["macd_signal"], cols["macd_hist"] = talib.MACD(close_p)
        # cols["ema_20"] = talib.EMA(close_p, timeperiod=20)
        # cols["ema_50"] = talib.EMA(close_p, timeperiod=50)
        cols["sma_20"] = talib.SMA(close_p, timeperiod=20)
        cols["sma_50"] = talib.SMA(close_p, timeperiod=50)
        cols["obv"] = talib.OBV(close_p, volume)
        cols["atr"] = talib.ATR(high_p, low_p, close_p, timeperiod=14)

        # cols["candle_range"] = (high_p - low_p) / (open_p + 1e-8)
        # cols["candle_body_pct"] = np.abs(close_p - open_p) / (high_p - low_p + 1e-8)
        # cols["candle_wick_pct"] = (high_p - np.maximum(open_p, close_p)) / (
        #     high_p - low_p + 1e-8
        # )
        # cols["temporal_sin"] = np.sin(2 * np.pi * df_raw["date"].dt.hour / 24.0).values
        # cols["temporal_cos"] = np.cos(2 * np.pi * df_raw["date"].dt.hour / 24.0).values

        labels_trade, labels_dir = _calculate_labels_numba(
            close_p,
            self.hparams.barrier_up,
            self.hparams.barrier_down,
            self.hparams.barrier_horizon,
        )
        cols["label_trade"] = labels_trade
        cols["label_dir"] = labels_dir

        df = pd.DataFrame(cols, index=dates)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().bfill().fillna(0)

        window = self.hparams.lookback_window
        for stat_name, source_col in self.stats_to_calc.items():
            if source_col not in df.columns:
                continue
            rolling_mean = bn.move_mean(df[source_col], window=window, min_count=1)
            rolling_std = bn.move_std(
                df[source_col], window=window, min_count=1, ddof=0
            )
            rolling_std = np.maximum(rolling_std, 1e-8)
            df[f"stats_{stat_name}_mean"] = rolling_mean
            df[f"stats_{stat_name}_std"] = rolling_std

        df = df.ffill().bfill().fillna(0)

        features_array = df[self.feature_names].values.astype(np.float32)
        stats_array = df[self.stats_cols].values.astype(np.float32)
        labels_trade_array = df["label_trade"].values.astype(np.bool_)
        labels_dir_array = df["label_dir"].values.astype(np.bool_)

        return (
            dates,
            features_array,
            stats_array,
            labels_trade_array,
            labels_dir_array,
        )

    def _calculate_validity_mask(self):
        """
        Calculates a boolean mask (n_coins, n_timestamps) indicating
        which (coin, time) pairs are valid for sampling.
        """
        close_idx = self.feature_names.index("close")
        history_mask = ~np.isnan(self.all_features_aligned[:, :, close_idx])

        history_counts = bn.move_sum(
            history_mask.astype(np.int8),
            window=self.hparams.lookback_window,
            axis=1,
            min_count=self.hparams.lookback_window,
        )
        valid_lookback = history_counts == self.hparams.lookback_window
        self.mask_aligned = history_mask & valid_lookback

    def _find_samples_and_split(self):
        """
        Finds all timestamps that have at least `portfolio_size` valid coins.
        Splits these valid timestamps into train and validation sets.
        """
        valid_coins_per_timestamp = np.sum(self.mask_aligned, axis=0)
        valid_sample_mask = valid_coins_per_timestamp >= self.hparams.portfolio_size
        valid_sample_indices = np.where(valid_sample_mask)[0]

        if len(valid_sample_indices) == 0:
            raise ValueError(
                f"No timestamps found with at least {self.hparams.portfolio_size} valid coins."
            )

        val_start_idx = np.searchsorted(
            self.master_timestamps,
            np.datetime64(self.validation_start_date),
            side="left",
        )

        val_mask = valid_sample_indices >= val_start_idx
        train_mask = ~val_mask

        self.train_indices = valid_sample_indices[train_mask]
        self.val_indices = valid_sample_indices[val_mask]

        if len(self.train_indices) == 0:
            raise ValueError(
                "No training samples found. Adjust `validation_start_date`."
            )
        if len(self.val_indices) == 0:
            raise ValueError(
                "No validation samples found. Adjust `validation_start_date`."
            )

    def _find_validation_coins(self):
        """
        Finds the P coins with the longest history to use as the
        fixed validation set.
        """
        valid_samples_per_coin = np.sum(self.mask_aligned, axis=1)
        top_p_coin_indices = np.argsort(valid_samples_per_coin)[
            -self.hparams.portfolio_size :
        ]
        self.val_coin_indices = np.sort(top_p_coin_indices)
        self.val_coin_names = [self.coins[i] for i in self.val_coin_indices]

    def _normalize_batch(self, features_batch: np.ndarray, stats_batch: np.ndarray):
        """
        Normalizes a (B, P, T, D) feature batch IN-PLACE
        using a (B, P, N_Stats) stats batch.
        """
        B, P, T, D = features_batch.shape

        for feat_idx, stat_name in self.normalization_map.items():
            mean_idx, std_idx = self.stats_map[stat_name]
            b_feat_mean = stats_batch[:, :, mean_idx].reshape(B, P, 1)
            b_feat_std = stats_batch[:, :, std_idx].reshape(B, P, 1)

            features_batch[:, :, :, feat_idx] = (
                features_batch[:, :, :, feat_idx] - b_feat_mean
            ) / b_feat_std

        np.clip(features_batch, -5.0, 5.0, out=features_batch)

    def _create_collator(self, is_train: bool):
        """
        Factory function to create the appropriate collate_fn for train or val.
        """

        def collate_fn(batch_timestamp_indices: List[int]) -> Dict[str, torch.Tensor]:
            B = len(batch_timestamp_indices)
            T = self.hparams.lookback_window

            list_features = []
            list_stats = []
            list_labels_trade = []
            list_labels_dir = []

            for i, ts_idx in enumerate(batch_timestamp_indices):
                if is_train:
                    valid_coin_indices_at_time = np.where(self.mask_aligned[:, ts_idx])[
                        0
                    ]
                    coin_indices = np.random.choice(
                        valid_coin_indices_at_time,
                        self.hparams.portfolio_size,
                        replace=False,
                    )
                    coin_indices.sort()
                else:
                    coin_indices = self.val_coin_indices
                    if not np.all(self.mask_aligned[coin_indices, ts_idx]):
                        valid_coin_indices_at_time = np.where(
                            self.mask_aligned[:, ts_idx]
                        )[0]
                        coin_indices = np.random.choice(
                            valid_coin_indices_at_time,
                            self.hparams.portfolio_size,
                            replace=False,
                        )
                        coin_indices.sort()

                features_slice = self.all_features_aligned[
                    coin_indices, (ts_idx - T + 1) : (ts_idx + 1), :
                ]
                stats_slice = self.all_stats_aligned[coin_indices, ts_idx, :]
                labels_trade_slice = self.all_labels_trade_aligned[coin_indices, ts_idx]
                labels_dir_slice = self.all_labels_dir_aligned[coin_indices, ts_idx]

                list_features.append(features_slice)
                list_stats.append(stats_slice)
                list_labels_trade.append(labels_trade_slice)
                list_labels_dir.append(labels_dir_slice)

            batch_features = np.stack(list_features)
            batch_stats = np.stack(list_stats)
            batch_labels_trade = np.stack(list_labels_trade)
            batch_labels_dir = np.stack(list_labels_dir)

            self._normalize_batch(batch_features, batch_stats)

            return {
                "features": torch.from_numpy(batch_features),
                "labels_trade": torch.from_numpy(batch_labels_trade),
                "labels_dir": torch.from_numpy(batch_labels_dir),
            }

        return collate_fn

    # --- DataLoader Definitions ---

    def train_dataloader(self) -> DataLoader:
        """Creates the training DataLoader."""
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
        """Creates the validation DataLoader."""
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
