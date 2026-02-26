"""
Multi-timeframe OHLCV data loader.
Fetches M1/M5/M15/H1 candle data from MT5 and builds a unified feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("FxBot.MultiTF")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore

# MT5 timeframe constants mapping
TF_MAP: dict[str, int | None] = {}
if mt5 is not None:
    TF_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
else:
    TF_MAP = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


@dataclass
class MultiTFData:
    """Container for multi-timeframe data."""
    m1: pd.DataFrame
    m5: pd.DataFrame
    m15: pd.DataFrame
    h1: pd.DataFrame
    feature_matrix: np.ndarray | None = None


class MultiTimeframeLoader:
    """
    Loads and aligns OHLCV data across multiple timeframes.

    The key output is a Feature Matrix where each row corresponds to a 1-minute
    candle enriched with context from higher timeframes (M5, M15, H1).
    """

    def __init__(
        self,
        symbol: str = "XAUUSDm",
        timeframes: list[str] | None = None,
        bars_per_tf: int = 500,
    ):
        self.symbol = symbol
        self.timeframes = timeframes or ["M1", "M5", "M15", "H1"]
        self.bars_per_tf = bars_per_tf

    def load_from_mt5(self) -> MultiTFData | None:
        """Load OHLCV data from MT5 for all configured timeframes."""
        if mt5 is None:
            logger.warning("MT5 not available, cannot load live data")
            return None

        frames: dict[str, pd.DataFrame] = {}
        for tf_name in self.timeframes:
            tf_const = TF_MAP.get(tf_name)
            if tf_const is None:
                logger.warning(f"Unknown timeframe: {tf_name}")
                continue

            rates = mt5.copy_rates_from_pos(self.symbol, tf_const, 0, self.bars_per_tf)
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to load {tf_name} data for {self.symbol}")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df.set_index("time", inplace=True)
            df.rename(columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
                "real_volume": "real_volume",
                "spread": "spread",
            }, inplace=True)
            frames[tf_name] = df

        if "M1" not in frames:
            logger.error("M1 data is required but not loaded")
            return None

        data = MultiTFData(
            m1=frames.get("M1", pd.DataFrame()),
            m5=frames.get("M5", pd.DataFrame()),
            m15=frames.get("M15", pd.DataFrame()),
            h1=frames.get("H1", pd.DataFrame()),
        )

        data.feature_matrix = self.build_feature_matrix(data)
        return data

    def load_from_dataframes(
        self, m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, h1: pd.DataFrame
    ) -> MultiTFData:
        """Load from pre-existing DataFrames (for backtesting)."""
        data = MultiTFData(m1=m1, m5=m5, m15=m15, h1=h1)
        data.feature_matrix = self.build_feature_matrix(data)
        return data

    def build_feature_matrix(self, data: MultiTFData) -> np.ndarray:
        """
        Build a unified feature matrix where each row = 1 M1 candle
        enriched with higher-TF context.

        Features per row:
        - M1: open, high, low, close, volume (5)
        - M5 context: close, body_pct, upper_wick_pct, lower_wick_pct (4)
        - M15 context: close, body_pct, trend_direction (3)
        - H1 context: close, range_pct, trend_direction (3)
        Total: 15 raw features per row
        """
        m1 = data.m1.copy()
        if m1.empty:
            return np.array([])

        # M1 base features
        features = m1[["open", "high", "low", "close", "volume"]].copy()
        features.columns = ["m1_open", "m1_high", "m1_low", "m1_close", "m1_volume"]

        # Add higher-TF context via forward-fill merge
        for tf_name, tf_df in [("m5", data.m5), ("m15", data.m15), ("h1", data.h1)]:
            if tf_df.empty:
                continue

            tf_features = pd.DataFrame(index=tf_df.index)
            tf_features[f"{tf_name}_close"] = tf_df["close"]

            # Body percentage (relative candle body size)
            body = (tf_df["close"] - tf_df["open"]).abs()
            full_range = tf_df["high"] - tf_df["low"]
            tf_features[f"{tf_name}_body_pct"] = body / full_range.replace(0, np.nan)

            # Trend direction (-1, 0, +1)
            tf_features[f"{tf_name}_trend"] = np.sign(tf_df["close"] - tf_df["open"])

            # Merge onto M1 index using forward fill (asof merge)
            tf_features = tf_features.reindex(features.index, method="ffill")
            features = pd.concat([features, tf_features], axis=1)

        # Fill NaN and convert
        features = features.fillna(0.0)
        return features.values.astype(np.float32)

    def generate_synthetic_data(self, n_bars: int = 500, base_price: float = 2650.0) -> MultiTFData:
        """Generate synthetic multi-TF data for testing."""
        np.random.seed(42)

        def _make_ohlcv(n: int, step_minutes: int) -> pd.DataFrame:
            prices = [base_price]
            for _ in range(n - 1):
                change = np.random.normal(0, 0.3 * np.sqrt(step_minutes))
                prices.append(prices[-1] + change)
            prices = np.array(prices)

            highs = prices + np.abs(np.random.normal(0.5, 0.3, n))
            lows = prices - np.abs(np.random.normal(0.5, 0.3, n))
            opens = prices + np.random.normal(0, 0.2, n)
            closes = prices + np.random.normal(0, 0.2, n)
            volumes = np.abs(np.random.normal(100, 50, n))

            idx = pd.date_range(
                start="2025-01-01",
                periods=n,
                freq=f"{step_minutes}min",
                tz="UTC",
            )
            return pd.DataFrame({
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
                "real_volume": volumes * 10,
                "spread": np.abs(np.random.normal(3, 1, n)),
            }, index=idx)

        m1 = _make_ohlcv(n_bars, 1)
        m5 = _make_ohlcv(n_bars // 5, 5)
        m15 = _make_ohlcv(n_bars // 15, 15)
        h1 = _make_ohlcv(n_bars // 60, 60)

        data = MultiTFData(m1=m1, m5=m5, m15=m15, h1=h1)
        data.feature_matrix = self.build_feature_matrix(data)

        logger.info(
            f"Synthetic data generated | M1: {len(m1)} bars | "
            f"Feature matrix: {data.feature_matrix.shape if data.feature_matrix is not None else 'N/A'}"
        )
        return data
