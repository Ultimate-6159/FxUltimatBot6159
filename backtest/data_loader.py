"""
Historical data loader for backtesting.
Loads tick and OHLCV data from MT5 or CSV files.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("FxBot.DataLoader")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


class HistoricalDataLoader:
    """Load historical data for backtesting from various sources."""

    def __init__(self, symbol: str = "XAUUSDm"):
        self.symbol = symbol

    def load_from_mt5(
        self,
        timeframe: str = "M1",
        start: datetime | None = None,
        end: datetime | None = None,
        bars: int = 50000,
    ) -> pd.DataFrame | None:
        """Load historical OHLCV from MT5."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 not available")
            return None

        from src.data.multi_tf_loader import TF_MAP
        tf = TF_MAP.get(timeframe)
        if tf is None:
            logger.error(f"Unknown timeframe: {timeframe}")
            return None

        if start and end:
            rates = mt5.copy_rates_range(self.symbol, tf, start, end)
        else:
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, bars)

        if rates is None or len(rates) == 0:
            logger.error(f"No data loaded for {self.symbol} {timeframe}")
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        logger.info(f"Loaded {len(df)} bars of {self.symbol} {timeframe} from MT5")
        return df

    def load_from_csv(self, filepath: str | Path) -> pd.DataFrame | None:
        """Load OHLCV data from CSV file."""
        path = Path(filepath)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return None

        df = pd.read_csv(path, parse_dates=["time"], index_col="time")
        # Ensure required columns
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.error(f"Missing columns in CSV: {missing}")
            return None

        logger.info(f"Loaded {len(df)} bars from {path.name}")
        return df

    def validate_data(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data quality and report issues."""
        issues: list[str] = []

        # Check for gaps
        if hasattr(df.index, "to_series"):
            time_diffs = df.index.to_series().diff()
            median_diff = time_diffs.median()
            large_gaps = time_diffs[time_diffs > median_diff * 5]
            if len(large_gaps) > 0:
                issues.append(f"{len(large_gaps)} large time gaps detected")

        # Check for NaN
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            issues.append(f"NaN values: {nan_counts[nan_counts > 0].to_dict()}")

        # Check for zero prices
        zero_prices = (df[["open", "high", "low", "close"]] == 0).any().any()
        if zero_prices:
            issues.append("Zero prices detected")

        # Check OHLC consistency
        invalid = (df["high"] < df["low"]).sum()
        if invalid > 0:
            issues.append(f"{invalid} bars where high < low")

        return {
            "total_bars": len(df),
            "start": str(df.index[0]) if len(df) > 0 else "N/A",
            "end": str(df.index[-1]) if len(df) > 0 else "N/A",
            "issues": issues,
            "is_valid": len(issues) == 0,
        }
