"""
Custom feature engineering pipeline.
Generates non-standard Price Action features (no raw RSI/MACD).
Uses ATR-normalized metrics, candle morphology, and microstructure signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("FxBot.FeatureEngine")


class FeatureEngine:
    """
    Builds a rich feature set from raw OHLCV + tick data.

    Design philosophy: AVOID standard indicator signals (RSI, MACD, Bollinger)
    that every bot uses. Instead, compute custom Price Action and
    microstructure features that are harder for brokers/market makers to predict.

    Feature Categories:
    1. Candle Morphology — body/wick ratios, engulfing patterns
    2. Price Dynamics — ATR-normalized velocity and acceleration
    3. Volume Profile — relative volume, volume momentum
    4. Microstructure — spread dynamics, tick intensity
    5. Volatility Regime — realized vs baseline volatility
    6. Cross-TF Divergence — momentum alignment across timeframes
    """

    def __init__(self, lookback: int = 60):
        self.lookback = lookback

    def compute_all_features(
        self,
        df: pd.DataFrame,
        spreads: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Compute all feature columns from OHLCV DataFrame.

        Args:
            df: DataFrame with columns [open, high, low, close, volume].
            spreads: Optional array of spread values (aligned with df index).

        Returns:
            DataFrame with all computed features, z-score normalized.
        """
        features = pd.DataFrame(index=df.index)

        # 1. Candle Morphology
        features = pd.concat([features, self._candle_morphology(df)], axis=1)

        # 2. Price Dynamics
        features = pd.concat([features, self._price_dynamics(df)], axis=1)

        # 3. Volume Profile
        features = pd.concat([features, self._volume_profile(df)], axis=1)

        # 4. Microstructure
        if spreads is not None:
            features = pd.concat([features, self._microstructure(df, spreads)], axis=1)

        # 5. Volatility Regime
        features = pd.concat([features, self._volatility_regime(df)], axis=1)

        # 6. Cross-TF features are added externally via multi_tf_loader

        # Fill NaN and normalize
        features = features.fillna(0.0)
        features = self._zscore_normalize(features)

        return features

    # ------------------------------------------------------------------
    # 1. Candle Morphology
    # ------------------------------------------------------------------

    def _candle_morphology(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Candle shape analysis — captures rejection, absorption, and momentum candles.
        """
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        full_range = (h - l).replace(0, np.nan)
        body = (c - o).abs()

        features = pd.DataFrame(index=df.index)

        # Body ratio (0=doji, 1=marubozu)
        features["body_ratio"] = body / full_range

        # Upper wick ratio — rejection from highs
        features["upper_wick_ratio"] = (h - np.maximum(o, c)) / full_range

        # Lower wick ratio — rejection from lows
        features["lower_wick_ratio"] = (np.minimum(o, c) - l) / full_range

        # Bullish/bearish direction
        features["candle_direction"] = np.sign(c - o)

        # Engulfing pattern (current body > previous body and opposite direction)
        prev_body = body.shift(1)
        prev_dir = np.sign(c.shift(1) - o.shift(1))
        curr_dir = np.sign(c - o)
        features["engulfing"] = ((body > prev_body) & (curr_dir != prev_dir)).astype(float)

        # Rejection wick: one-sided wick > 60% of range
        features["rejection_top"] = (features["upper_wick_ratio"] > 0.6).astype(float)
        features["rejection_bottom"] = (features["lower_wick_ratio"] > 0.6).astype(float)

        # Consecutive same-direction candles
        features["consecutive_dir"] = (
            curr_dir.groupby((curr_dir != curr_dir.shift()).cumsum()).cumcount() + 1
        ) * curr_dir

        return features

    # ------------------------------------------------------------------
    # 2. Price Dynamics
    # ------------------------------------------------------------------

    def _price_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ATR-normalized price velocity and acceleration.
        """
        c = df["close"]
        h, l = df["high"], df["low"]

        features = pd.DataFrame(index=df.index)

        # ATR (14-period)
        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        features["atr"] = atr

        # Price velocity (1-bar change normalized by ATR)
        raw_velocity = c.diff()
        features["velocity_1"] = raw_velocity / atr.replace(0, np.nan)

        # Price velocity (5-bar)
        features["velocity_5"] = c.diff(5) / atr.replace(0, np.nan)

        # Price acceleration (change in velocity)
        features["acceleration"] = features["velocity_1"].diff()

        # Distance from recent high/low (normalized)
        rolling_high = h.rolling(20).max()
        rolling_low = l.rolling(20).min()
        rolling_range = (rolling_high - rolling_low).replace(0, np.nan)
        features["dist_from_high"] = (rolling_high - c) / rolling_range
        features["dist_from_low"] = (c - rolling_low) / rolling_range

        # Momentum (rate-of-change over 10 bars)
        features["momentum_10"] = c.pct_change(10)

        return features

    # ------------------------------------------------------------------
    # 3. Volume Profile
    # ------------------------------------------------------------------

    def _volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume analysis relative to recent averages.
        """
        v = df["volume"]
        features = pd.DataFrame(index=df.index)

        # Relative volume (vs 20-period mean)
        vol_mean = v.rolling(20).mean()
        features["rel_volume"] = v / vol_mean.replace(0, np.nan)

        # Volume momentum (5-bar change in volume)
        features["vol_momentum"] = v.rolling(5).mean() / vol_mean.replace(0, np.nan)

        # Volume-weighted price (proxy VWAP approximation)
        tp = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (tp * v).rolling(20).sum()
        cum_vol = v.rolling(20).sum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)
        features["dist_from_vwap"] = (df["close"] - vwap) / vwap.replace(0, np.nan)

        # Volume spike detection (> 2x average)
        features["vol_spike"] = (features["rel_volume"] > 2.0).astype(float)

        return features

    # ------------------------------------------------------------------
    # 4. Microstructure
    # ------------------------------------------------------------------

    def _microstructure(self, df: pd.DataFrame, spreads: np.ndarray) -> pd.DataFrame:
        """
        Spread dynamics and tick intensity features.
        """
        features = pd.DataFrame(index=df.index)

        # Align spreads array with DataFrame index
        n = min(len(spreads), len(df))
        spread_series = pd.Series(spreads[-n:], index=df.index[-n:])

        # Spread relative to baseline
        spread_mean = spread_series.rolling(100, min_periods=1).mean()
        spread_std = spread_series.rolling(100, min_periods=1).std()
        features["spread_zscore"] = (spread_series - spread_mean) / spread_std.replace(0, np.nan)

        # Spread momentum
        features["spread_momentum"] = spread_series.diff(5)

        return features

    # ------------------------------------------------------------------
    # 5. Volatility Regime
    # ------------------------------------------------------------------

    def _volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Realized volatility relative to longer-term baseline.
        """
        c = df["close"]
        features = pd.DataFrame(index=df.index)

        # Short-term realized volatility (10-bar)
        returns = c.pct_change()
        short_vol = returns.rolling(10).std() * np.sqrt(252 * 24 * 60)  # Annualized

        # Long-term baseline volatility (100-bar)
        long_vol = returns.rolling(100).std() * np.sqrt(252 * 24 * 60)

        # Volatility ratio
        features["vol_ratio"] = short_vol / long_vol.replace(0, np.nan)

        # Volatility regime state (low < 0.8, normal 0.8-1.2, high > 1.2)
        features["vol_regime"] = pd.cut(
            features["vol_ratio"],
            bins=[0, 0.8, 1.2, float("inf")],
            labels=[0, 1, 2],  # 0=low, 1=normal, 2=high
        ).astype(float)

        # Mean reversion signal (large deviation → expect reversion)
        features["vol_mean_revert"] = -(features["vol_ratio"] - 1.0)

        return features

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _zscore_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score normalize all feature columns using rolling window."""
        normalized = pd.DataFrame(index=df.index)
        for col in df.columns:
            mean = df[col].rolling(self.lookback, min_periods=1).mean()
            std = df[col].rolling(self.lookback, min_periods=1).std()
            normalized[col] = (df[col] - mean) / std.replace(0, np.nan)

        # Clip extreme values and fill NaN
        normalized = normalized.clip(-5, 5).fillna(0.0)
        return normalized

    def get_feature_names(self) -> list[str]:
        """Return list of all feature names."""
        return [
            # Candle Morphology
            "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
            "candle_direction", "engulfing", "rejection_top",
            "rejection_bottom", "consecutive_dir",
            # Price Dynamics
            "atr", "velocity_1", "velocity_5", "acceleration",
            "dist_from_high", "dist_from_low", "momentum_10",
            # Volume Profile
            "rel_volume", "vol_momentum", "dist_from_vwap", "vol_spike",
            # Microstructure
            "spread_zscore", "spread_momentum",
            # Volatility Regime
            "vol_ratio", "vol_regime", "vol_mean_revert",
        ]
