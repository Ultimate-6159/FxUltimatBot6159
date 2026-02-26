"""Tests for FeatureEngine — custom Price Action feature generation."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_engine import FeatureEngine


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data."""
    np.random.seed(42)
    n = 200
    prices = 2650 + np.cumsum(np.random.normal(0, 0.3, n))
    highs = prices + np.abs(np.random.normal(0.5, 0.3, n))
    lows = prices - np.abs(np.random.normal(0.5, 0.3, n))
    opens = prices + np.random.normal(0, 0.2, n)
    closes = prices + np.random.normal(0, 0.2, n)
    volumes = np.abs(np.random.normal(100, 50, n))

    return pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    }, index=pd.date_range("2025-01-01", periods=n, freq="1min"))


@pytest.fixture
def engine():
    return FeatureEngine(lookback=60)


class TestFeatureEngine:
    def test_compute_all_features_returns_dataframe(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

    def test_no_nan_in_output(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert result.isnull().sum().sum() == 0, "Features should have no NaN after processing"

    def test_features_are_normalized(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        # Z-score normalized features should be clipped to [-5, 5]
        assert result.max().max() <= 5.0
        assert result.min().min() >= -5.0

    def test_candle_morphology_features(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert "body_ratio" in result.columns
        assert "upper_wick_ratio" in result.columns
        assert "candle_direction" in result.columns

    def test_price_dynamics_features(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert "velocity_1" in result.columns
        assert "atr" in result.columns
        assert "momentum_10" in result.columns

    def test_volume_features(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert "rel_volume" in result.columns
        assert "dist_from_vwap" in result.columns

    def test_volatility_regime(self, engine, sample_ohlcv):
        result = engine.compute_all_features(sample_ohlcv)
        assert "vol_ratio" in result.columns

    def test_with_spread_data(self, engine, sample_ohlcv):
        spreads = np.abs(np.random.normal(3, 1, len(sample_ohlcv)))
        result = engine.compute_all_features(sample_ohlcv, spreads=spreads)
        assert "spread_zscore" in result.columns

    def test_feature_names_list(self, engine):
        names = engine.get_feature_names()
        assert len(names) >= 20
        assert "body_ratio" in names
        assert "velocity_1" in names

    def test_small_dataframe(self, engine):
        """Should handle very small DataFrames without crashing."""
        df = pd.DataFrame({
            "open": [100, 101], "high": [102, 103],
            "low": [99, 100], "close": [101, 102],
            "volume": [100, 200],
        })
        result = engine.compute_all_features(df)
        assert len(result) == 2
