"""Tests for rule-based market structure filters."""

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.filters.market_filter import MarketFilter


def _make_h1_df(n: int = 100, base: float = 5180.0, trend: str = "up") -> pd.DataFrame:
    """Create synthetic H1 OHLCV data."""
    np.random.seed(42)
    if trend == "up":
        closes = base + np.cumsum(np.random.uniform(0.5, 2.0, n))
    elif trend == "down":
        closes = base - np.cumsum(np.random.uniform(0.5, 2.0, n))
    else:  # flat
        closes = base + np.random.normal(0, 0.2, n)

    idx = pd.date_range("2026-02-20", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": closes - np.random.uniform(-0.5, 0.5, n),
        "high": closes + np.abs(np.random.normal(1.0, 0.5, n)),
        "low": closes - np.abs(np.random.normal(1.0, 0.5, n)),
        "close": closes,
        "volume": np.random.randint(50, 200, n),
    }, index=idx)


def _make_m5_df(n: int = 200, rsi_target: float = 60.0) -> pd.DataFrame:
    """Create synthetic M5 data with approximate RSI."""
    np.random.seed(42)
    if rsi_target > 50:
        changes = np.random.normal(0.3, 0.5, n)  # Bias up
    else:
        changes = np.random.normal(-0.3, 0.5, n)  # Bias down
    closes = 5180.0 + np.cumsum(changes)
    idx = pd.date_range("2026-02-26", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": closes - np.random.uniform(-0.2, 0.2, n),
        "high": closes + np.abs(np.random.normal(0.5, 0.3, n)),
        "low": closes - np.abs(np.random.normal(0.5, 0.3, n)),
        "close": closes,
        "volume": np.random.randint(20, 100, n),
    }, index=idx)


def _make_m1_df(n: int = 500) -> pd.DataFrame:
    """Create synthetic M1 data with normal volatility."""
    np.random.seed(42)
    closes = 5180.0 + np.cumsum(np.random.normal(0, 0.3, n))
    idx = pd.date_range("2026-02-26", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "open": closes - np.random.uniform(-0.1, 0.1, n),
        "high": closes + np.abs(np.random.normal(0.3, 0.2, n)),
        "low": closes - np.abs(np.random.normal(0.3, 0.2, n)),
        "close": closes,
        "volume": np.random.randint(10, 80, n),
    }, index=idx)


class TestMarketFilter(unittest.TestCase):

    def setUp(self):
        self.mf = MarketFilter(
            ema_period=20,
            session_start_utc=7,
            session_end_utc=21,
        )

    # ---- HTF Trend ----

    def test_buy_allowed_in_uptrend(self):
        h1 = _make_h1_df(100, trend="up")
        ok, reason = self.mf.check_htf_trend(h1, direction=1)
        self.assertTrue(ok, reason)

    def test_buy_blocked_in_downtrend(self):
        h1 = _make_h1_df(100, trend="down")
        ok, reason = self.mf.check_htf_trend(h1, direction=1)
        self.assertFalse(ok)
        self.assertIn("HTF_TREND", reason)

    def test_sell_allowed_in_downtrend(self):
        h1 = _make_h1_df(100, trend="down")
        ok, reason = self.mf.check_htf_trend(h1, direction=-1)
        self.assertTrue(ok, reason)

    def test_sell_blocked_in_uptrend(self):
        h1 = _make_h1_df(100, trend="up")
        ok, reason = self.mf.check_htf_trend(h1, direction=-1)
        self.assertFalse(ok)
        self.assertIn("HTF_TREND", reason)

    def test_trend_filter_empty_df(self):
        ok, _ = self.mf.check_htf_trend(pd.DataFrame(), direction=1)
        self.assertTrue(ok)  # Fail-open

    # ---- Session Filter ----

    def test_session_allowed_during_london(self):
        with patch("src.filters.market_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, 10, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            ok, _ = self.mf.check_session()
            self.assertTrue(ok)

    def test_session_blocked_during_asian(self):
        with patch("src.filters.market_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, 3, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            ok, reason = self.mf.check_session()
            self.assertFalse(ok)
            self.assertIn("SESSION", reason)

    # ---- Momentum ----

    def test_momentum_buy_with_bullish_rsi(self):
        m5 = _make_m5_df(200, rsi_target=60)
        ok, _ = self.mf.check_momentum(m5, direction=1)
        self.assertTrue(ok)

    def test_momentum_sell_with_bearish_rsi(self):
        m5 = _make_m5_df(200, rsi_target=40)
        ok, _ = self.mf.check_momentum(m5, direction=-1)
        self.assertTrue(ok)

    # ---- Volatility ----

    def test_normal_volatility_passes(self):
        m1 = _make_m1_df(500)
        ok, _ = self.mf.check_volatility(m1)
        self.assertTrue(ok)

    def test_volatility_empty_df(self):
        ok, _ = self.mf.check_volatility(pd.DataFrame())
        self.assertTrue(ok)  # Fail-open

    # ---- Combined ----

    def test_check_all_passes_good_conditions(self):
        """All filters should pass in an uptrend during London with good momentum."""
        h1 = _make_h1_df(100, trend="up")
        m5 = _make_m5_df(200, rsi_target=60)
        m1 = _make_m1_df(500)

        with patch("src.filters.market_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, 14, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            ok, reason = self.mf.check_all(h1, m5, m1, direction=1)
            self.assertTrue(ok, reason)

    def test_check_all_blocked_by_session(self):
        h1 = _make_h1_df(100, trend="up")
        m5 = _make_m5_df(200, rsi_target=60)
        m1 = _make_m1_df(500)

        with patch("src.filters.market_filter.datetime") as mock_dt:
            mock_dt.now.return_value = datetime(2026, 2, 26, 3, 0, tzinfo=timezone.utc)
            mock_dt.side_effect = lambda *a, **k: datetime(*a, **k)
            ok, reason = self.mf.check_all(h1, m5, m1, direction=1)
            self.assertFalse(ok)
            self.assertIn("SESSION", reason)


if __name__ == "__main__":
    unittest.main()
