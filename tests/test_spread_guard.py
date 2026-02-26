"""Tests for Spread Guard — anomaly detection and trading block."""

import time
import numpy as np
import pytest

from src.execution.spread_guard import SpreadGuard


class TestSpreadGuard:
    def setup_method(self):
        self.guard = SpreadGuard(
            zscore_threshold=3.0,
            baseline_window=100,
            cooldown_seconds=1.0,
            max_slippage_points=5.0,
            max_spread_points=8.0,
        )

    def _fill_baseline(self, n=50, mean=3.0, std=0.5):
        """Fill spread history with normal spreads."""
        for _ in range(n):
            self.guard.update_spread(np.random.normal(mean, std))

    def test_can_trade_initially(self):
        assert self.guard.can_trade() is True

    def test_normal_spread_no_block(self):
        self._fill_baseline()
        alert = self.guard.update_spread(3.5)
        assert alert is None
        assert self.guard.can_trade() is True

    def test_absolute_spread_limit(self):
        self._fill_baseline()
        alert = self.guard.update_spread(9.0)  # Above max_spread_points=8
        assert alert is not None
        assert alert.alert_type == "spread_spike"
        assert self.guard.can_trade() is False

    def test_zscore_spike(self):
        self._fill_baseline(n=100, mean=3.0, std=0.3)
        # Inject extreme spread
        alert = self.guard.update_spread(10.0)  # Way above z-score threshold
        assert alert is not None
        assert self.guard.can_trade() is False

    def test_cooldown_auto_release(self):
        self._fill_baseline()
        self.guard.update_spread(9.0)  # Trigger block
        assert self.guard.can_trade() is False

        # Wait for cooldown (1 second)
        time.sleep(1.1)
        assert self.guard.can_trade() is True

    def test_slippage_detection(self):
        # Record several normal slippages
        for _ in range(5):
            self.guard.record_slippage(1.0)

        # Record extreme slippage
        alert = self.guard.record_slippage(7.0)  # Above max_slippage_points=5
        assert alert is not None
        assert alert.alert_type == "slippage_anomaly"

    def test_force_unblock(self):
        self._fill_baseline()
        self.guard.update_spread(9.0)
        assert self.guard.can_trade() is False

        self.guard.force_unblock()
        assert self.guard.can_trade() is True

    def test_stats(self):
        self._fill_baseline(n=50)
        stats = self.guard.get_stats()
        assert "spread_mean" in stats
        assert "is_blocked" in stats
        assert stats["spread_mean"] > 0

    def test_baseline_spread(self):
        self._fill_baseline(n=50, mean=3.0)
        baseline = self.guard.get_baseline_spread()
        assert 2.0 < baseline < 4.0  # Should be close to 3.0
