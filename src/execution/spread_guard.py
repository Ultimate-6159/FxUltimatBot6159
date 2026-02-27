"""
Dynamic Spread & Slippage Filter (Broker-Proof).
Detects abnormal spread widening and slippage to prevent trading during
broker manipulation or high-volatility events.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("FxBot.SpreadGuard")


@dataclass
class SpreadAlert:
    """Alert event when spread anomaly detected."""
    timestamp: float
    alert_type: str          # "spread_spike", "sustained_wide", "slippage_anomaly"
    current_value: float
    threshold: float
    zscore: float
    message: str


class SpreadGuard:
    """
    Dynamic Spread & Slippage Filter.

    Protects against:
    1. Spread spikes — sudden widening (Z-score detection)
    2. Sustained wide spread — mean shift detection
    3. Slippage anomalies — tracking fill quality over time

    When triggered:
    - Trading is BLOCKED until spread normalizes
    - Cooldown period prevents premature re-entry
    - All alerts are logged for post-analysis
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        baseline_window: int = 1000,
        cooldown_seconds: float = 5.0,
        max_slippage_points: float = 5.0,
        max_spread_points: float = 50.0,
    ):
        self.zscore_threshold = zscore_threshold
        self.baseline_window = baseline_window
        self.cooldown_seconds = cooldown_seconds
        self.max_slippage_points = max_slippage_points
        self.max_spread_points = max_spread_points

        # Rolling spread history
        self._spread_history: deque[float] = deque(maxlen=baseline_window)

        # Slippage tracking
        self._slippage_history: deque[float] = deque(maxlen=100)

        # State
        self._blocked = False
        self._block_until: float = 0.0
        self._alerts: list[SpreadAlert] = []

    @property
    def is_blocked(self) -> bool:
        """Check if trading is currently blocked."""
        now = time.time()
        if self._blocked and now >= self._block_until:
            self._blocked = False
            logger.info("SpreadGuard: Block lifted, trading re-enabled")
        return self._blocked

    def can_trade(self) -> bool:
        """Returns True if conditions are safe to trade."""
        return not self.is_blocked

    # ------------------------------------------------------------------
    # Spread Monitoring
    # ------------------------------------------------------------------

    def update_spread(self, spread_points: float) -> SpreadAlert | None:
        """
        Update with latest spread value and check for anomalies.

        Args:
            spread_points: Current spread in points.

        Returns:
            SpreadAlert if anomaly detected, None otherwise.
        """
        self._spread_history.append(spread_points)

        # Need minimum history for baseline
        if len(self._spread_history) < 30:
            return None

        spreads = np.array(self._spread_history)

        # --- Check 1: Absolute spread limit ---
        if spread_points > self.max_spread_points:
            return self._trigger_block(
                "spread_spike",
                spread_points,
                self.max_spread_points,
                zscore=0.0,
                message=f"Spread {spread_points:.1f} exceeds absolute maximum {self.max_spread_points:.1f}",
            )

        # --- Check 2: Z-score anomaly ---
        mean = np.mean(spreads)
        std = np.std(spreads)
        if std > 0:
            zscore = (spread_points - mean) / std
            if zscore > self.zscore_threshold:
                return self._trigger_block(
                    "spread_spike",
                    spread_points,
                    mean + self.zscore_threshold * std,
                    zscore=zscore,
                    message=f"Spread Z-score {zscore:.2f} > {self.zscore_threshold} "
                            f"(spread={spread_points:.1f}, μ={mean:.1f}, σ={std:.1f})",
                )

        # --- Check 3: Sustained wide spread (mean shift) ---
        if len(spreads) >= 100:
            recent_mean = np.mean(spreads[-20:])
            baseline_mean = np.mean(spreads[-100:])
            if baseline_mean > 0 and recent_mean / baseline_mean > 2.0:
                return self._trigger_block(
                    "sustained_wide",
                    recent_mean,
                    baseline_mean * 2.0,
                    zscore=0.0,
                    message=f"Sustained wide spread: recent={recent_mean:.1f} vs baseline={baseline_mean:.1f}",
                )

        return None

    # ------------------------------------------------------------------
    # Slippage Monitoring
    # ------------------------------------------------------------------

    def record_slippage(self, slippage_points: float) -> SpreadAlert | None:
        """
        Record slippage from a filled order and check for anomalies.

        Args:
            slippage_points: Slippage in points (positive = unfavorable).

        Returns:
            SpreadAlert if slippage is abnormal.
        """
        self._slippage_history.append(abs(slippage_points))

        if len(self._slippage_history) < 5:
            return None

        # Check if individual slippage is too high
        if abs(slippage_points) > self.max_slippage_points:
            return self._trigger_block(
                "slippage_anomaly",
                abs(slippage_points),
                self.max_slippage_points,
                zscore=0.0,
                message=f"High slippage: {abs(slippage_points):.1f} pts > max {self.max_slippage_points:.1f}",
            )

        # Check if average slippage is trending up
        recent_avg = np.mean(list(self._slippage_history)[-10:])
        if recent_avg > self.max_slippage_points * 0.8:
            return self._trigger_block(
                "slippage_anomaly",
                recent_avg,
                self.max_slippage_points * 0.8,
                zscore=0.0,
                message=f"Average slippage trending high: {recent_avg:.1f} pts",
            )

        return None

    # ------------------------------------------------------------------
    # Block Control
    # ------------------------------------------------------------------

    def _trigger_block(
        self,
        alert_type: str,
        current_value: float,
        threshold: float,
        zscore: float,
        message: str,
    ) -> SpreadAlert:
        """Trigger trading block and create alert."""
        now = time.time()
        self._blocked = True
        self._block_until = now + self.cooldown_seconds

        alert = SpreadAlert(
            timestamp=now,
            alert_type=alert_type,
            current_value=current_value,
            threshold=threshold,
            zscore=zscore,
            message=message,
        )
        self._alerts.append(alert)

        logger.warning(
            f"⚠ SPREAD GUARD TRIGGERED | {alert_type} | {message} | "
            f"Trading blocked for {self.cooldown_seconds}s"
        )
        return alert

    def force_unblock(self) -> None:
        """Manually unblock trading."""
        self._blocked = False
        self._block_until = 0
        logger.info("SpreadGuard: Manually unblocked")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get spread guard statistics."""
        spreads = np.array(self._spread_history) if self._spread_history else np.array([0])
        slippages = np.array(self._slippage_history) if self._slippage_history else np.array([0])

        return {
            "is_blocked": self.is_blocked,
            "total_alerts": len(self._alerts),
            "spread_mean": float(np.mean(spreads)),
            "spread_std": float(np.std(spreads)),
            "spread_max": float(np.max(spreads)),
            "slippage_mean": float(np.mean(slippages)),
            "slippage_max": float(np.max(slippages)),
            "recent_alerts": [
                {"type": a.alert_type, "message": a.message}
                for a in self._alerts[-5:]
            ],
        }

    def get_baseline_spread(self) -> float:
        """Get current baseline (mean) spread."""
        if not self._spread_history:
            return 0.0
        return float(np.mean(self._spread_history))
