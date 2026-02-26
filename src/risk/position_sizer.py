"""
Position sizing using Kelly Criterion and RL-adjusted scaling.
Calculates optimal lot size based on win rate, risk/reward, and equity.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("FxBot.PositionSizer")


class PositionSizer:
    """
    Calculates optimal position size for each trade.

    Methods:
    1. Kelly Criterion (half-Kelly for conservative approach)
    2. Fixed lot size
    3. RL-adjusted (Kelly base × RL confidence multiplier)
    """

    def __init__(
        self,
        method: str = "kelly",
        kelly_fraction: float = 0.5,
        fixed_lot: float = 0.01,
        min_lot: float = 0.01,
        max_lot: float = 1.0,
        lot_step: float = 0.01,
        pip_value_per_lot: float = 1.0,
        point: float = 0.01,
    ):
        self.method = method
        self.kelly_fraction = kelly_fraction
        self.fixed_lot = fixed_lot
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.lot_step = lot_step
        self.pip_value = pip_value_per_lot
        self.point = point

        # Running statistics for Kelly calculation
        self._win_count = 0
        self._loss_count = 0
        self._total_wins = 0.0
        self._total_losses = 0.0

    def calculate_lot_size(
        self,
        equity: float,
        sl_distance_price: float,
        confidence: float = 1.0,
        max_risk_amount: float | None = None,
    ) -> float:
        """
        Calculate optimal lot size.

        Args:
            equity: Current account equity.
            sl_distance_price: Stop loss distance in price units.
            confidence: AI confidence score (0-1), used to scale Kelly.
            max_risk_amount: Maximum dollar risk per trade.

        Returns:
            Lot size (rounded to lot_step).
        """
        if self.method == "fixed":
            return self._clamp_lot(self.fixed_lot)

        if self.method == "kelly":
            return self._kelly_lot(equity, sl_distance_price, confidence, max_risk_amount)

        # Default: fixed
        return self._clamp_lot(self.fixed_lot)

    def _kelly_lot(
        self,
        equity: float,
        sl_distance_price: float,
        confidence: float = 1.0,
        max_risk_amount: float | None = None,
    ) -> float:
        """
        Kelly Criterion position sizing.

        f* = (p × b - q) / b
        where:
            p = win probability
            b = avg_win / avg_loss (reward-to-risk ratio)
            q = 1 - p

        lot = (f* × kelly_fraction × confidence × equity) / (sl_distance × pip_value_per_point)
        """
        b = self._get_reward_risk_ratio()
        p = self._get_win_rate()
        q = 1.0 - p

        # Kelly fraction
        if b <= 0:
            kelly_f = 0.0
        else:
            kelly_f = max(0.0, (p * b - q) / b)

        # Apply fractional Kelly (conservative) and confidence scaling
        adjusted_f = kelly_f * self.kelly_fraction * confidence

        # Calculate dollar risk
        risk_dollars = equity * adjusted_f
        if max_risk_amount is not None:
            risk_dollars = min(risk_dollars, max_risk_amount)

        # Convert to lot size
        if sl_distance_price <= 0 or self.pip_value <= 0:
            return self._clamp_lot(self.min_lot)

        sl_points = sl_distance_price / self.point
        risk_per_lot = sl_points * self.pip_value
        lot = risk_dollars / max(risk_per_lot, 0.01)

        logger.debug(
            f"Kelly sizing | f*={kelly_f:.4f} | adj_f={adjusted_f:.4f} | "
            f"risk=${risk_dollars:.2f} | lot={lot:.3f} | "
            f"p={p:.2f} | b={b:.2f}"
        )

        return self._clamp_lot(lot)

    # ------------------------------------------------------------------
    # Statistics Update
    # ------------------------------------------------------------------

    def record_trade(self, pnl: float) -> None:
        """Record a trade result for Kelly calculation updates."""
        if pnl > 0:
            self._win_count += 1
            self._total_wins += pnl
        else:
            self._loss_count += 1
            self._total_losses += abs(pnl)

    def _get_win_rate(self) -> float:
        """Get current win rate (defaults to 0.5 with few trades)."""
        total = self._win_count + self._loss_count
        if total < 10:
            return 0.5  # Prior: assume 50% win rate
        return self._win_count / total

    def _get_reward_risk_ratio(self) -> float:
        """Get average win/loss ratio (defaults to 1.0)."""
        if self._win_count == 0 or self._loss_count == 0:
            return 1.0  # Prior: assume 1:1 ratio
        avg_win = self._total_wins / self._win_count
        avg_loss = self._total_losses / self._loss_count
        if avg_loss == 0:
            return 1.0
        return avg_win / avg_loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _clamp_lot(self, lot: float) -> float:
        """Clamp lot size to valid range and round to lot_step."""
        lot = max(self.min_lot, min(lot, self.max_lot))
        lot = round(lot / self.lot_step) * self.lot_step
        return round(lot, 2)

    def get_sizing_stats(self) -> dict[str, Any]:
        """Get position sizing statistics."""
        return {
            "method": self.method,
            "win_rate": self._get_win_rate(),
            "reward_risk_ratio": self._get_reward_risk_ratio(),
            "kelly_fraction": self.kelly_fraction,
            "total_recorded_trades": self._win_count + self._loss_count,
        }
