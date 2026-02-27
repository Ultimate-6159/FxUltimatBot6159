"""
Risk Management — Global drawdown control and circuit breaker.
Protects equity with multiple layers of risk limits.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("FxBot.RiskManager")


@dataclass
class RiskState:
    """Current risk management state."""
    initial_balance: float = 10000.0
    current_equity: float = 10000.0
    daily_start_equity: float = 10000.0
    peak_equity: float = 10000.0
    daily_pnl: float = 0.0
    total_trades_today: int = 0
    consecutive_losses: int = 0
    is_halted: bool = False
    halt_reason: str = ""
    cooldown_until: float = 0.0


class RiskManager:
    """
    Multi-layer risk management system.

    Layers:
    1. Per-Trade Risk — max loss per single trade
    2. Daily Drawdown — max total losses in one day
    3. Consecutive Loss Cooldown — pause after N consecutive losses
    4. Global Circuit Breaker — STOP everything if equity drops too far
    5. Equity monitoring — track peak equity and drawdown in real-time
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        max_daily_drawdown_pct: float = 5.0,
        max_trade_risk_pct: float = 1.5,
        max_consecutive_losses: int = 5,
        cooldown_minutes: float = 30.0,
        global_max_drawdown_pct: float = 15.0,
    ):
        self.max_daily_dd_pct = max_daily_drawdown_pct
        self.max_trade_risk_pct = max_trade_risk_pct
        self.max_consec_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.global_max_dd_pct = global_max_drawdown_pct
        self._lock = threading.Lock()

        self._state = RiskState(
            initial_balance=initial_equity,
            current_equity=initial_equity,
            daily_start_equity=initial_equity,
            peak_equity=initial_equity,
        )

    @property
    def state(self) -> RiskState:
        return self._state

    # ------------------------------------------------------------------
    # Pre-Trade Check
    # ------------------------------------------------------------------

    def can_trade(self) -> tuple[bool, str]:
        """
        Check all risk conditions before allowing a trade.

        Returns:
            (bool, reason) — True if trade allowed, False with reason.
        """
        with self._lock:
            now = time.time()

            # Check 1: Global halt
            if self._state.is_halted:
                return False, f"HALTED: {self._state.halt_reason}"

            # Check 2: Cooldown after consecutive losses
            if now < self._state.cooldown_until:
                remaining = int(self._state.cooldown_until - now)
                return False, f"Cooldown active: {remaining}s remaining"

            # Check 3: Daily drawdown
            daily_dd_pct = self._get_daily_drawdown_pct()
            if daily_dd_pct >= self.max_daily_dd_pct:
                self._halt(f"Daily drawdown {daily_dd_pct:.1f}% >= {self.max_daily_dd_pct}%")
                return False, self._state.halt_reason

            # Check 4: Global drawdown
            global_dd_pct = self._get_global_drawdown_pct()
            if global_dd_pct >= self.global_max_dd_pct:
                self._halt(f"Global drawdown {global_dd_pct:.1f}% >= {self.global_max_dd_pct}%")
                return False, self._state.halt_reason

            return True, "OK"

    def max_trade_risk_amount(self) -> float:
        """Get maximum dollar amount at risk for a single trade."""
        with self._lock:
            return self._state.current_equity * (self.max_trade_risk_pct / 100.0)

    # ------------------------------------------------------------------
    # Trade Events
    # ------------------------------------------------------------------

    def on_trade_result(self, pnl: float) -> None:
        """
        Update risk state after a trade closes.

        Args:
            pnl: Realized P&L of the trade (positive = profit).
        """
        with self._lock:
            self._state.daily_pnl += pnl
            self._state.current_equity += pnl
            self._state.total_trades_today += 1

            # Update peak equity
            if self._state.current_equity > self._state.peak_equity:
                self._state.peak_equity = self._state.current_equity

            if pnl < 0:
                self._state.consecutive_losses += 1
                consec = self._state.consecutive_losses
                daily_dd = self._get_daily_drawdown_pct()

                # Check consecutive loss limit
                if self._state.consecutive_losses >= self.max_consec_losses:
                    self._state.cooldown_until = time.time() + self.cooldown_minutes * 60
            else:
                self._state.consecutive_losses = 0
                equity = self._state.current_equity

        if pnl < 0:
            logger.info(
                f"LOSS | PnL={pnl:.2f} | consecutive={consec} | "
                f"daily_dd={daily_dd:.1f}%"
            )
            if consec >= self.max_consec_losses:
                logger.warning(
                    f"COOLDOWN ACTIVATED: {consec} consecutive losses | "
                    f"Waiting {self.cooldown_minutes} minutes"
                )
        else:
            logger.info(f"WIN | PnL=+{pnl:.2f} | equity={equity:.2f}")

    def update_equity(self, current_equity: float) -> None:
        """Update current equity (call periodically with account equity)."""
        with self._lock:
            self._state.current_equity = current_equity
            if current_equity > self._state.peak_equity:
                self._state.peak_equity = current_equity

    def reset_daily(self) -> None:
        """Reset daily counters (call at start of each trading day)."""
        with self._lock:
            self._state.daily_start_equity = self._state.current_equity
            self._state.daily_pnl = 0.0
            self._state.total_trades_today = 0
            self._state.is_halted = False
            self._state.halt_reason = ""
            equity = self._state.current_equity
        logger.info(f"Daily risk reset | Equity: {equity:.2f}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_daily_drawdown_pct(self) -> float:
        """Calculate daily drawdown percentage."""
        if self._state.daily_start_equity <= 0:
            return 0.0
        dd = max(0, self._state.daily_start_equity - self._state.current_equity)
        return (dd / self._state.daily_start_equity) * 100.0

    def _get_global_drawdown_pct(self) -> float:
        """Calculate global drawdown from peak equity."""
        if self._state.peak_equity <= 0:
            return 0.0
        dd = max(0, self._state.peak_equity - self._state.current_equity)
        return (dd / self._state.peak_equity) * 100.0

    def _halt(self, reason: str) -> None:
        """Halt all trading."""
        self._state.is_halted = True
        self._state.halt_reason = reason
        logger.critical(f"⛔ TRADING HALTED: {reason}")

    def emergency_resume(self) -> None:
        """Manually resume trading after halt (use with caution)."""
        with self._lock:
            self._state.is_halted = False
            self._state.halt_reason = ""
            self._state.consecutive_losses = 0
            self._state.cooldown_until = 0
            self._state.daily_start_equity = self._state.current_equity
            self._state.daily_pnl = 0.0
        logger.warning("Trading manually resumed after halt")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_risk_report(self) -> dict[str, Any]:
        """Get comprehensive risk report."""
        with self._lock:
            return {
                "current_equity": self._state.current_equity,
                "peak_equity": self._state.peak_equity,
                "daily_pnl": self._state.daily_pnl,
                "daily_drawdown_pct": self._get_daily_drawdown_pct(),
                "global_drawdown_pct": self._get_global_drawdown_pct(),
                "consecutive_losses": self._state.consecutive_losses,
                "total_trades_today": self._state.total_trades_today,
                "is_halted": self._state.is_halted,
                "halt_reason": self._state.halt_reason,
                "max_trade_risk": self._state.current_equity * (self.max_trade_risk_pct / 100.0),
            }
