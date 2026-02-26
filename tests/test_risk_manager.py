"""Tests for Risk Manager — drawdown control and circuit breaker."""

import pytest

from src.risk.risk_manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            initial_equity=10000.0,
            max_daily_drawdown_pct=5.0,
            max_trade_risk_pct=1.5,
            max_consecutive_losses=3,
            cooldown_minutes=0.01,  # Very short for testing
            global_max_drawdown_pct=15.0,
        )

    def test_initial_can_trade(self):
        can, reason = self.rm.can_trade()
        assert can is True
        assert reason == "OK"

    def test_max_trade_risk_amount(self):
        risk = self.rm.max_trade_risk_amount()
        assert risk == 150.0  # 1.5% of 10000

    def test_winning_trade(self):
        self.rm.on_trade_result(50.0)
        assert self.rm.state.consecutive_losses == 0
        assert self.rm.state.current_equity == 10050.0

    def test_losing_trade(self):
        self.rm.on_trade_result(-100.0)
        assert self.rm.state.consecutive_losses == 1
        assert self.rm.state.current_equity == 9900.0

    def test_consecutive_loss_cooldown(self):
        # 3 consecutive losses → cooldown
        self.rm.on_trade_result(-50.0)
        self.rm.on_trade_result(-50.0)
        self.rm.on_trade_result(-50.0)

        can, reason = self.rm.can_trade()
        assert can is False
        assert "Cooldown" in reason

    def test_daily_drawdown_halt(self):
        # Lose more than 5% in one session
        self.rm.on_trade_result(-500.0)  # 5% of 10000
        can, reason = self.rm.can_trade()
        assert can is False
        assert "Daily drawdown" in reason or "HALTED" in reason

    def test_win_resets_consecutive_losses(self):
        self.rm.on_trade_result(-50.0)
        self.rm.on_trade_result(-50.0)
        self.rm.on_trade_result(100.0)  # Win resets counter
        assert self.rm.state.consecutive_losses == 0

    def test_reset_daily(self):
        self.rm.on_trade_result(-100.0)
        self.rm.reset_daily()
        assert self.rm.state.daily_pnl == 0.0
        assert self.rm.state.total_trades_today == 0
        can, _ = self.rm.can_trade()
        assert can is True

    def test_emergency_resume(self):
        self.rm.on_trade_result(-600.0)  # Trigger halt
        can, _ = self.rm.can_trade()
        assert can is False

        self.rm.emergency_resume()
        can, _ = self.rm.can_trade()
        assert can is True

    def test_risk_report(self):
        self.rm.on_trade_result(100.0)
        report = self.rm.get_risk_report()
        assert "current_equity" in report
        assert report["current_equity"] == 10100.0
        assert report["total_trades_today"] == 1
