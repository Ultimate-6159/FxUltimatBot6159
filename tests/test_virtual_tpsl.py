"""Tests for Virtual TP/SL — Stealth Mode broker-proof system."""

import time
import pytest

from src.execution.virtual_tpsl import VirtualTPSL, VirtualLevel


class TestVirtualTPSL:
    def setup_method(self):
        self.closed_tickets = []
        self.vtpsl = VirtualTPSL(
            close_position_fn=lambda t: self.closed_tickets.append(t),
            monitor_interval_ms=100,
            trailing_enabled=True,
        )

    # --- Level Management ---

    def test_set_levels_long(self):
        level = self.vtpsl.set_levels(
            ticket=1001, direction=1, entry_price=2650.0,
            tp_distance=4.0, sl_distance=3.0,
        )
        assert level.take_profit == 2654.0
        assert level.stop_loss == 2647.0
        assert level.direction == 1

    def test_set_levels_short(self):
        level = self.vtpsl.set_levels(
            ticket=1002, direction=-1, entry_price=2650.0,
            tp_distance=4.0, sl_distance=3.0,
        )
        assert level.take_profit == 2646.0
        assert level.stop_loss == 2653.0

    def test_remove_levels(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 4.0, 3.0)
        self.vtpsl.remove_levels(1001)
        assert self.vtpsl.get_level(1001) is None

    # --- TP/SL Triggering ---

    def test_tp_hit_long(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 4.0, 3.0)  # TP=2654
        events = self.vtpsl.check_price(current_bid=2654.5, current_ask=2655.0)
        assert len(events) == 1
        assert events[0]["event"] == "TP_HIT"
        assert 1001 in self.closed_tickets

    def test_sl_hit_long(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 4.0, 3.0)  # SL=2647
        events = self.vtpsl.check_price(current_bid=2646.5, current_ask=2647.0)
        assert len(events) == 1
        assert events[0]["event"] == "SL_HIT"

    def test_tp_hit_short(self):
        self.vtpsl.set_levels(1002, -1, 2650.0, 4.0, 3.0)  # TP=2646
        events = self.vtpsl.check_price(current_bid=2645.5, current_ask=2645.8)
        assert len(events) == 1
        assert events[0]["event"] == "TP_HIT"

    def test_sl_hit_short(self):
        self.vtpsl.set_levels(1002, -1, 2650.0, 4.0, 3.0)  # SL=2653
        events = self.vtpsl.check_price(current_bid=2653.5, current_ask=2653.8)
        assert len(events) == 1
        assert events[0]["event"] == "SL_HIT"

    def test_no_trigger_within_range(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 4.0, 3.0)
        events = self.vtpsl.check_price(current_bid=2651.0, current_ask=2651.5)
        assert len(events) == 0

    # --- Trailing Stop ---

    def test_trailing_stop_long(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 10.0, 3.0, trailing_distance=2.0)
        # SL starts at 2647.0

        # Price moves up to 2655 → SL should trail to 2653
        self.vtpsl.check_price(current_bid=2655.0, current_ask=2655.5)
        level = self.vtpsl.get_level(1001)
        assert level is not None
        assert level.stop_loss == 2653.0  # 2655 - 2.0

    def test_trailing_never_widens(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 10.0, 3.0, trailing_distance=2.0)

        # Move up then down — SL should not decrease
        self.vtpsl.check_price(current_bid=2655.0, current_ask=2655.5)
        self.vtpsl.check_price(current_bid=2652.0, current_ask=2652.5)
        level = self.vtpsl.get_level(1001)
        assert level is not None
        assert level.stop_loss == 2653.0  # Should stay at 2653, not go back

    # --- Statistics ---

    def test_stats(self):
        self.vtpsl.set_levels(1001, 1, 2650.0, 4.0, 3.0)
        self.vtpsl.check_price(current_bid=2654.5, current_ask=2655.0)  # TP hit

        stats = self.vtpsl.get_stats()
        assert stats["tp_hits"] == 1
        assert stats["hit_ratio"] == 1.0
