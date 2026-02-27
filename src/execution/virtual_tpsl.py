"""
Virtual TP/SL — Stealth Mode (Broker-Proof).
Hides Take Profit and Stop Loss from broker by managing them in bot memory.
Triggers close orders via market when price hits virtual levels.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("FxBot.VirtualTPSL")


@dataclass
class VirtualLevel:
    """Virtual TP/SL level for a position."""
    ticket: int
    direction: int           # +1 long, -1 short
    entry_price: float
    take_profit: float
    stop_loss: float
    trailing_active: bool = False
    trailing_distance: float = 0.0
    best_price: float = 0.0  # Best price seen (for trailing)
    created_at: float = field(default_factory=time.time)


class VirtualTPSL:
    """
    Broker-Proof TP/SL Management System.

    Instead of sending TP/SL to the broker (where they can be seen and
    potentially hunted), this module:

    1. Stores TP/SL levels ONLY in bot memory
    2. Monitors price every ~100ms
    3. When price hits TP or SL → sends instant market close order
    4. Supports ATR-based dynamic trailing stop

    Advantages:
    - Broker sees NO stop levels on any order
    - Prevents stop hunting / stop widening by broker
    - Trailing stop adjusts to volatility (ATR-based)
    """

    def __init__(
        self,
        close_position_fn: Callable[[int], Any] | None = None,
        monitor_interval_ms: int = 100,
        trailing_enabled: bool = True,
        on_close_callback: Callable[[int, float, float, str], None] | None = None,
    ):
        self._levels: dict[int, VirtualLevel] = {}
        self._close_fn = close_position_fn
        self._on_close = on_close_callback
        self._monitor_interval = monitor_interval_ms / 1000.0
        self._trailing_enabled = trailing_enabled
        self._running = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Statistics
        self._tp_hits = 0
        self._sl_hits = 0
        self._trailing_adjustments = 0

    # ------------------------------------------------------------------
    # Level Management
    # ------------------------------------------------------------------

    def set_levels(
        self,
        ticket: int,
        direction: int,
        entry_price: float,
        tp_distance: float,
        sl_distance: float,
        trailing_distance: float = 0.0,
    ) -> VirtualLevel:
        """
        Set virtual TP/SL for a position.

        Args:
            ticket: MT5 position ticket number.
            direction: +1 for long, -1 for short.
            entry_price: Position entry price.
            tp_distance: TP distance in price units (positive).
            sl_distance: SL distance in price units (positive).
            trailing_distance: Trailing stop distance (0 = no trailing).

        Returns:
            Created VirtualLevel.
        """
        if direction > 0:
            tp = entry_price + tp_distance
            sl = entry_price - sl_distance
        else:
            tp = entry_price - tp_distance
            sl = entry_price + sl_distance

        level = VirtualLevel(
            ticket=ticket,
            direction=direction,
            entry_price=entry_price,
            take_profit=tp,
            stop_loss=sl,
            trailing_active=trailing_distance > 0 and self._trailing_enabled,
            trailing_distance=trailing_distance,
            best_price=entry_price,
        )

        with self._lock:
            self._levels[ticket] = level

        logger.info(
            f"VIRTUAL TP/SL SET | ticket={ticket} | dir={'LONG' if direction > 0 else 'SHORT'} | "
            f"entry={entry_price:.2f} | TP={tp:.2f} | SL={sl:.2f} | "
            f"trail={'ON' if level.trailing_active else 'OFF'}"
        )
        return level

    def remove_levels(self, ticket: int) -> None:
        """Remove virtual levels for a closed position."""
        with self._lock:
            if ticket in self._levels:
                del self._levels[ticket]
                logger.debug(f"Virtual levels removed for ticket {ticket}")

    def get_level(self, ticket: int) -> VirtualLevel | None:
        """Get virtual level for a ticket."""
        with self._lock:
            return self._levels.get(ticket)

    def get_all_levels(self) -> dict[int, VirtualLevel]:
        """Get all active virtual levels."""
        with self._lock:
            return dict(self._levels)

    # ------------------------------------------------------------------
    # Price Monitoring
    # ------------------------------------------------------------------

    def check_price(self, current_bid: float, current_ask: float) -> list[dict[str, Any]]:
        """
        Check current price against all virtual TP/SL levels.
        Returns list of triggered events.

        For LONG positions: TP hit when bid >= TP, SL hit when bid <= SL
        For SHORT positions: TP hit when ask <= TP, SL hit when ask >= SL
        """
        triggered: list[dict[str, Any]] = []

        with self._lock:
            tickets_to_process = list(self._levels.keys())

        for ticket in tickets_to_process:
            with self._lock:
                level = self._levels.get(ticket)
                if level is None:
                    continue

            check_price = current_bid if level.direction > 0 else current_ask

            # --- TP Check ---
            tp_hit = False
            if level.direction > 0 and check_price >= level.take_profit:
                tp_hit = True
            elif level.direction < 0 and check_price <= level.take_profit:
                tp_hit = True

            if tp_hit:
                self._tp_hits += 1
                pnl_points = abs(check_price - level.entry_price)
                triggered.append({
                    "ticket": ticket,
                    "event": "TP_HIT",
                    "price": check_price,
                    "level": level.take_profit,
                    "direction": level.direction,
                    "pnl_points": pnl_points,
                })
                logger.info(
                    f"TP_HIT | ticket={ticket} | price={check_price:.2f} | "
                    f"TP={level.take_profit:.2f} | PnL={pnl_points:.2f}"
                )
                # Execute close
                if self._close_fn:
                    self._close_fn(ticket)
                if self._on_close:
                    realized_pnl = level.direction * (check_price - level.entry_price)
                    self._on_close(ticket, check_price, realized_pnl, "tp_hit")
                self.remove_levels(ticket)
                continue

            # --- SL Check ---
            sl_hit = False
            if level.direction > 0 and check_price <= level.stop_loss:
                sl_hit = True
            elif level.direction < 0 and check_price >= level.stop_loss:
                sl_hit = True

            if sl_hit:
                self._sl_hits += 1
                pnl_points = -abs(check_price - level.entry_price)
                triggered.append({
                    "ticket": ticket,
                    "event": "SL_HIT",
                    "price": check_price,
                    "level": level.stop_loss,
                    "direction": level.direction,
                    "pnl_points": pnl_points,
                })
                logger.info(
                    f"SL_HIT | ticket={ticket} | price={check_price:.2f} | "
                    f"SL={level.stop_loss:.2f} | Loss={abs(check_price - level.entry_price):.2f}"
                )
                if self._close_fn:
                    self._close_fn(ticket)
                if self._on_close:
                    realized_pnl = level.direction * (check_price - level.entry_price)
                    self._on_close(ticket, check_price, realized_pnl, "sl_hit")
                self.remove_levels(ticket)
                continue

            # --- Trailing Stop Update ---
            if level.trailing_active and level.trailing_distance > 0:
                self._update_trailing(ticket, level, check_price)

        return triggered

    def _update_trailing(self, ticket: int, level: VirtualLevel, current_price: float) -> None:
        """
        Update trailing stop.
        Only trail in profit direction, never widen the stop.
        """
        with self._lock:
            if ticket not in self._levels:
                return

            if level.direction > 0:
                # Long: trail up only
                if current_price > level.best_price:
                    level.best_price = current_price
                    new_sl = current_price - level.trailing_distance
                    if new_sl > level.stop_loss:
                        old_sl = level.stop_loss
                        level.stop_loss = new_sl
                        self._trailing_adjustments += 1
                        logger.debug(
                            f"TRAILING | ticket={ticket} | SL: {old_sl:.2f} → {new_sl:.2f} | "
                            f"price={current_price:.2f}"
                        )
            else:
                # Short: trail down only
                if current_price < level.best_price or level.best_price == level.entry_price:
                    if current_price < level.best_price or level.best_price == level.entry_price:
                        level.best_price = min(current_price, level.best_price) if level.best_price != level.entry_price else current_price
                    new_sl = current_price + level.trailing_distance
                    if new_sl < level.stop_loss:
                        old_sl = level.stop_loss
                        level.stop_loss = new_sl
                        self._trailing_adjustments += 1
                        logger.debug(
                            f"TRAILING | ticket={ticket} | SL: {old_sl:.2f} → {new_sl:.2f} | "
                            f"price={current_price:.2f}"
                        )

    # ------------------------------------------------------------------
    # Background Monitor
    # ------------------------------------------------------------------

    def start_monitor(self, get_tick_fn: Callable[[], dict[str, float] | None]) -> None:
        """Start background price monitoring thread."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(get_tick_fn,),
            daemon=True,
            name="VirtualTPSL-Monitor",
        )
        self._monitor_thread.start()
        logger.info(f"Virtual TP/SL monitor started (interval: {self._monitor_interval*1000:.0f}ms)")

    def stop_monitor(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        logger.info("Virtual TP/SL monitor stopped")

    def _monitor_loop(self, get_tick_fn: Callable[[], dict[str, float] | None]) -> None:
        """Background loop that checks price against virtual levels."""
        while self._running:
            try:
                if self._levels:
                    tick = get_tick_fn()
                    if tick and "bid" in tick and "ask" in tick:
                        self.check_price(tick["bid"], tick["ask"])
            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self._monitor_interval)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get TP/SL statistics."""
        return {
            "active_levels": len(self._levels),
            "tp_hits": self._tp_hits,
            "sl_hits": self._sl_hits,
            "trailing_adjustments": self._trailing_adjustments,
            "hit_ratio": self._tp_hits / max(self._tp_hits + self._sl_hits, 1),
        }
