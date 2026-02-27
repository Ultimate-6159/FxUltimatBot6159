"""
Order lifecycle management.
Tracks all orders from signal to fill to close, with trade journal logging.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("FxBot.OrderManager")


class OrderStatus(Enum):
    PENDING = "pending"
    SENT = "sent"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    MONITORING = "monitoring"     # Position open, monitoring virtual TP/SL
    CLOSED = "closed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ManagedOrder:
    """Complete order record with lifecycle tracking."""
    order_id: str                    # Internal unique ID
    ticket: int = 0                  # MT5 ticket (after fill)
    symbol: str = "XAUUSDm"
    direction: int = 0               # +1 buy, -1 sell
    volume: float = 0.0
    status: OrderStatus = OrderStatus.PENDING

    # Prices
    signal_price: float = 0.0       # Price when signal generated
    entry_price: float = 0.0        # Actual fill price
    exit_price: float = 0.0         # Close price
    virtual_tp: float = 0.0
    virtual_sl: float = 0.0

    # PnL
    realized_pnl: float = 0.0
    commission: float = 0.0
    slippage_points: float = 0.0

    # Timing
    signal_time: float = field(default_factory=time.time)
    fill_time: float = 0.0
    close_time: float = 0.0
    duration_seconds: float = 0.0

    # Metadata
    signal_source: str = ""          # "ensemble", "lstm_only", etc.
    signal_confidence: float = 0.0
    close_reason: str = ""           # "tp_hit", "sl_hit", "trailing", "manual", "risk_limit"
    latency_ms: float = 0.0

    def to_journal_entry(self) -> str:
        """Format as trade journal entry."""
        dir_str = "BUY" if self.direction > 0 else "SELL" if self.direction < 0 else "FLAT"
        return (
            f"ORDER {self.order_id} | {dir_str} {self.volume} {self.symbol} | "
            f"Status={self.status.value} | Entry={self.entry_price:.2f} | "
            f"Exit={self.exit_price:.2f} | PnL={self.realized_pnl:.2f} | "
            f"Reason={self.close_reason} | Conf={self.signal_confidence:.2f} | "
            f"Duration={self.duration_seconds:.0f}s | Latency={self.latency_ms:.1f}ms"
        )


class OrderManager:
    """
    Manages the complete lifecycle of trading orders.

    Responsibilities:
    - Track order state transitions
    - Enforce maximum concurrent positions
    - Maintain trade journal
    - Calculate running statistics (win rate, avg PnL, etc.)
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        symbol: str = "XAUUSDm",
    ):
        self.max_concurrent = max_concurrent
        self.symbol = symbol
        self._lock = threading.Lock()

        self._orders: dict[str, ManagedOrder] = {}
        self._active_tickets: set[int] = set()
        self._order_counter = 0

        # Running statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._total_pnl = 0.0
        self._max_drawdown = 0.0
        self._peak_pnl = 0.0

    # ------------------------------------------------------------------
    # Order Creation & Updates
    # ------------------------------------------------------------------

    def create_order(
        self,
        direction: int,
        volume: float,
        signal_price: float,
        signal_source: str = "",
        signal_confidence: float = 0.0,
    ) -> ManagedOrder | None:
        """
        Create a new managed order.

        Returns None if max concurrent positions reached.
        """
        with self._lock:
            if len(self._active_tickets) >= self.max_concurrent:
                logger.warning(
                    f"Max concurrent positions ({self.max_concurrent}) reached, "
                    f"rejecting new order"
                )
                return None

            self._order_counter += 1
            order_id = f"ORD-{self._order_counter:06d}"

            order = ManagedOrder(
                order_id=order_id,
                symbol=self.symbol,
                direction=direction,
                volume=volume,
                signal_price=signal_price,
                signal_source=signal_source,
                signal_confidence=signal_confidence,
            )

            self._orders[order_id] = order
        logger.info(f"ORDER CREATED | {order.to_journal_entry()}")
        return order

    def mark_filled(
        self,
        order_id: str,
        ticket: int,
        fill_price: float,
        slippage_points: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        """Mark order as filled with MT5 ticket."""
        with self._lock:
            order = self._orders.get(order_id)
            if order is None:
                logger.error(f"Order {order_id} not found")
                return

            order.status = OrderStatus.MONITORING
            order.ticket = ticket
            order.entry_price = fill_price
            order.fill_time = time.time()
            order.slippage_points = slippage_points
            order.latency_ms = latency_ms

            self._active_tickets.add(ticket)
        logger.info(f"ORDER FILLED | {order.to_journal_entry()}")

    def mark_closed(
        self,
        ticket: int,
        exit_price: float,
        realized_pnl: float,
        close_reason: str = "manual",
    ) -> ManagedOrder | None:
        """Mark position as closed and update statistics."""
        with self._lock:
            order = self._find_by_ticket(ticket)
            if order is None:
                logger.warning(f"No managed order found for ticket {ticket}")
                return None

            order.status = OrderStatus.CLOSED
            order.exit_price = exit_price
            order.realized_pnl = realized_pnl
            order.close_time = time.time()
            order.close_reason = close_reason
            order.duration_seconds = order.close_time - order.fill_time

            self._active_tickets.discard(ticket)

            # Update statistics
            self._total_trades += 1
            if realized_pnl > 0:
                self._winning_trades += 1
            self._total_pnl += realized_pnl
            self._peak_pnl = max(self._peak_pnl, self._total_pnl)
            dd = self._peak_pnl - self._total_pnl
            self._max_drawdown = max(self._max_drawdown, dd)

        logger.info(f"ORDER CLOSED | {order.to_journal_entry()}")
        return order

    def mark_failed(self, order_id: str, error: str = "") -> None:
        """Mark order as failed."""
        with self._lock:
            order = self._orders.get(order_id)
            if order:
                order.status = OrderStatus.FAILED
                order.close_reason = f"failed: {error}"
        if order:
            logger.error(f"ORDER FAILED | {order_id} | {error}")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_active_orders(self) -> list[ManagedOrder]:
        """Get all currently active (monitoring) orders."""
        with self._lock:
            return [
                o for o in self._orders.values()
                if o.status == OrderStatus.MONITORING
            ]

    def get_active_count(self) -> int:
        """Get number of active positions."""
        with self._lock:
            return len(self._active_tickets)

    def has_capacity(self) -> bool:
        """Check if we can open more positions."""
        with self._lock:
            return len(self._active_tickets) < self.max_concurrent

    def get_order_by_ticket(self, ticket: int) -> ManagedOrder | None:
        """Find managed order by MT5 ticket."""
        return self._find_by_ticket(ticket)

    def _find_by_ticket(self, ticket: int) -> ManagedOrder | None:
        for order in self._orders.values():
            if order.ticket == ticket:
                return order
        return None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get running trade statistics."""
        with self._lock:
            closed_orders = [o for o in self._orders.values() if o.status == OrderStatus.CLOSED]
            if not closed_orders:
                return {
                    "total_trades": 0,
                    "win_rate": 0.0,
                    "total_pnl": 0.0,
                    "active_positions": len(self._active_tickets),
                }

            pnls = [o.realized_pnl for o in closed_orders]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float("inf")

            return {
                "total_trades": self._total_trades,
                "winning_trades": self._winning_trades,
                "win_rate": self._winning_trades / max(self._total_trades, 1),
                "total_pnl": self._total_pnl,
                "max_drawdown": self._max_drawdown,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_duration_s": sum(o.duration_seconds for o in closed_orders) / len(closed_orders),
                "avg_latency_ms": sum(o.latency_ms for o in closed_orders) / len(closed_orders),
                "active_positions": len(self._active_tickets),
            }

    def get_trade_journal(self, last_n: int = 50) -> list[str]:
        """Get last N trade journal entries."""
        with self._lock:
            closed = [o for o in self._orders.values() if o.status == OrderStatus.CLOSED]
            closed.sort(key=lambda o: o.close_time, reverse=True)
            return [o.to_journal_entry() for o in closed[:last_n]]
