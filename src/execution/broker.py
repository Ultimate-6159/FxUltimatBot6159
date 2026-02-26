"""
MT5 Broker API abstraction layer.
Handles connection, order execution, position management, and latency tracking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.utils.logger import get_logger

logger = get_logger("FxBot.Broker")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore
    MT5_AVAILABLE = False


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    latency_ms: float = 0.0
    slippage_points: float = 0.0
    error: str = ""
    retcode: int = 0


@dataclass
class PositionInfo:
    """Current open position information."""
    ticket: int
    symbol: str
    direction: int          # +1 long, -1 short
    volume: float
    open_price: float
    current_price: float
    profit: float
    swap: float
    open_time: float


class MT5Broker:
    """
    MT5 API abstraction with latency tracking and auto-reconnect.

    Key design: ALL orders are sent WITHOUT TP/SL —
    Stop management is handled by VirtualTPSL module (Stealth Mode).
    """

    def __init__(
        self,
        symbol: str = "XAUUSDm",
        magic_number: int = 6159,
        max_retries: int = 3,
        retry_delay_ms: int = 100,
    ):
        self.symbol = symbol
        self.magic = magic_number
        self.max_retries = max_retries
        self.retry_delay = retry_delay_ms / 1000.0
        self._connected = False
        self._latency_history: list[float] = []

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def avg_latency_ms(self) -> float:
        if not self._latency_history:
            return 0.0
        return sum(self._latency_history[-50:]) / min(len(self._latency_history), 50)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(
        self,
        login: int = 0,
        password: str = "",
        server: str = "",
        path: str = "",
        timeout: int = 10000,
    ) -> bool:
        """Initialize MT5 terminal connection."""
        if not MT5_AVAILABLE:
            logger.warning("MT5 library not available")
            return False

        init_kwargs: dict[str, Any] = {"timeout": timeout}
        if path:
            init_kwargs["path"] = path
        if login:
            init_kwargs["login"] = login
            init_kwargs["password"] = password
            init_kwargs["server"] = server

        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            logger.error(f"MT5 connection failed: {error}")
            return False

        # Ensure symbol is visible
        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return False

        if not info.visible:
            mt5.symbol_select(self.symbol, True)

        self._connected = True
        account = mt5.account_info()
        logger.info(
            f"MT5 connected | Account: {account.login if account else 'N/A'} | "
            f"Balance: {account.balance if account else 0:.2f} | "
            f"Symbol: {self.symbol}"
        )
        return True

    def disconnect(self) -> None:
        """Shut down MT5 connection."""
        if MT5_AVAILABLE and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def ensure_connected(self) -> bool:
        """Check connection and reconnect if needed."""
        if not MT5_AVAILABLE:
            return False
        if not self._connected:
            return self.connect()
        # Health check
        info = mt5.terminal_info()
        if info is None or not info.connected:
            logger.warning("MT5 connection lost, reconnecting...")
            self._connected = False
            return self.connect()
        return True

    # ------------------------------------------------------------------
    # Market Data
    # ------------------------------------------------------------------

    def get_tick(self) -> dict[str, float] | None:
        """Get current tick data."""
        if not self.ensure_connected():
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return None
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "time": tick.time_msc / 1000.0,
            "spread": (tick.ask - tick.bid),
        }

    def get_account_info(self) -> dict[str, float] | None:
        """Get account equity, balance, margin info."""
        if not self.ensure_connected():
            return None
        info = mt5.account_info()
        if info is None:
            return None
        return {
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "free_margin": info.margin_free,
            "margin_level": info.margin_level if info.margin_level else 0,
            "profit": info.profit,
        }

    # ------------------------------------------------------------------
    # Order Execution (NO TP/SL — Stealth Mode)
    # ------------------------------------------------------------------

    def market_buy(self, volume: float, comment: str = "AI_BUY") -> OrderResult:
        """
        Send market buy order WITHOUT TP/SL.
        TP/SL is managed by VirtualTPSL module.
        """
        return self._send_order(
            order_type="buy",
            volume=volume,
            comment=comment,
        )

    def market_sell(self, volume: float, comment: str = "AI_SELL") -> OrderResult:
        """
        Send market sell order WITHOUT TP/SL.
        TP/SL is managed by VirtualTPSL module.
        """
        return self._send_order(
            order_type="sell",
            volume=volume,
            comment=comment,
        )

    def close_position(self, ticket: int) -> OrderResult:
        """Close a specific position by ticket."""
        if not self.ensure_connected():
            return OrderResult(success=False, error="Not connected")

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return OrderResult(success=False, error=f"Position {ticket} not found")

        pos = position[0]
        # Close = reverse order
        close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": pos.volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": self.magic,
            "comment": "AI_CLOSE",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        start_time = time.perf_counter()
        result = mt5.order_send(request)
        latency = (time.perf_counter() - start_time) * 1000
        self._latency_history.append(latency)

        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            error = f"Close failed: {result.retcode if result else 'None'} - {result.comment if result else ''}"
            logger.error(error)
            return OrderResult(success=False, error=error, latency_ms=latency)

        logger.info(f"CLOSE position #{ticket} | price={result.price:.2f} | latency={latency:.1f}ms")
        return OrderResult(
            success=True,
            ticket=ticket,
            price=result.price,
            volume=pos.volume,
            latency_ms=latency,
        )

    def close_all_positions(self) -> list[OrderResult]:
        """Emergency: close all open positions."""
        results = []
        positions = self.get_positions()
        for pos in positions:
            result = self.close_position(pos.ticket)
            results.append(result)
        if results:
            logger.warning(f"EMERGENCY CLOSE ALL: {len(results)} positions closed")
        return results

    def get_positions(self) -> list[PositionInfo]:
        """Get all open positions for the symbol."""
        if not self.ensure_connected():
            return []

        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []

        result = []
        for pos in positions:
            if pos.magic != self.magic:
                continue
            result.append(PositionInfo(
                ticket=pos.ticket,
                symbol=pos.symbol,
                direction=1 if pos.type == 0 else -1,
                volume=pos.volume,
                open_price=pos.price_open,
                current_price=pos.price_current,
                profit=pos.profit,
                swap=pos.swap,
                open_time=pos.time,
            ))
        return result

    # ------------------------------------------------------------------
    # Internal: Send Order
    # ------------------------------------------------------------------

    def _send_order(
        self,
        order_type: str,
        volume: float,
        comment: str = "",
    ) -> OrderResult:
        """Send market order with retry logic and latency tracking."""
        if not self.ensure_connected():
            return OrderResult(success=False, error="Not connected")

        for attempt in range(self.max_retries):
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(self.retry_delay)
                continue

            if order_type == "buy":
                mt5_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
            else:
                mt5_type = mt5.ORDER_TYPE_SELL
                price = tick.bid

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": mt5_type,
                "price": price,
                "deviation": 20,
                "magic": self.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
                # NOTE: NO sl= and tp= → Stealth Mode
            }

            start_time = time.perf_counter()
            result = mt5.order_send(request)
            latency = (time.perf_counter() - start_time) * 1000
            self._latency_history.append(latency)

            if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                slippage = abs(result.price - price) / (mt5.symbol_info(self.symbol).point or 0.01)
                logger.info(
                    f"ORDER {order_type.upper()} | vol={volume} | "
                    f"price={result.price:.2f} | slippage={slippage:.1f}pts | "
                    f"latency={latency:.1f}ms | ticket={result.order}"
                )
                return OrderResult(
                    success=True,
                    ticket=result.order,
                    price=result.price,
                    volume=volume,
                    latency_ms=latency,
                    slippage_points=slippage,
                )

            error_msg = f"Attempt {attempt+1}/{self.max_retries}: {result.retcode if result else 'None'}"
            logger.warning(error_msg)
            time.sleep(self.retry_delay)

        return OrderResult(
            success=False,
            error=f"Order failed after {self.max_retries} retries",
            latency_ms=self._latency_history[-1] if self._latency_history else 0,
        )
