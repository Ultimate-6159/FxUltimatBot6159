"""
Real-time tick data collector via MT5.
Detects Order Blocks and Liquidity Sweeps from tick-level data.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("FxBot.TickCollector")

# MT5 is optional at import time (not available during testing)
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore


@dataclass
class Tick:
    """Single tick data point."""
    time: float           # Unix timestamp (seconds)
    bid: float
    ask: float
    last: float
    volume: float
    spread: float         # ask - bid in points
    mid: float            # (bid + ask) / 2
    direction: int        # +1 uptick, -1 downtick, 0 unchanged

    @classmethod
    def from_mt5(cls, mt5_tick: Any, prev_mid: float = 0.0, point: float = 0.01) -> Tick:
        bid = float(mt5_tick.bid)
        ask = float(mt5_tick.ask)
        mid = (bid + ask) / 2.0
        spread = (ask - bid) / point
        direction = 1 if mid > prev_mid else (-1 if mid < prev_mid else 0)
        return cls(
            time=float(mt5_tick.time_msc) / 1000.0,
            bid=bid,
            ask=ask,
            last=float(getattr(mt5_tick, "last", mid)),
            volume=float(getattr(mt5_tick, "volume_real", getattr(mt5_tick, "volume", 0))),
            spread=spread,
            mid=mid,
            direction=direction,
        )


@dataclass
class OrderBlock:
    """Detected order block zone."""
    price_low: float
    price_high: float
    volume: float
    timestamp: float
    block_type: str       # "demand" or "supply"
    strength: float       # 0-1 normalized


@dataclass
class LiquiditySweep:
    """Detected liquidity sweep event."""
    sweep_price: float
    reversal_price: float
    direction: str        # "above" (hunted highs) or "below" (hunted lows)
    timestamp: float
    magnitude: float      # ATR-normalized movement


class TickCollector:
    """
    Collects and analyzes real-time tick data from MT5.

    Features:
    - Rolling tick buffer with configurable size
    - Order Block detection (volume absorption zones)
    - Liquidity Sweep detection (stop hunts)
    - Spread and tick direction tracking
    """

    def __init__(
        self,
        symbol: str = "XAUUSDm",
        buffer_size: int = 10000,
        point: float = 0.01,
        order_block_window: int = 20,
        liquidity_sweep_atr_mult: float = 1.5,
    ):
        self.symbol = symbol
        self.buffer_size = buffer_size
        self.point = point
        self.order_block_window = order_block_window
        self.liquidity_sweep_atr_mult = liquidity_sweep_atr_mult

        self.ticks: deque[Tick] = deque(maxlen=buffer_size)
        self._prev_mid: float = 0.0

    # ------------------------------------------------------------------
    # MT5 Connection
    # ------------------------------------------------------------------

    def connect(self, login: int = 0, password: str = "", server: str = "", path: str = "") -> bool:
        """Initialize MT5 connection."""
        if mt5 is None:
            logger.warning("MetaTrader5 library not available, using synthetic mode")
            return False

        init_kwargs: dict[str, Any] = {}
        if path:
            init_kwargs["path"] = path
        if login:
            init_kwargs["login"] = login
            init_kwargs["password"] = password
            init_kwargs["server"] = server

        if not mt5.initialize(**init_kwargs):
            logger.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        info = mt5.symbol_info(self.symbol)
        if info is None:
            logger.error(f"Symbol {self.symbol} not found")
            return False

        if not info.visible:
            mt5.symbol_select(self.symbol, True)

        self.point = info.point
        logger.info(f"Connected to MT5 | Symbol: {self.symbol} | Point: {self.point}")
        return True

    # ------------------------------------------------------------------
    # Tick Collection
    # ------------------------------------------------------------------

    def collect_ticks(self, count: int = 100) -> list[Tick]:
        """Fetch latest ticks from MT5 and add to buffer."""
        if mt5 is None:
            return []

        raw_ticks = mt5.copy_ticks_from(self.symbol, datetime.now(timezone.utc), count, mt5.COPY_TICKS_ALL)
        if raw_ticks is None or len(raw_ticks) == 0:
            return []

        new_ticks = []
        for rt in raw_ticks:
            tick = Tick.from_mt5(rt, self._prev_mid, self.point)
            self._prev_mid = tick.mid
            self.ticks.append(tick)
            new_ticks.append(tick)

        return new_ticks

    def get_latest_ticks(self, n: int = 100) -> list[Tick]:
        """Get last N ticks from buffer."""
        ticks_list = list(self.ticks)
        return ticks_list[-n:] if len(ticks_list) >= n else ticks_list

    def get_spread_array(self, n: int = 1000) -> np.ndarray:
        """Get last N spread values as numpy array."""
        ticks = self.get_latest_ticks(n)
        if not ticks:
            return np.array([])
        return np.array([t.spread for t in ticks])

    def get_mid_prices(self, n: int = 1000) -> np.ndarray:
        """Get last N mid prices as numpy array."""
        ticks = self.get_latest_ticks(n)
        if not ticks:
            return np.array([])
        return np.array([t.mid for t in ticks])

    # ------------------------------------------------------------------
    # Order Block Detection
    # ------------------------------------------------------------------

    def detect_order_blocks(self, lookback: int | None = None) -> list[OrderBlock]:
        """
        Detect order blocks from recent tick data.

        An Order Block is a zone where heavy volume was absorbed, indicating
        institutional interest. These zones often act as support/resistance.

        Detection logic:
        1. Group ticks into micro-candles (N ticks each)
        2. Find candles with volume > 2x average
        3. If followed by directional move → mark as order block
        """
        lookback = lookback or self.order_block_window
        ticks = self.get_latest_ticks(lookback * 50)  # Need enough ticks
        if len(ticks) < lookback * 10:
            return []

        # Group into micro-candles of 50 ticks each
        candle_size = 50
        candles = []
        for i in range(0, len(ticks) - candle_size, candle_size):
            chunk = ticks[i:i + candle_size]
            prices = [t.mid for t in chunk]
            volumes = [t.volume for t in chunk]
            candles.append({
                "open": prices[0],
                "high": max(prices),
                "low": min(prices),
                "close": prices[-1],
                "volume": sum(volumes),
                "time": chunk[-1].time,
            })

        if len(candles) < 5:
            return []

        avg_vol = np.mean([c["volume"] for c in candles])
        blocks: list[OrderBlock] = []

        for i in range(1, len(candles) - 1):
            c = candles[i]
            vol_ratio = c["volume"] / max(avg_vol, 1e-10)

            if vol_ratio < 2.0:
                continue

            # Check if next candle moves directionally (confirms absorption)
            next_c = candles[i + 1]
            body_prev = c["close"] - c["open"]
            body_next = next_c["close"] - next_c["open"]

            if body_next > 0 and body_prev <= 0:
                # Bullish reversal → demand block
                blocks.append(OrderBlock(
                    price_low=c["low"],
                    price_high=c["high"],
                    volume=c["volume"],
                    timestamp=c["time"],
                    block_type="demand",
                    strength=min(vol_ratio / 5.0, 1.0),
                ))
            elif body_next < 0 and body_prev >= 0:
                # Bearish reversal → supply block
                blocks.append(OrderBlock(
                    price_low=c["low"],
                    price_high=c["high"],
                    volume=c["volume"],
                    timestamp=c["time"],
                    block_type="supply",
                    strength=min(vol_ratio / 5.0, 1.0),
                ))

        return blocks

    # ------------------------------------------------------------------
    # Liquidity Sweep Detection
    # ------------------------------------------------------------------

    def detect_liquidity_sweep(self, window: int = 200) -> list[LiquiditySweep]:
        """
        Detect liquidity sweeps (stop hunts) from tick data.

        A liquidity sweep occurs when price briefly spikes beyond a
        recent high/low (taking out stops) then quickly reverses.

        Detection logic:
        1. Find recent high/low over a lookback period
        2. Check if price broke above/below then reversed within N ticks
        3. Magnitude must exceed ATR threshold
        """
        ticks = self.get_latest_ticks(window)
        if len(ticks) < 50:
            return []

        prices = np.array([t.mid for t in ticks])

        # Calculate simple ATR proxy from tick volatility
        price_diffs = np.abs(np.diff(prices))
        atr_proxy = np.mean(price_diffs) * 14 if len(price_diffs) > 0 else 1.0
        sweep_threshold = atr_proxy * self.liquidity_sweep_atr_mult

        sweeps: list[LiquiditySweep] = []
        check_window = 50  # Ticks to look ahead for reversal

        # Split into lookback and check zones
        lookback_prices = prices[:-check_window] if len(prices) > check_window else prices[:len(prices)//2]
        check_start = len(lookback_prices)

        if len(lookback_prices) < 10:
            return sweeps

        recent_high = np.max(lookback_prices)
        recent_low = np.min(lookback_prices)

        for i in range(check_start, min(check_start + check_window, len(prices))):
            price = prices[i]

            # Check sweep above (hunted highs)
            if price > recent_high + sweep_threshold:
                # Look for reversal in next ticks
                remaining = prices[i:i + 20]
                if len(remaining) > 5:
                    min_after = np.min(remaining[1:])
                    if min_after < recent_high:
                        sweeps.append(LiquiditySweep(
                            sweep_price=price,
                            reversal_price=min_after,
                            direction="above",
                            timestamp=ticks[i].time,
                            magnitude=(price - recent_high) / max(atr_proxy, 1e-10),
                        ))

            # Check sweep below (hunted lows)
            if price < recent_low - sweep_threshold:
                remaining = prices[i:i + 20]
                if len(remaining) > 5:
                    max_after = np.max(remaining[1:])
                    if max_after > recent_low:
                        sweeps.append(LiquiditySweep(
                            sweep_price=price,
                            reversal_price=max_after,
                            direction="below",
                            timestamp=ticks[i].time,
                            magnitude=(recent_low - price) / max(atr_proxy, 1e-10),
                        ))

        return sweeps

    # ------------------------------------------------------------------
    # Synthetic Data (for testing / backtesting)
    # ------------------------------------------------------------------

    def generate_synthetic_ticks(self, n: int = 1000, base_price: float = 2650.0) -> list[Tick]:
        """Generate synthetic tick data for testing."""
        np.random.seed(42)
        ticks: list[Tick] = []
        price = base_price
        t = time.time()

        for i in range(n):
            # Random walk with slight mean reversion
            change = np.random.normal(0, 0.5) - 0.001 * (price - base_price)
            price += change
            spread = max(0.01, abs(np.random.normal(0.03, 0.01)))
            bid = price - spread / 2
            ask = price + spread / 2
            mid = (bid + ask) / 2
            volume = abs(np.random.normal(10, 5))
            direction = 1 if change > 0 else (-1 if change < 0 else 0)

            tick = Tick(
                time=t + i * 0.1,
                bid=round(bid, 2),
                ask=round(ask, 2),
                last=round(mid, 2),
                volume=round(volume, 2),
                spread=round(spread / self.point, 1),
                mid=round(mid, 2),
                direction=direction,
            )
            ticks.append(tick)
            self.ticks.append(tick)
            self._prev_mid = mid

        logger.info(f"Generated {n} synthetic ticks, price range: {min(t.mid for t in ticks):.2f} - {max(t.mid for t in ticks):.2f}")
        return ticks
