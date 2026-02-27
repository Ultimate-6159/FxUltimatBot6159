"""
Rule-based market structure filters.
Prevents trading against the trend, during low-liquidity sessions,
or without multi-timeframe momentum confirmation.

These filters are independent of AI models and act as hard gates
that must ALL pass before any trade is executed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("FxBot.MarketFilter")


class MarketFilter:
    """
    Multi-layer market structure filter.

    Layer 1 — HTF Trend: Trade only in the direction of the H1 EMA.
    Layer 2 — Session: Trade only during London/NY sessions (high liquidity).
    Layer 3 — Momentum: M5 RSI must confirm the signal direction.
    Layer 4 — Volatility: Skip when M1 ATR is abnormally low (no movement)
              or abnormally high (chaos/news).
    """

    def __init__(
        self,
        ema_period: int = 50,
        rsi_period: int = 14,
        rsi_buy_threshold: float = 45.0,
        rsi_sell_threshold: float = 55.0,
        session_start_utc: int = 7,
        session_end_utc: int = 21,
        atr_low_percentile: float = 10.0,
        atr_high_percentile: float = 95.0,
        server_utc_offset: int = 2,
    ):
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.atr_low_percentile = atr_low_percentile
        self.atr_high_percentile = atr_high_percentile
        self.server_utc_offset = server_utc_offset

    # ------------------------------------------------------------------
    # Combined check
    # ------------------------------------------------------------------

    def check_all(
        self,
        h1_df: pd.DataFrame,
        m5_df: pd.DataFrame,
        m1_df: pd.DataFrame,
        direction: int,
    ) -> tuple[bool, str]:
        """
        Run all filters. Returns (passed, reason).

        Args:
            h1_df: H1 OHLCV DataFrame.
            m5_df: M5 OHLCV DataFrame.
            m1_df: M1 OHLCV DataFrame.
            direction: +1 (buy) or -1 (sell).

        Returns:
            (True, "") if all filters pass, (False, reason) if blocked.
        """
        # Layer 1 — HTF Trend
        ok, reason = self.check_htf_trend(h1_df, direction)
        if not ok:
            return False, reason

        # Layer 2 — Session
        ok, reason = self.check_session()
        if not ok:
            return False, reason

        # Layer 3 — Momentum
        ok, reason = self.check_momentum(m5_df, direction)
        if not ok:
            return False, reason

        # Layer 4 — Volatility regime
        ok, reason = self.check_volatility(m1_df)
        if not ok:
            return False, reason

        return True, ""

    # ------------------------------------------------------------------
    # Layer 1 — H1 Trend Filter
    # ------------------------------------------------------------------

    def check_htf_trend(self, h1_df: pd.DataFrame, direction: int) -> tuple[bool, str]:
        """
        Only trade in the direction of the H1 EMA trend.

        BUY allowed only if price > EMA(50) on H1.
        SELL allowed only if price < EMA(50) on H1.
        """
        if h1_df is None or h1_df.empty or len(h1_df) < self.ema_period:
            return True, ""  # Not enough data — allow (fail-open)

        close = h1_df["close"]
        ema = close.ewm(span=self.ema_period, adjust=False).mean()

        current_price = float(close.iloc[-1])
        ema_value = float(ema.iloc[-1])
        ema_slope = float(ema.iloc[-1] - ema.iloc[-3]) if len(ema) >= 3 else 0.0

        if direction > 0 and current_price < ema_value:
            return False, f"HTF_TREND: BUY blocked — price {current_price:.2f} < H1 EMA({self.ema_period}) {ema_value:.2f}"
        if direction < 0 and current_price > ema_value:
            return False, f"HTF_TREND: SELL blocked — price {current_price:.2f} > H1 EMA({self.ema_period}) {ema_value:.2f}"

        # Also check if EMA is flat (ranging market) — avoid
        if abs(ema_slope) < 0.3:
            return False, f"HTF_TREND: Ranging market — H1 EMA slope {ema_slope:.3f} too flat"

        return True, ""

    # ------------------------------------------------------------------
    # Layer 2 — Session Filter
    # ------------------------------------------------------------------

    def check_session(self) -> tuple[bool, str]:
        """
        Only trade during high-liquidity sessions.

        XAUUSD is most active during:
        - London session: 07:00-16:00 UTC
        - NY session: 13:00-22:00 UTC
        - Best overlap: 13:00-16:00 UTC

        Block trading during Asian session (22:00-07:00 UTC)
        where spreads are wide and price is choppy.
        """
        now_utc = datetime.now(timezone.utc)
        hour_utc = now_utc.hour

        if self.session_start_utc <= hour_utc < self.session_end_utc:
            return True, ""

        return False, f"SESSION: Trading blocked — UTC hour {hour_utc} outside active window [{self.session_start_utc}:00-{self.session_end_utc}:00]"

    # ------------------------------------------------------------------
    # Layer 3 — M5 Momentum (RSI)
    # ------------------------------------------------------------------

    def check_momentum(self, m5_df: pd.DataFrame, direction: int) -> tuple[bool, str]:
        """
        M5 RSI must confirm signal direction.

        BUY: RSI > 45 (not oversold reversal — momentum exists).
        SELL: RSI < 55 (not overbought reversal — momentum exists).

        This prevents counter-trend entries at exhaustion points.
        """
        if m5_df is None or m5_df.empty or len(m5_df) < self.rsi_period + 1:
            return True, ""  # Not enough data — allow

        rsi = self._compute_rsi(m5_df["close"], self.rsi_period)
        current_rsi = float(rsi.iloc[-1])

        if np.isnan(current_rsi):
            return True, ""

        if direction > 0 and current_rsi < self.rsi_buy_threshold:
            return False, f"MOMENTUM: BUY blocked — M5 RSI {current_rsi:.1f} < {self.rsi_buy_threshold} (weak momentum)"
        if direction < 0 and current_rsi > self.rsi_sell_threshold:
            return False, f"MOMENTUM: SELL blocked — M5 RSI {current_rsi:.1f} > {self.rsi_sell_threshold} (weak momentum)"

        return True, ""

    # ------------------------------------------------------------------
    # Layer 4 — Volatility Regime
    # ------------------------------------------------------------------

    def check_volatility(self, m1_df: pd.DataFrame) -> tuple[bool, str]:
        """
        Skip trading when volatility is abnormal.

        Too low ATR → no movement, spread will eat profits.
        Too high ATR → news/chaos, unpredictable.
        """
        if m1_df is None or m1_df.empty or len(m1_df) < 100:
            return True, ""

        h = m1_df["high"]
        l = m1_df["low"]
        c = m1_df["close"]

        tr = pd.concat([
            h - l,
            (h - c.shift(1)).abs(),
            (l - c.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr_series = tr.rolling(14).mean().dropna()
        if len(atr_series) < 50:
            return True, ""

        current_atr = float(atr_series.iloc[-1])
        low_threshold = float(np.percentile(atr_series.values, self.atr_low_percentile))
        high_threshold = float(np.percentile(atr_series.values, self.atr_high_percentile))

        if current_atr < low_threshold:
            return False, f"VOLATILITY: ATR {current_atr:.3f} < p{self.atr_low_percentile:.0f} ({low_threshold:.3f}) — dead market"
        if current_atr > high_threshold:
            return False, f"VOLATILITY: ATR {current_atr:.3f} > p{self.atr_high_percentile:.0f} ({high_threshold:.3f}) — too volatile"

        return True, ""

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI from a price series."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))
