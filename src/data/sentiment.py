"""
DXY and economic news sentiment data provider.
Optional alternative data source for improved AI decision-making.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("FxBot.Sentiment")

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore


@dataclass
class SentimentSignal:
    """Aggregated sentiment data point."""
    timestamp: float
    dxy_trend: float          # -1 to +1 (negative = dollar weak → gold up)
    news_impact: float        # 0 = no news, 1 = high impact imminent
    volatility_forecast: float  # Expected volatility multiplier
    composite_score: float    # Weighted combination


class SentimentProvider:
    """
    Provides alternative data signals for AI decision-making.

    Data sources:
    1. DXY (US Dollar Index) — inverse correlation with gold
    2. Economic calendar awareness — avoid high-impact news
    3. Implied volatility estimation from spread behavior
    """

    def __init__(
        self,
        dxy_symbol: str = "USDX",  # Some brokers use DXY.f, DX-SEP24, etc.
        lookback_bars: int = 100,
    ):
        self.dxy_symbol = dxy_symbol
        self.lookback_bars = lookback_bars
        self._dxy_cache: pd.DataFrame | None = None

    def get_dxy_trend(self) -> float:
        """
        Compute DXY trend score from -1 (bearish USD) to +1 (bullish USD).
        Bearish DXY is generally bullish for gold.
        """
        if mt5 is None:
            return 0.0

        try:
            rates = mt5.copy_rates_from_pos(
                self.dxy_symbol, mt5.TIMEFRAME_M15, 0, self.lookback_bars
            )
            if rates is None or len(rates) < 20:
                logger.debug(f"DXY data unavailable for {self.dxy_symbol}")
                return 0.0

            df = pd.DataFrame(rates)
            close = df["close"].values

            # Simple trend: compare short MA vs long MA
            short_ma = np.mean(close[-10:])
            long_ma = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)
            trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0.0

            # Normalize to [-1, 1]
            return float(np.clip(trend * 100, -1.0, 1.0))

        except Exception as e:
            logger.debug(f"DXY trend error: {e}")
            return 0.0

    def get_news_impact_score(self) -> float:
        """
        Estimate upcoming news impact.

        In production, this would query an economic calendar API
        (e.g., Finnhub, ForexFactory scraper). For now, returns 0.

        Returns:
            0.0 = no high-impact news nearby
            0.5 = medium impact within 30 min
            1.0 = high impact (NFP, FOMC, CPI) within 15 min
        """
        # TODO: Integrate with economic calendar API
        # For now, return 0 (no news detected)
        return 0.0

    def estimate_volatility_forecast(
        self,
        recent_spreads: np.ndarray | None = None,
    ) -> float:
        """
        Estimate near-term volatility multiplier.

        Uses spread behavior as a proxy for expected volatility:
        - Widening spreads → higher expected volatility
        - Narrowing spreads → lower expected volatility

        Returns:
            Multiplier around 1.0 (1.0 = normal, >1.5 = elevated)
        """
        if recent_spreads is None or len(recent_spreads) < 20:
            return 1.0

        recent_mean = np.mean(recent_spreads[-20:])
        baseline_mean = np.mean(recent_spreads)

        if baseline_mean <= 0:
            return 1.0

        ratio = recent_mean / baseline_mean
        return float(np.clip(ratio, 0.5, 3.0))

    def get_composite_sentiment(
        self,
        recent_spreads: np.ndarray | None = None,
    ) -> SentimentSignal:
        """
        Get composite sentiment signal combining all sources.

        Returns:
            SentimentSignal with weighted composite score.
        """
        dxy_trend = self.get_dxy_trend()
        news_impact = self.get_news_impact_score()
        vol_forecast = self.estimate_volatility_forecast(recent_spreads)

        # Composite: DXY trend inverted (bearish USD = bullish gold)
        # Penalize high-impact news windows
        composite = (
            -dxy_trend * 0.4          # DXY inverse correlation
            + (1.0 - news_impact) * 0.3  # Prefer no-news periods
            + (1.0 / vol_forecast) * 0.3  # Prefer normal volatility
        )

        return SentimentSignal(
            timestamp=datetime.now(timezone.utc).timestamp(),
            dxy_trend=dxy_trend,
            news_impact=news_impact,
            volatility_forecast=vol_forecast,
            composite_score=float(np.clip(composite, -1.0, 1.0)),
        )
