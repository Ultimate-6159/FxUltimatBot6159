"""
Event-driven backtest engine with walk-forward analysis.
Simulates realistic trading with variable spread, slippage, and commission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engine import FeatureEngine
from src.models.lstm_transformer import LSTMTransformerPredictor
from src.models.ensemble import EnsembleAggregator, TradeSignal
from src.risk.position_sizer import PositionSizer
from src.utils.logger import get_logger

logger = get_logger("FxBot.Backtest")


@dataclass
class BacktestTrade:
    """Record of a backtested trade."""
    entry_bar: int
    exit_bar: int
    direction: int
    entry_price: float
    exit_price: float
    volume: float
    pnl: float
    commission: float
    slippage: float
    duration_bars: int
    close_reason: str


@dataclass
class BacktestResult:
    """Complete backtest result."""
    trades: list[BacktestTrade]
    equity_curve: list[float]
    metrics: dict[str, float]
    config: dict[str, Any]


class BacktestEngine:
    """
    Event-driven backtest engine.

    Features:
    - Walk-forward analysis (rolling train/test windows)
    - Variable spread simulation
    - Random slippage model
    - Commission per lot
    - Multi-timeframe data support
    """

    def __init__(
        self,
        initial_equity: float = 10000.0,
        spread_model: str = "variable",
        fixed_spread: float = 3.0,
        commission_per_lot: float = 7.0,
        max_slippage: float = 2.0,
        point: float = 0.01,
        pip_value: float = 1.0,
    ):
        self.initial_equity = initial_equity
        self.spread_model = spread_model
        self.fixed_spread = fixed_spread
        self.commission = commission_per_lot
        self.max_slippage = max_slippage
        self.point = point
        self.pip_value = pip_value

    def run_backtest(
        self,
        df: pd.DataFrame,
        predictor: LSTMTransformerPredictor | None = None,
        ensemble: EnsembleAggregator | None = None,
        seq_length: int = 60,
        tp_atr_mult: float = 2.0,
        sl_atr_mult: float = 1.5,
    ) -> BacktestResult:
        """
        Run a single-pass backtest on OHLCV data.

        Args:
            df: DataFrame with OHLCV columns.
            predictor: LSTM/Transformer predictor (optional, uses random signal if None).
            ensemble: Ensemble aggregator (optional).
            seq_length: Sequence length for predictions.
            tp_atr_mult: Take profit as ATR multiplier.
            sl_atr_mult: Stop loss as ATR multiplier.

        Returns:
            BacktestResult with trades, equity curve, and metrics.
        """
        feature_engine = FeatureEngine(lookback=seq_length)
        features_df = feature_engine.compute_all_features(df)
        feature_matrix = features_df.values.astype(np.float32)

        position_sizer = PositionSizer(method="kelly", kelly_fraction=0.5)

        # State
        equity = self.initial_equity
        equity_curve = [equity]
        trades: list[BacktestTrade] = []
        position = 0
        entry_price = 0.0
        entry_bar = 0
        volume = 0.0

        # ATR for TP/SL
        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        for i in range(seq_length, len(df) - 1):
            current_price = closes[i]

            # Calculate ATR
            atr = self._calc_atr(highs, lows, closes, i, period=14)
            spread = self._get_spread(df, i)

            # --- Check TP/SL for open position ---
            if position != 0:
                tp_distance = atr * tp_atr_mult
                sl_distance = atr * sl_atr_mult

                if position > 0:  # Long
                    tp_level = entry_price + tp_distance
                    sl_level = entry_price - sl_distance
                    if current_price >= tp_level:
                        pnl = self._close_trade(entry_price, current_price, volume, 1, spread)
                        trades.append(BacktestTrade(
                            entry_bar=entry_bar, exit_bar=i, direction=1,
                            entry_price=entry_price, exit_price=current_price,
                            volume=volume, pnl=pnl, commission=self.commission * volume,
                            slippage=0, duration_bars=i - entry_bar, close_reason="tp_hit",
                        ))
                        equity += pnl
                        position = 0
                        position_sizer.record_trade(pnl)
                    elif current_price <= sl_level:
                        pnl = self._close_trade(entry_price, current_price, volume, 1, spread)
                        trades.append(BacktestTrade(
                            entry_bar=entry_bar, exit_bar=i, direction=1,
                            entry_price=entry_price, exit_price=current_price,
                            volume=volume, pnl=pnl, commission=self.commission * volume,
                            slippage=0, duration_bars=i - entry_bar, close_reason="sl_hit",
                        ))
                        equity += pnl
                        position = 0
                        position_sizer.record_trade(pnl)

                elif position < 0:  # Short
                    tp_level = entry_price - tp_distance
                    sl_level = entry_price + sl_distance
                    if current_price <= tp_level:
                        pnl = self._close_trade(entry_price, current_price, volume, -1, spread)
                        trades.append(BacktestTrade(
                            entry_bar=entry_bar, exit_bar=i, direction=-1,
                            entry_price=entry_price, exit_price=current_price,
                            volume=volume, pnl=pnl, commission=self.commission * volume,
                            slippage=0, duration_bars=i - entry_bar, close_reason="tp_hit",
                        ))
                        equity += pnl
                        position = 0
                        position_sizer.record_trade(pnl)
                    elif current_price >= sl_level:
                        pnl = self._close_trade(entry_price, current_price, volume, -1, spread)
                        trades.append(BacktestTrade(
                            entry_bar=entry_bar, exit_bar=i, direction=-1,
                            entry_price=entry_price, exit_price=current_price,
                            volume=volume, pnl=pnl, commission=self.commission * volume,
                            slippage=0, duration_bars=i - entry_bar, close_reason="sl_hit",
                        ))
                        equity += pnl
                        position = 0
                        position_sizer.record_trade(pnl)

            # --- Generate signal for new entry ---
            if position == 0 and i >= seq_length:
                signal_direction = 0

                if predictor is not None:
                    feature_seq = feature_matrix[i - seq_length:i]
                    if len(feature_seq) == seq_length:
                        pred = predictor.predict(feature_seq)
                        if pred.confidence >= 0.65:
                            signal_direction = pred.direction

                if signal_direction != 0:
                    sl_dist = atr * sl_atr_mult
                    volume = position_sizer.calculate_lot_size(
                        equity=equity,
                        sl_distance_price=sl_dist,
                        confidence=0.7,
                        max_risk_amount=equity * 0.015,
                    )

                    # Apply spread and slippage
                    slippage = np.random.uniform(0, self.max_slippage) * self.point
                    if signal_direction > 0:
                        entry_price = current_price + spread * self.point / 2 + slippage
                    else:
                        entry_price = current_price - spread * self.point / 2 - slippage

                    position = signal_direction
                    entry_bar = i

            equity_curve.append(equity)

        # Close any remaining position
        if position != 0:
            final_price = closes[-1]
            pnl = self._close_trade(entry_price, final_price, volume, position, 0)
            trades.append(BacktestTrade(
                entry_bar=entry_bar, exit_bar=len(df)-1, direction=position,
                entry_price=entry_price, exit_price=final_price,
                volume=volume, pnl=pnl, commission=self.commission * volume,
                slippage=0, duration_bars=len(df)-1-entry_bar, close_reason="end_of_data",
            ))
            equity += pnl
            equity_curve.append(equity)

        # Calculate metrics
        from backtest.metrics import calculate_metrics
        metrics = calculate_metrics(trades, equity_curve, self.initial_equity)

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics,
            config={
                "initial_equity": self.initial_equity,
                "spread_model": self.spread_model,
                "commission": self.commission,
                "tp_atr_mult": tp_atr_mult,
                "sl_atr_mult": sl_atr_mult,
            },
        )

    def walk_forward(
        self,
        df: pd.DataFrame,
        train_days: int = 60,
        test_days: int = 15,
        step_days: int = 15,
        predictor: LSTMTransformerPredictor | None = None,
    ) -> list[BacktestResult]:
        """
        Walk-forward analysis — prevents overfitting.

        1. Split data into rolling train/test windows
        2. Train on window N, test on window N+1
        3. Aggregate out-of-sample results

        Returns list of BacktestResult (one per test window).
        """
        bars_per_day = 1440  # M1 candles per day
        train_bars = train_days * bars_per_day
        test_bars = test_days * bars_per_day
        step_bars = step_days * bars_per_day

        results: list[BacktestResult] = []
        start = 0

        while start + train_bars + test_bars <= len(df):
            train_end = start + train_bars
            test_end = train_end + test_bars

            train_df = df.iloc[start:train_end]
            test_df = df.iloc[train_end:test_end]

            logger.info(
                f"Walk-forward window | Train: bars {start}-{train_end} | "
                f"Test: bars {train_end}-{test_end}"
            )

            # Train model on training window (if predictor provided)
            if predictor is not None:
                feature_engine = FeatureEngine(lookback=60)
                features = feature_engine.compute_all_features(train_df)
                fm = features.values.astype(np.float32)
                sequences, labels = predictor.prepare_sequences(fm, lookahead=5)
                if len(sequences) > 0:
                    for epoch in range(10):  # Quick training
                        predictor.train_epoch(sequences, labels)

            # Test on out-of-sample window
            result = self.run_backtest(test_df, predictor=predictor)
            results.append(result)

            logger.info(
                f"  Test result | Trades: {len(result.trades)} | "
                f"PnL: {result.metrics.get('total_pnl', 0):.2f} | "
                f"Win rate: {result.metrics.get('win_rate', 0):.1%}"
            )

            start += step_bars

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _close_trade(
        self,
        entry: float,
        exit_price: float,
        volume: float,
        direction: int,
        spread: float,
    ) -> float:
        """Calculate trade PnL."""
        price_diff = (exit_price - entry) * direction
        pnl_points = price_diff / self.point
        pnl = pnl_points * self.pip_value * volume
        pnl -= self.commission * volume  # Commission
        pnl -= spread * self.pip_value * volume * 0.5  # Exit spread cost
        return pnl

    def _calc_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
        idx: int, period: int = 14,
    ) -> float:
        """Calculate ATR at given index."""
        start = max(0, idx - period)
        h = highs[start:idx+1]
        l = lows[start:idx+1]
        c = closes[start:idx+1]
        if len(c) < 2:
            return (h[-1] - l[-1]) if len(h) > 0 else 1.0

        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return float(np.mean(tr)) if len(tr) > 0 else 1.0

    def _get_spread(self, df: pd.DataFrame, idx: int) -> float:
        """Get spread for bar (variable or fixed)."""
        if self.spread_model == "fixed":
            return self.fixed_spread
        # Variable: use actual spread if available, else simulate
        if "spread" in df.columns:
            return float(df.iloc[idx]["spread"])
        return self.fixed_spread + np.random.uniform(-0.5, 1.5)
