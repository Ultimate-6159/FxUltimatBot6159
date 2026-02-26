"""
Performance metrics calculator.
Computes Sharpe, Sortino, Calmar, win rate, profit factor, and more.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def calculate_metrics(
    trades: list,
    equity_curve: list[float],
    initial_equity: float = 10000.0,
) -> dict[str, float]:
    """
    Calculate comprehensive performance metrics from backtest results.

    Args:
        trades: List of BacktestTrade objects.
        equity_curve: List of equity values over time.
        initial_equity: Starting equity.

    Returns:
        Dictionary of performance metrics.
    """
    if not trades:
        return {
            "total_trades": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "avg_trade_duration": 0.0,
        }

    # Basic stats
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_pnl = sum(pnls)
    win_rate = len(wins) / len(pnls) if pnls else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0

    # Profit Factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Expectancy
    expectancy = np.mean(pnls) if pnls else 0.0

    # Equity curve analysis
    eq = np.array(equity_curve)

    # Max Drawdown
    peak = np.maximum.accumulate(eq)
    drawdown = peak - eq
    max_dd = float(np.max(drawdown))
    max_dd_pct = (max_dd / np.max(peak)) * 100 if np.max(peak) > 0 else 0.0

    # Returns for Sharpe/Sortino
    if len(eq) > 1:
        returns = np.diff(eq) / eq[:-1]
        returns = returns[np.isfinite(returns)]
    else:
        returns = np.array([0.0])

    # Sharpe Ratio (annualized, assuming M1 bars → 525600 per year)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 * 24 * 60)
    else:
        sharpe = 0.0

    # Sortino Ratio (using downside deviation only)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 1 and np.std(downside_returns) > 0:
        sortino = (np.mean(returns) / np.std(downside_returns)) * np.sqrt(252 * 24 * 60)
    else:
        sortino = 0.0

    # Calmar Ratio (return / max drawdown)
    total_return = (eq[-1] - eq[0]) / eq[0] if eq[0] > 0 else 0.0
    calmar = total_return / (max_dd_pct / 100) if max_dd_pct > 0 else 0.0

    # Average trade duration
    durations = [t.duration_bars for t in trades]
    avg_duration = np.mean(durations) if durations else 0.0

    # Trade frequency
    total_bars = len(equity_curve)
    trades_per_day = len(trades) / (total_bars / 1440) if total_bars > 0 else 0.0

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "profit_factor": round(profit_factor, 2),
        "expectancy": round(float(expectancy), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "sortino_ratio": round(float(sortino), 2),
        "calmar_ratio": round(float(calmar), 2),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "avg_trade_duration_bars": round(float(avg_duration), 1),
        "trades_per_day": round(trades_per_day, 1),
        "total_return_pct": round(total_return * 100, 2),
        "final_equity": round(float(eq[-1]), 2),
    }


def format_metrics_report(metrics: dict[str, float]) -> str:
    """Format metrics as a human-readable report."""
    lines = [
        "=" * 50,
        "  BACKTEST PERFORMANCE REPORT",
        "=" * 50,
        f"  Total Trades:       {metrics.get('total_trades', 0)}",
        f"  Win Rate:           {metrics.get('win_rate', 0):.1%}",
        f"  Profit Factor:      {metrics.get('profit_factor', 0):.2f}",
        f"  Expectancy:         ${metrics.get('expectancy', 0):.2f}",
        "",
        f"  Total PnL:          ${metrics.get('total_pnl', 0):.2f}",
        f"  Total Return:       {metrics.get('total_return_pct', 0):.2f}%",
        f"  Final Equity:       ${metrics.get('final_equity', 0):.2f}",
        "",
        f"  Sharpe Ratio:       {metrics.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio:      {metrics.get('sortino_ratio', 0):.2f}",
        f"  Calmar Ratio:       {metrics.get('calmar_ratio', 0):.2f}",
        "",
        f"  Max Drawdown:       ${metrics.get('max_drawdown', 0):.2f} ({metrics.get('max_drawdown_pct', 0):.1f}%)",
        f"  Avg Win:            ${metrics.get('avg_win', 0):.2f}",
        f"  Avg Loss:           ${metrics.get('avg_loss', 0):.2f}",
        f"  Avg Duration:       {metrics.get('avg_trade_duration_bars', 0):.0f} bars",
        f"  Trades/Day:         {metrics.get('trades_per_day', 0):.1f}",
        "=" * 50,
    ]
    return "\n".join(lines)
