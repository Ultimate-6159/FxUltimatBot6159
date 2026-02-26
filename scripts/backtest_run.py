#!/usr/bin/env python3
"""
Backtest runner script.
Runs walk-forward analysis with configurable parameters.

Usage:
    python scripts/backtest_run.py
    python scripts/backtest_run.py --bars 10000 --walk-forward
"""

import argparse
import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backtest.engine import BacktestEngine
from backtest.metrics import format_metrics_report
from src.data.multi_tf_loader import MultiTimeframeLoader
from src.models.lstm_transformer import LSTMTransformerPredictor
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run Backtest")
    parser.add_argument("--bars", type=int, default=5000, help="Number of M1 bars")
    parser.add_argument("--walk-forward", action="store_true", help="Use walk-forward analysis")
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    args = parser.parse_args()

    logger = setup_logger("FxBot.Backtest")
    cfg = load_config()

    logger.info("=" * 50)
    logger.info("  BACKTEST RUNNER")
    logger.info("=" * 50)

    # Generate synthetic data for testing
    logger.info("Generating synthetic data...")
    tf_loader = MultiTimeframeLoader()
    tf_data = tf_loader.generate_synthetic_data(n_bars=args.bars)

    # Load trained model if available, otherwise detect feature count
    model_path = Path("models/lstm_transformer.pt")
    if model_path.exists():
        logger.info(f"Loading trained model from {model_path}")
        predictor = LSTMTransformerPredictor.from_checkpoint(model_path)
    else:
        from src.data.feature_engine import FeatureEngine
        feature_engine = FeatureEngine(lookback=60)
        features_df = feature_engine.compute_all_features(tf_data.m1)
        features_cols = features_df.shape[1]
        logger.info(f"No trained model found — using untrained predictor (features={features_cols})")
        predictor = LSTMTransformerPredictor(
            input_dim=features_cols,
            d_model=64,
            seq_length=60,
        )

    # Run backtest
    engine = BacktestEngine(
        initial_equity=args.initial_equity,
        spread_model="variable",
        commission_per_lot=7.0,
    )

    if args.walk_forward:
        logger.info("Running walk-forward analysis...")
        results = engine.walk_forward(
            tf_data.m1,
            train_days=args.train_days,
            test_days=args.test_days,
            predictor=predictor,
        )

        # Aggregate results
        all_trades = []
        for r in results:
            all_trades.extend(r.trades)

        logger.info(f"\nWalk-forward complete | {len(results)} windows | {len(all_trades)} total trades")
        for i, r in enumerate(results):
            logger.info(f"  Window {i+1}: {len(r.trades)} trades, PnL={r.metrics.get('total_pnl', 0):.2f}")
    else:
        logger.info("Running single-pass backtest...")
        result = engine.run_backtest(tf_data.m1, predictor=predictor)
        report = format_metrics_report(result.metrics)
        print(report)


if __name__ == "__main__":
    main()
