#!/usr/bin/env python3
"""
Live/Paper Trading Entry Point.
Starts the trading coordinator with full pipeline.

Usage:
    python scripts/live.py                   # Default config
    python scripts/live.py --config path     # Custom config
    python scripts/live.py --paper           # Paper trading mode
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.coordinator import TradingCoordinator
from src.utils.config import load_config
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="FxUltimatBot6159 — AI Trading Bot")
    parser.add_argument("--config", type=str, default=None, help="Path to config directory")
    parser.add_argument("--symbol", type=str, default=None, help="Trading symbol override")
    parser.add_argument("--paper", action="store_true", help="Paper trading mode (demo account)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("FxBot", level=args.log_level)

    # Load config
    config_dir = args.config or str(project_root / "config")
    cfg = load_config(config_dir=config_dir)

    if args.symbol:
        cfg["symbol"] = args.symbol

    if args.paper:
        logger.info("=" * 50)
        logger.info("  PAPER TRADING MODE (Demo Account)")
        logger.info("=" * 50)

    # Create and start coordinator
    coordinator = TradingCoordinator(config=cfg)

    logger.info("Starting FxUltimatBot6159...")
    logger.info(f"Symbol: {cfg.get('symbol', 'XAUUSDm')}")
    logger.info(f"Mode: {'Paper' if args.paper else 'Live'}")
    logger.info(f"Config: {config_dir}")

    coordinator.start()


if __name__ == "__main__":
    main()
