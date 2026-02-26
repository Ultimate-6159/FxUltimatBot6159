"""
Structured logging with trade journal support.
Provides rotating file logs + console output with color-coded levels.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class TradeJournalFormatter(logging.Formatter):
    """Custom formatter for trade journal entries."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        level = record.levelname
        module = record.module
        msg = record.getMessage()

        if self.use_color and sys.stdout.isatty():
            color = self.COLORS.get(level, "")
            return f"{ts} | {color}{level:<8}{self.RESET} | {module:<20} | {msg}"
        return f"{ts} | {level:<8} | {module:<20} | {msg}"


def setup_logger(
    name: str = "FxBot",
    log_dir: str = "logs",
    level: str = "INFO",
    trade_journal: bool = True,
) -> logging.Logger:
    """
    Set up application logger with console + file output.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        trade_journal: If True, create separate trade journal log.

    Returns:
        Configured logger instance.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # --- Console Handler ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(TradeJournalFormatter(use_color=True))
    logger.addHandler(console)

    # --- Main File Handler (rotating, 10MB, keep 5) ---
    main_file = log_path / "bot.log"
    file_handler = RotatingFileHandler(
        main_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(TradeJournalFormatter(use_color=False))
    logger.addHandler(file_handler)

    # --- Trade Journal (separate file for trade events only) ---
    if trade_journal:
        journal_file = log_path / "trade_journal.log"
        journal_handler = RotatingFileHandler(
            journal_file, maxBytes=10 * 1024 * 1024, backupCount=10, encoding="utf-8"
        )
        journal_handler.setLevel(logging.INFO)
        journal_handler.setFormatter(TradeJournalFormatter(use_color=False))
        journal_handler.addFilter(TradeFilter())
        logger.addHandler(journal_handler)

    return logger


class TradeFilter(logging.Filter):
    """Only pass log records that contain trade-related keywords."""

    TRADE_KEYWORDS = {
        "OPEN", "CLOSE", "BUY", "SELL", "TP_HIT", "SL_HIT",
        "TRAILING", "SIGNAL", "ORDER", "POSITION", "FILL",
    }

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage().upper()
        return any(kw in msg for kw in self.TRADE_KEYWORDS)


def get_logger(name: str = "FxBot") -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
