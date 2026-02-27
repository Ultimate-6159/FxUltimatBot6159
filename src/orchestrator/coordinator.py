"""
Main trading coordinator — the heart of the bot.
Orchestrates Data → AI → Risk → Execution pipeline in a continuous loop.
"""

from __future__ import annotations

import signal
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.data.tick_collector import TickCollector
from src.data.multi_tf_loader import MultiTimeframeLoader
from src.data.feature_engine import FeatureEngine
from src.data.sentiment import SentimentProvider
from src.models.lstm_transformer import LSTMTransformerPredictor
from src.models.rl_agent import RLAgent
from src.models.ensemble import EnsembleAggregator, TradeSignal
from src.execution.broker import MT5Broker, OrderResult
from src.execution.virtual_tpsl import VirtualTPSL
from src.execution.spread_guard import SpreadGuard
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizer
from src.utils.logger import get_logger
from src.utils.config import load_config, get_nested

logger = get_logger("FxBot.Coordinator")


class TradingCoordinator:
    """
    Main trading loop controller.

    Pipeline per iteration:
    1. Collect tick data + OHLCV (Data Pipeline)
    2. Generate features (Feature Engine)
    3. Get AI prediction (Ensemble: LSTM + RL)
    4. Check risk limits (Risk Manager)
    5. Check spread conditions (Spread Guard)
    6. Execute order if approved (Broker + Virtual TP/SL)
    7. Monitor open positions
    8. Log heartbeat

    Runs indefinitely until stopped via SIGINT/SIGTERM.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.cfg = config or load_config()
        self._running = False

        # --- Initialize all components ---
        symbol = self.cfg.get("symbol", "XAUUSDm")

        # Data Pipeline
        self.tick_collector = TickCollector(
            symbol=symbol,
            buffer_size=get_nested(self.cfg, "data.tick_buffer_size", 10000),
            point=0.01,
        )
        self.tf_loader = MultiTimeframeLoader(
            symbol=symbol,
            timeframes=self.cfg.get("timeframes", ["M1", "M5", "M15", "H1"]),
        )
        self.feature_engine = FeatureEngine(
            lookback=get_nested(self.cfg, "data.feature_lookback", 60),
        )
        self.sentiment = SentimentProvider()

        # AI Models
        ai_cfg = self.cfg.get("ai", {})
        lstm_cfg = ai_cfg.get("lstm_transformer", {})
        self.lstm_predictor = LSTMTransformerPredictor(
            input_dim=lstm_cfg.get("input_dim", 24),
            d_model=lstm_cfg.get("d_model", 64),
            n_heads=lstm_cfg.get("n_heads", 4),
            n_lstm_layers=lstm_cfg.get("n_lstm_layers", 2),
            n_transformer_layers=lstm_cfg.get("n_transformer_layers", 2),
            seq_length=lstm_cfg.get("seq_length", 60),
            learning_rate=lstm_cfg.get("learning_rate", 3e-4),
        )
        self.rl_agent = RLAgent(algorithm=ai_cfg.get("rl_agent", {}).get("algorithm", "PPO"))
        self.ensemble = EnsembleAggregator(
            lstm_predictor=self.lstm_predictor,
            rl_agent=self.rl_agent,
            min_confidence=get_nested(self.cfg, "ai.ensemble.min_confidence", 0.65),
            require_agreement=get_nested(self.cfg, "ai.ensemble.require_agreement", True),
        )

        # Execution
        exec_cfg = self.cfg.get("execution", {})
        self.broker = MT5Broker(
            symbol=symbol,
            max_retries=exec_cfg.get("max_retries", 3),
        )
        self.virtual_tpsl = VirtualTPSL(
            close_position_fn=self.broker.close_position,
            monitor_interval_ms=get_nested(self.cfg, "execution.virtual_tpsl.monitor_interval_ms", 100),
            trailing_enabled=get_nested(self.cfg, "execution.virtual_tpsl.trailing_enabled", True),
            on_close_callback=self._on_position_closed,
        )
        self.spread_guard = SpreadGuard(
            zscore_threshold=get_nested(self.cfg, "execution.spread_guard.zscore_threshold", 3.0),
            baseline_window=get_nested(self.cfg, "execution.spread_guard.baseline_window", 1000),
            cooldown_seconds=get_nested(self.cfg, "execution.spread_guard.cooldown_seconds", 5),
            max_slippage_points=get_nested(self.cfg, "execution.spread_guard.max_slippage_points", 5.0),
            max_spread_points=get_nested(self.cfg, "execution.spread_guard.max_spread_points", 50.0),
        )
        self.order_manager = OrderManager(
            max_concurrent=exec_cfg.get("max_concurrent_positions", 3),
            symbol=symbol,
        )

        # Risk
        risk_cfg = self.cfg.get("risk", {})
        account_info = self.broker.get_account_info()
        initial_equity = account_info["equity"] if account_info else 10000.0
        self.risk_manager = RiskManager(
            initial_equity=initial_equity,
            max_daily_drawdown_pct=risk_cfg.get("max_daily_drawdown_pct", 5.0),
            max_trade_risk_pct=risk_cfg.get("max_trade_risk_pct", 1.5),
            max_consecutive_losses=risk_cfg.get("max_consecutive_losses", 5),
            cooldown_minutes=risk_cfg.get("cooldown_minutes", 30),
        )

        sizing_cfg = risk_cfg.get("position_sizing", {})
        self.position_sizer = PositionSizer(
            method=sizing_cfg.get("method", "kelly"),
            kelly_fraction=sizing_cfg.get("kelly_fraction", 0.5),
            fixed_lot=sizing_cfg.get("fixed_lot", 0.01),
            min_lot=sizing_cfg.get("min_lot", 0.01),
            max_lot=sizing_cfg.get("max_lot", 1.0),
        )

        # Heartbeat
        self._heartbeat_interval = get_nested(self.cfg, "logging.heartbeat_interval_seconds", 60)
        self._last_heartbeat = 0.0
        self._iteration_count = 0

        # Entry cooldowns
        self._min_entry_interval = exec_cfg.get("min_entry_interval_seconds", 120)
        self._sl_cooldown = exec_cfg.get("sl_cooldown_seconds", 300)
        self._last_entry_time = 0.0
        self._last_sl_time = 0.0

        # Signal consistency — require N consecutive same-direction signals
        self._signal_consistency_bars = exec_cfg.get("signal_consistency_bars", 3)
        self._recent_signals: list[int] = []  # last N signal directions

        # Spread-to-SL guard — skip trades where spread eats too much of the edge
        self._max_spread_to_sl_pct = exec_cfg.get("max_spread_to_sl_pct", 15)

        # Post-fill slippage rejection — close immediately if fill was too bad
        self._max_fill_slippage = exec_cfg.get("max_fill_slippage_points", 30.0)

    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the main trading loop."""
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Connect to broker
        mt5_cfg = self.cfg.get("mt5", {})
        connected = self.broker.connect(
            login=mt5_cfg.get("login", 0),
            password=mt5_cfg.get("password", ""),
            server=mt5_cfg.get("server", ""),
            path=mt5_cfg.get("path", ""),
        )
        if not connected:
            logger.error("Failed to connect to MT5 broker, cannot start")
            return

        # Start virtual TP/SL monitor
        self.virtual_tpsl.start_monitor(self.broker.get_tick)

        # Load saved models if available
        self._load_models()

        self._running = True
        logger.info("🚀 Trading bot started")

        try:
            while self._running:
                self._iteration()
                time.sleep(1.0)  # Main loop interval (1 second)
        except Exception as e:
            logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
        finally:
            self.stop()

    def stop(self) -> None:
        """Gracefully stop the trading bot."""
        self._running = False
        self.virtual_tpsl.stop_monitor()

        # Log final statistics
        stats = self.order_manager.get_statistics()
        risk_report = self.risk_manager.get_risk_report()
        tpsl_stats = self.virtual_tpsl.get_stats()

        logger.info("=" * 60)
        logger.info("BOT SHUTDOWN — Final Report")
        logger.info(f"  Total trades: {stats.get('total_trades', 0)}")
        logger.info(f"  Win rate: {stats.get('win_rate', 0):.1%}")
        logger.info(f"  Total PnL: {stats.get('total_pnl', 0):.2f}")
        logger.info(f"  Max drawdown: {risk_report.get('global_drawdown_pct', 0):.1f}%")
        logger.info(f"  TP/SL hits: {tpsl_stats.get('tp_hits', 0)}/{tpsl_stats.get('sl_hits', 0)}")
        logger.info(f"  Trailing adjustments: {tpsl_stats.get('trailing_adjustments', 0)}")
        logger.info("=" * 60)

        self.broker.disconnect()
        logger.info("Bot stopped")

    # ------------------------------------------------------------------
    # Single Iteration
    # ------------------------------------------------------------------

    def _iteration(self) -> None:
        """Execute one iteration of the trading pipeline."""
        self._iteration_count += 1

        try:
            # 1. Collect market data
            self.tick_collector.collect_ticks(count=100)
            tick = self.broker.get_tick()
            if tick is None:
                return

            # 2. Update spread guard
            spread_points = tick["spread"] / self.tick_collector.point
            spread_alert = self.spread_guard.update_spread(spread_points)

            # 3. Load multi-TF data and compute features
            tf_data = self.tf_loader.load_from_mt5()
            if tf_data is None or tf_data.feature_matrix is None:
                return

            m1_df = tf_data.m1
            if m1_df.empty:
                return

            spreads_array = self.tick_collector.get_spread_array(n=len(m1_df))
            features_df = self.feature_engine.compute_all_features(m1_df, spreads_array)
            feature_matrix = features_df.values.astype(np.float32)

            if len(feature_matrix) < self.lstm_predictor.seq_length:
                return

            # 4. Generate AI signal — adapt features to model dimensions
            lstm_input_dim = self.lstm_predictor.model.input_dim
            feature_seq = feature_matrix[-self.lstm_predictor.seq_length:, :lstm_input_dim]

            # RL observation: match trained observation space
            rl_feature_dim = lstm_input_dim  # RL was trained with same feature set
            if self.rl_agent.agent is not None and hasattr(self.rl_agent.agent, 'observation_space'):
                rl_feature_dim = self.rl_agent.agent.observation_space.shape[0] - 4
            rl_features = feature_matrix[-1, :rl_feature_dim]
            rl_obs = np.concatenate([
                rl_features,
                np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Position state
            ])

            positions = self.order_manager.get_active_orders()
            current_pos = positions[0].direction if positions else 0

            signal = self.ensemble.generate_signal(
                feature_sequence=feature_seq,
                rl_observation=rl_obs,
                current_position=current_pos,
            )

            # 5. Compute real ATR from M1 data
            _h = m1_df["high"]
            _l = m1_df["low"]
            _c = m1_df["close"]
            _tr = pd.concat([
                _h - _l,
                (_h - _c.shift(1)).abs(),
                (_l - _c.shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_value = float(_tr.rolling(14).mean().iloc[-1])
            if np.isnan(atr_value) or atr_value < 0.5:
                atr_value = float((_h - _l).rolling(14).mean().iloc[-1])
            if np.isnan(atr_value) or atr_value < 0.5:
                atr_value = 2.0  # Absolute fallback
            logger.debug(f"ATR(14) = {atr_value:.3f}")

            # 6. Track signal consistency and process
            self._recent_signals.append(signal.direction)
            if len(self._recent_signals) > self._signal_consistency_bars:
                self._recent_signals = self._recent_signals[-self._signal_consistency_bars:]

            if signal.direction != 0 and self._is_signal_consistent(signal.direction):
                self._process_signal(signal, tick, atr_value)

            # 7. Heartbeat
            self._heartbeat()

        except Exception as e:
            logger.error(f"Iteration error: {e}", exc_info=True)

    def _process_signal(self, signal: TradeSignal, tick: dict[str, float], atr_value: float = 2.0) -> None:
        """Process a trade signal through risk checks and execution."""
        now = time.time()

        # Entry interval cooldown — prevent rapid-fire entries
        if now - self._last_entry_time < self._min_entry_interval:
            return

        # Post-SL cooldown — don't chase after a stop loss
        if now - self._last_sl_time < self._sl_cooldown:
            return

        # No hedging — block ALL entries while ANY position is open
        active = self.order_manager.get_active_orders()
        if active:
            return

        # Risk check
        can_trade, reason = self.risk_manager.can_trade()
        if not can_trade:
            logger.debug(f"Trade rejected by risk manager: {reason}")
            return

        # Spread check
        if not self.spread_guard.can_trade():
            logger.debug("Trade rejected by spread guard")
            return

        # Capacity check
        if not self.order_manager.has_capacity():
            logger.debug("Max concurrent positions reached")
            return

        # Calculate position size (ATR from live market data)
        sl_distance = atr_value * get_nested(
            self.cfg, "execution.virtual_tpsl.default_sl_atr_mult", 2.0
        )

        # Spread-to-SL guard — skip trade if spread eats too much of the edge
        spread_price = tick["spread"] if tick.get("spread", 0) > 0 else 0.0
        if sl_distance > 0:
            spread_pct = (spread_price / sl_distance) * 100
            if spread_pct > self._max_spread_to_sl_pct:
                logger.debug(
                    f"Trade rejected: spread/SL ratio {spread_pct:.1f}% > {self._max_spread_to_sl_pct}%"
                )
                return

        lot = self.position_sizer.calculate_lot_size(
            equity=self.risk_manager.state.current_equity,
            sl_distance_price=sl_distance,
            confidence=signal.confidence,
            max_risk_amount=self.risk_manager.max_trade_risk_amount(),
        )

        # Create managed order
        order = self.order_manager.create_order(
            direction=signal.direction,
            volume=lot,
            signal_price=tick["bid"] if signal.direction > 0 else tick["ask"],
            signal_source=signal.source,
            signal_confidence=signal.confidence,
        )
        if order is None:
            return

        # Execute order
        if signal.direction > 0:
            result = self.broker.market_buy(lot, comment=f"AI_{order.order_id}")
        else:
            result = self.broker.market_sell(lot, comment=f"AI_{order.order_id}")

        if not result.success:
            self.order_manager.mark_failed(order.order_id, result.error)
            return

        # Mark filled
        self.order_manager.mark_filled(
            order.order_id, result.ticket, result.price,
            slippage_points=result.slippage_points,
            latency_ms=result.latency_ms,
        )

        # Track slippage
        self.spread_guard.record_slippage(result.slippage_points)

        # Post-fill slippage rejection — close immediately if fill was too bad
        if abs(result.slippage_points) > self._max_fill_slippage:
            logger.warning(
                f"SLIPPAGE REJECTION | ticket={result.ticket} | "
                f"slippage={result.slippage_points:.1f}pts > max {self._max_fill_slippage:.1f} | "
                f"Closing immediately"
            )
            close_result = self.broker.close_position(result.ticket)
            if close_result.success:
                realized_pnl = signal.direction * (close_result.price - result.price)
                self.order_manager.mark_closed(
                    ticket=result.ticket,
                    exit_price=close_result.price,
                    realized_pnl=realized_pnl,
                    close_reason="slippage_rejection",
                )
                self.risk_manager.on_trade_result(realized_pnl)
            return

        # Set virtual TP/SL (Stealth Mode — NO TP/SL sent to broker)
        tp_distance = atr_value * get_nested(
            self.cfg, "execution.virtual_tpsl.default_tp_atr_mult", 4.0
        )
        trailing_distance = atr_value * get_nested(
            self.cfg, "execution.virtual_tpsl.trailing_atr_mult", 1.5
        )

        self.virtual_tpsl.set_levels(
            ticket=result.ticket,
            direction=signal.direction,
            entry_price=result.price,
            tp_distance=tp_distance,
            sl_distance=sl_distance,
            trailing_distance=trailing_distance,
        )

        logger.info(
            f"SIGNAL EXECUTED | {signal.reason} | lot={lot} | ticket={result.ticket} | "
            f"latency={result.latency_ms:.1f}ms"
        )
        self._last_entry_time = time.time()

    def _is_signal_consistent(self, direction: int) -> bool:
        """Check if signal direction has been stable for N consecutive bars."""
        if len(self._recent_signals) < self._signal_consistency_bars:
            return False
        return all(s == direction for s in self._recent_signals[-self._signal_consistency_bars:])

    def _on_position_closed(self, ticket: int, exit_price: float, realized_pnl: float, reason: str) -> None:
        """Callback from VirtualTPSL when a position is closed via TP/SL."""
        self.order_manager.mark_closed(
            ticket=ticket,
            exit_price=exit_price,
            realized_pnl=realized_pnl,
            close_reason=reason,
        )
        self.risk_manager.on_trade_result(realized_pnl)
        if reason == "sl_hit":
            self._last_sl_time = time.time()

    def _heartbeat(self) -> None:
        """Periodic status log."""
        now = time.time()
        if now - self._last_heartbeat < self._heartbeat_interval:
            return

        self._last_heartbeat = now
        stats = self.order_manager.get_statistics()
        risk = self.risk_manager.get_risk_report()
        spread = self.spread_guard.get_stats()

        logger.info(
            f"💓 HEARTBEAT | iter={self._iteration_count} | "
            f"trades={stats.get('total_trades', 0)} | "
            f"PnL={stats.get('total_pnl', 0):.2f} | "
            f"equity={risk.get('current_equity', 0):.2f} | "
            f"DD={risk.get('daily_drawdown_pct', 0):.1f}% | "
            f"spread_μ={spread.get('spread_mean', 0):.1f} | "
            f"active={stats.get('active_positions', 0)}"
        )

    def _load_models(self) -> None:
        """Load pre-trained model weights if available."""
        from pathlib import Path
        models_dir = Path("models")

        lstm_path = models_dir / "lstm_transformer.pt"
        if lstm_path.exists():
            self.lstm_predictor = LSTMTransformerPredictor.from_checkpoint(lstm_path)
            self.ensemble.lstm = self.lstm_predictor

        rl_path = models_dir / "rl_agent"
        if rl_path.exists() or (models_dir / "rl_agent.zip").exists():
            self.rl_agent.load(rl_path)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle SIGINT/SIGTERM for graceful shutdown."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self._running = False
