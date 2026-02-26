"""
Ensemble signal aggregator.
Combines LSTM/Transformer directional prediction + RL agent action
into a unified trade signal with confidence filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.lstm_transformer import LSTMTransformerPredictor, Prediction
from src.models.rl_agent import RLAgent
from src.utils.logger import get_logger

logger = get_logger("FxBot.Ensemble")


@dataclass
class TradeSignal:
    """Unified trade signal from ensemble."""
    direction: int          # +1 (buy), -1 (sell), 0 (hold)
    confidence: float       # 0.0 - 1.0
    suggested_lot: float    # Suggested lot size from RL
    source: str             # "ensemble", "lstm_only", "rl_only"
    lstm_prediction: Prediction | None = None
    rl_action: int = 0      # Raw RL action
    reason: str = ""        # Human-readable reasoning


class EnsembleAggregator:
    """
    Combines signals from multiple AI models into a single trade decision.

    Voting Logic:
    1. LSTM/Transformer provides direction + confidence
    2. RL Agent provides action (buy/sell/hold/close) and timing
    3. Both must agree on direction for a trade to execute
    4. Minimum confidence threshold must be met
    5. RL determines the final action and lot sizing
    """

    def __init__(
        self,
        lstm_predictor: LSTMTransformerPredictor | None = None,
        rl_agent: RLAgent | None = None,
        min_confidence: float = 0.65,
        require_agreement: bool = True,
    ):
        self.lstm = lstm_predictor
        self.rl = rl_agent
        self.min_confidence = min_confidence
        self.require_agreement = require_agreement

        self._signal_history: list[TradeSignal] = []

    def generate_signal(
        self,
        feature_sequence: np.ndarray,
        rl_observation: np.ndarray,
        current_position: int = 0,
    ) -> TradeSignal:
        """
        Generate a unified trade signal from all models.

        Args:
            feature_sequence: (seq_length, n_features) for LSTM.
            rl_observation: Flattened observation for RL agent.
            current_position: Current position state (-1, 0, +1).

        Returns:
            TradeSignal with direction, confidence, and reasoning.
        """
        # --- Get LSTM prediction ---
        lstm_pred = None
        lstm_direction = 0
        lstm_confidence = 0.0

        if self.lstm is not None:
            try:
                lstm_pred = self.lstm.predict(feature_sequence)
                lstm_direction = lstm_pred.direction
                lstm_confidence = lstm_pred.confidence
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        # --- Get RL action ---
        rl_action = 0  # Default: hold
        rl_direction = 0

        if self.rl is not None:
            try:
                rl_action = self.rl.predict_action(rl_observation)
                # Map RL action to direction
                rl_direction = {0: 0, 1: 1, 2: -1, 3: 0}.get(rl_action, 0)
            except Exception as e:
                logger.warning(f"RL prediction failed: {e}")

        # --- Ensemble Decision Logic ---
        signal = self._aggregate(
            lstm_direction=lstm_direction,
            lstm_confidence=lstm_confidence,
            rl_action=rl_action,
            rl_direction=rl_direction,
            current_position=current_position,
            lstm_pred=lstm_pred,
        )

        self._signal_history.append(signal)
        return signal

    def _aggregate(
        self,
        lstm_direction: int,
        lstm_confidence: float,
        rl_action: int,
        rl_direction: int,
        current_position: int,
        lstm_pred: Prediction | None,
    ) -> TradeSignal:
        """Core aggregation logic."""

        # Case 1: RL says close → always close
        if rl_action == 3 and current_position != 0:
            return TradeSignal(
                direction=0,
                confidence=1.0,
                suggested_lot=0.0,
                source="rl_only",
                lstm_prediction=lstm_pred,
                rl_action=rl_action,
                reason="RL signal: close position",
            )

        # Case 2: Confidence too low → hold
        if lstm_confidence < self.min_confidence:
            return TradeSignal(
                direction=0,
                confidence=lstm_confidence,
                suggested_lot=0.0,
                source="ensemble",
                lstm_prediction=lstm_pred,
                rl_action=rl_action,
                reason=f"Confidence too low: {lstm_confidence:.2f} < {self.min_confidence}",
            )

        # Case 3: Both agree on direction → trade!
        if self.require_agreement:
            if lstm_direction == rl_direction and lstm_direction != 0:
                combined_confidence = lstm_confidence * 0.6 + 0.4  # Boost for agreement
                return TradeSignal(
                    direction=lstm_direction,
                    confidence=combined_confidence,
                    suggested_lot=0.01,  # Will be adjusted by position_sizer
                    source="ensemble",
                    lstm_prediction=lstm_pred,
                    rl_action=rl_action,
                    reason=f"LSTM+RL agree: {'BUY' if lstm_direction > 0 else 'SELL'} "
                           f"(conf={combined_confidence:.2f})",
                )
            elif lstm_direction != 0 and rl_direction == 0:
                # RL says hold but LSTM has direction → weight LSTM with reduced confidence
                reduced_confidence = lstm_confidence * 0.5
                if reduced_confidence >= self.min_confidence:
                    return TradeSignal(
                        direction=lstm_direction,
                        confidence=reduced_confidence,
                        suggested_lot=0.01,
                        source="lstm_only",
                        lstm_prediction=lstm_pred,
                        rl_action=rl_action,
                        reason=f"LSTM signal only (RL hold): {'BUY' if lstm_direction > 0 else 'SELL'} "
                               f"(conf={reduced_confidence:.2f})",
                    )
        else:
            # No agreement required: LSTM direction + RL timing
            if lstm_direction != 0:
                return TradeSignal(
                    direction=lstm_direction,
                    confidence=lstm_confidence,
                    suggested_lot=0.01,
                    source="lstm_only",
                    lstm_prediction=lstm_pred,
                    rl_action=rl_action,
                    reason=f"LSTM signal: {'BUY' if lstm_direction > 0 else 'SELL'} "
                           f"(conf={lstm_confidence:.2f})",
                )

        # Default: hold
        return TradeSignal(
            direction=0,
            confidence=0.0,
            suggested_lot=0.0,
            source="ensemble",
            lstm_prediction=lstm_pred,
            rl_action=rl_action,
            reason="No consensus / neutral signal",
        )

    def get_recent_signals(self, n: int = 10) -> list[TradeSignal]:
        """Get last N signals for analysis."""
        return self._signal_history[-n:]

    def get_signal_stats(self) -> dict[str, Any]:
        """Get statistics about generated signals."""
        if not self._signal_history:
            return {"total_signals": 0}

        directions = [s.direction for s in self._signal_history]
        confidences = [s.confidence for s in self._signal_history]

        return {
            "total_signals": len(self._signal_history),
            "buy_signals": sum(1 for d in directions if d > 0),
            "sell_signals": sum(1 for d in directions if d < 0),
            "hold_signals": sum(1 for d in directions if d == 0),
            "avg_confidence": float(np.mean(confidences)),
            "max_confidence": float(np.max(confidences)),
        }
