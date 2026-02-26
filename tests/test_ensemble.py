"""Tests for Ensemble Aggregator — signal combination logic."""

import numpy as np
import pytest

from src.models.ensemble import EnsembleAggregator, TradeSignal
from src.models.lstm_transformer import LSTMTransformerPredictor, Prediction


class MockLSTM:
    """Mock LSTM predictor for testing."""
    def __init__(self, direction=1, confidence=0.8):
        self._direction = direction
        self._confidence = confidence

    def predict(self, features):
        return Prediction(
            direction=self._direction,
            confidence=self._confidence,
            probabilities={"up": 0.8 if self._direction > 0 else 0.1,
                          "down": 0.1 if self._direction > 0 else 0.8,
                          "neutral": 0.1},
        )


class MockRL:
    """Mock RL agent for testing."""
    def __init__(self, action=1):
        self._action = action

    def predict_action(self, obs):
        return self._action


class TestEnsembleAggregator:
    def test_both_agree_buy(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.8),
            rl_agent=MockRL(action=1),  # Buy
            min_confidence=0.65,
            require_agreement=True,
        )
        signal = ensemble.generate_signal(
            feature_sequence=np.zeros((60, 24)),
            rl_observation=np.zeros(28),
        )
        assert signal.direction == 1
        assert signal.confidence > 0.65

    def test_both_agree_sell(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=-1, confidence=0.85),
            rl_agent=MockRL(action=2),  # Sell
            min_confidence=0.65,
        )
        signal = ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))
        assert signal.direction == -1

    def test_disagreement_hold(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.8),
            rl_agent=MockRL(action=2),  # Sell (disagree)
            min_confidence=0.65,
            require_agreement=True,
        )
        signal = ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))
        # Disagreement → either hold or reduced confidence action
        assert signal.direction in (0, 1)  # Depends on logic

    def test_low_confidence_hold(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.4),
            rl_agent=MockRL(action=1),
            min_confidence=0.65,
        )
        signal = ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))
        assert signal.direction == 0  # Too low confidence

    def test_rl_close_signal(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.9),
            rl_agent=MockRL(action=3),  # Close
            min_confidence=0.65,
        )
        signal = ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28), current_position=1)
        assert signal.direction == 0  # Close signal
        assert signal.source == "rl_only"

    def test_signal_history(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.8),
            rl_agent=MockRL(action=1),
            min_confidence=0.65,
        )
        for _ in range(5):
            ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))

        recent = ensemble.get_recent_signals(3)
        assert len(recent) == 3

    def test_signal_stats(self):
        ensemble = EnsembleAggregator(
            lstm_predictor=MockLSTM(direction=1, confidence=0.8),
            rl_agent=MockRL(action=1),
            min_confidence=0.65,
        )
        for _ in range(5):
            ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))

        stats = ensemble.get_signal_stats()
        assert stats["total_signals"] == 5
        assert stats["buy_signals"] == 5

    def test_no_models(self):
        ensemble = EnsembleAggregator(min_confidence=0.65)
        signal = ensemble.generate_signal(np.zeros((60, 24)), np.zeros(28))
        assert signal.direction == 0
