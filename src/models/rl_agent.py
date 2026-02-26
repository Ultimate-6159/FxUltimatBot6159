"""
Reinforcement Learning Agent for dynamic position sizing and trade timing.
Uses PPO via stable-baselines3 with custom reward shaping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utils.logger import get_logger

logger = get_logger("FxBot.RLAgent")

try:
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed, RL agent will be limited")


# ======================================================================
# Trading Environment (Gymnasium)
# ======================================================================

@dataclass
class TradeState:
    """Current trading state."""
    position: int = 0           # -1 (short), 0 (flat), +1 (long)
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    equity: float = 10000.0
    peak_equity: float = 10000.0
    bars_in_trade: int = 0
    consecutive_losses: int = 0
    bars_since_last_loss: int = 999
    total_trades: int = 0
    winning_trades: int = 0


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for RL agent training.

    Observation space:
    - Feature vector from FeatureEngine (N features)
    - LSTM/Transformer prediction (direction, confidence)
    - Current position state (position, unrealized PnL, bars in trade)

    Action space:
    - Discrete(4): Hold=0, Buy=1, Sell=2, Close=3
    - Continuous lot size is handled externally by position_sizer
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        feature_matrix: np.ndarray,
        initial_equity: float = 10000.0,
        spread_cost: float = 3.0,        # Points
        commission_per_lot: float = 7.0,  # USD per round-trip lot
        point_value: float = 0.01,
        lot_size: float = 0.01,
        pip_value_per_lot: float = 1.0,
        max_bars: int | None = None,
    ):
        super().__init__()

        self.feature_matrix = feature_matrix
        self.initial_equity = initial_equity
        self.spread_cost = spread_cost
        self.commission = commission_per_lot
        self.point_value = point_value
        self.lot_size = lot_size
        self.pip_value = pip_value_per_lot

        self.n_features = feature_matrix.shape[1] if feature_matrix.ndim > 1 else 1
        self.max_bars = max_bars or len(feature_matrix) - 1

        # Observation: features + [position, unrealized_pnl_norm, bars_in_trade_norm, consecutive_losses_norm]
        obs_dim = self.n_features + 4
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action: 0=Hold, 1=Buy, 2=Sell, 3=Close
        self.action_space = spaces.Discrete(4)

        self._state = TradeState(equity=initial_equity, peak_equity=initial_equity)
        self._step_idx = 0

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._state = TradeState(equity=self.initial_equity, peak_equity=self.initial_equity)
        self._step_idx = 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        prev_equity = self._state.equity
        reward = 0.0
        info: dict[str, Any] = {}

        # Get current price (use close price, assumed column index 3)
        current_price = self._get_price()
        next_price = self._get_price(offset=1) if self._step_idx < self.max_bars - 1 else current_price

        # Execute action
        if action == 1 and self._state.position <= 0:  # BUY
            if self._state.position == -1:
                # Close short first
                reward += self._close_position(current_price)
            self._open_position(1, current_price)
            reward -= self.spread_cost * self.point_value  # Spread cost

        elif action == 2 and self._state.position >= 0:  # SELL
            if self._state.position == 1:
                # Close long first
                reward += self._close_position(current_price)
            self._open_position(-1, current_price)
            reward -= self.spread_cost * self.point_value

        elif action == 3 and self._state.position != 0:  # CLOSE
            reward += self._close_position(current_price)

        # Update unrealized PnL
        if self._state.position != 0:
            self._state.bars_in_trade += 1
            price_diff = (next_price - self._state.entry_price) * self._state.position
            self._state.unrealized_pnl = price_diff * self.pip_value * (self.lot_size / self.point_value)

        # Update equity
        self._state.equity = prev_equity + self._state.unrealized_pnl + self._state.realized_pnl - prev_equity + self._state.equity
        self._state.peak_equity = max(self._state.peak_equity, self._state.equity)

        # --- Reward Shaping ---
        reward += self._compute_shaped_reward(prev_equity)

        # Advance step
        self._step_idx += 1
        self._state.bars_since_last_loss += 1

        # Done conditions
        done = self._step_idx >= self.max_bars - 1
        drawdown = (self._state.peak_equity - self._state.equity) / max(self._state.peak_equity, 1)
        if drawdown > 0.05:
            reward -= 5.0  # Circuit breaker penalty
            done = True

        obs = self._get_obs()
        return obs, float(reward), done, False, info

    def _get_price(self, offset: int = 0) -> float:
        """Get close price at current step + offset."""
        idx = min(self._step_idx + offset, len(self.feature_matrix) - 1)
        if self.feature_matrix.ndim > 1 and self.feature_matrix.shape[1] > 3:
            return float(self.feature_matrix[idx, 3])  # close price column
        return float(self.feature_matrix[idx, 0]) if self.feature_matrix.ndim > 1 else float(self.feature_matrix[idx])

    def _open_position(self, direction: int, price: float) -> None:
        self._state.position = direction
        self._state.entry_price = price
        self._state.bars_in_trade = 0
        self._state.unrealized_pnl = 0.0
        self._state.total_trades += 1

    def _close_position(self, price: float) -> float:
        """Close current position, return realized PnL."""
        if self._state.position == 0:
            return 0.0

        price_diff = (price - self._state.entry_price) * self._state.position
        pnl = price_diff * self.pip_value * (self.lot_size / self.point_value)
        pnl -= self.commission * self.lot_size  # Commission

        self._state.realized_pnl += pnl
        self._state.equity += pnl

        if pnl > 0:
            self._state.winning_trades += 1
            self._state.consecutive_losses = 0
        else:
            self._state.consecutive_losses += 1
            self._state.bars_since_last_loss = 0

        # Reset position
        self._state.position = 0
        self._state.entry_price = 0.0
        self._state.unrealized_pnl = 0.0
        self._state.bars_in_trade = 0

        return pnl

    def _compute_shaped_reward(self, prev_equity: float) -> float:
        """Additional reward shaping beyond raw PnL."""
        reward = 0.0

        # Drawdown penalty (quadratic)
        dd = max(0, self._state.peak_equity - self._state.equity) / max(self._state.peak_equity, 1)
        reward -= 2.0 * dd * dd

        # Anti-revenge trading: penalty for re-entering too soon after a loss
        if self._state.bars_since_last_loss < 5 and abs(self._state.position) > 0:
            reward -= 0.3

        # Holding penalty (discourage over-holding)
        if self._state.bars_in_trade > 60:
            reward -= 0.01 * (self._state.bars_in_trade - 60)

        return reward

    def _get_obs(self) -> np.ndarray:
        """Build observation vector."""
        idx = min(self._step_idx, len(self.feature_matrix) - 1)
        features = self.feature_matrix[idx] if self.feature_matrix.ndim > 1 else np.array([self.feature_matrix[idx]])

        # Position state (normalized)
        state_features = np.array([
            float(self._state.position),
            self._state.unrealized_pnl / max(self.initial_equity, 1) * 100,
            min(self._state.bars_in_trade / 60.0, 1.0),
            min(self._state.consecutive_losses / 5.0, 1.0),
        ], dtype=np.float32)

        obs = np.concatenate([features.astype(np.float32), state_features])
        return np.clip(obs, -10.0, 10.0)


# ======================================================================
# RL Agent Wrapper
# ======================================================================

class RLAgent:
    """
    High-level RL agent wrapper.
    Manages training, prediction, and model persistence.
    """

    def __init__(
        self,
        env: TradingEnv | None = None,
        algorithm: str = "PPO",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        device: str = "auto",
    ):
        self.algorithm = algorithm
        self.env = env
        self.agent = None

        if env is not None and SB3_AVAILABLE:
            AgentClass = PPO if algorithm.upper() == "PPO" else SAC
            self.agent = AgentClass(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps if algorithm.upper() == "PPO" else 1,
                batch_size=batch_size,
                n_epochs=n_epochs if algorithm.upper() == "PPO" else 1,
                gamma=gamma,
                verbose=0,
                device=device,
            )
            logger.info(f"RL Agent initialized | algo={algorithm} | lr={learning_rate}")

    def predict_action(self, obs: np.ndarray) -> int:
        """Predict best action from observation."""
        if self.agent is None:
            return 0  # Hold by default
        action, _ = self.agent.predict(obs, deterministic=True)
        return int(action)

    def train(self, total_timesteps: int = 500000) -> None:
        """Train the RL agent."""
        if self.agent is None:
            logger.error("Agent not initialized, cannot train")
            return

        logger.info(f"Starting RL training | timesteps={total_timesteps}")
        self.agent.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )
        logger.info("RL training complete")

    def save(self, path: str | Path) -> None:
        """Save trained agent."""
        if self.agent is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        logger.info(f"RL agent saved to {path}")

    def load(self, path: str | Path, env: TradingEnv | None = None) -> None:
        """Load trained agent."""
        path = Path(path)
        if not path.exists() and not Path(str(path) + ".zip").exists():
            logger.warning(f"Agent file not found: {path}")
            return
        if not SB3_AVAILABLE:
            logger.error("stable-baselines3 not available")
            return
        AgentClass = PPO if self.algorithm.upper() == "PPO" else SAC
        self.agent = AgentClass.load(str(path), env=env or self.env)
        logger.info(f"RL agent loaded from {path}")
