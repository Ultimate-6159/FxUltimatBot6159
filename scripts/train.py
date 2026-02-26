#!/usr/bin/env python3
"""
AI Model Training Script.
Trains LSTM/Transformer and RL agent on historical or synthetic data.

Usage:
    python scripts/train.py                          # Default training
    python scripts/train.py --model lstm              # Train LSTM only
    python scripts/train.py --model rl                # Train RL only
    python scripts/train.py --timesteps 100000        # Custom RL steps
    python scripts/train.py --epochs 100              # Custom LSTM epochs
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.tick_collector import TickCollector
from src.data.multi_tf_loader import MultiTimeframeLoader
from src.data.feature_engine import FeatureEngine
from src.models.lstm_transformer import LSTMTransformerPredictor
from src.models.rl_agent import RLAgent, TradingEnv
from src.utils.config import load_config, get_nested
from src.utils.logger import setup_logger


def train_lstm(cfg: dict, epochs: int = 50, batch_size: int = 64) -> None:
    """Train the LSTM/Transformer model on historical data."""
    logger = setup_logger("FxBot.Train")
    logger.info("=" * 50)
    logger.info("  Training LSTM/Transformer Model")
    logger.info("=" * 50)

    # Generate or load training data
    logger.info("Generating synthetic training data...")
    tf_loader = MultiTimeframeLoader()
    tf_data = tf_loader.generate_synthetic_data(n_bars=5000)

    feature_engine = FeatureEngine(lookback=60)
    features_df = feature_engine.compute_all_features(tf_data.m1)
    feature_matrix = features_df.values.astype(np.float32)

    logger.info(f"Feature matrix shape: {feature_matrix.shape}")

    # Initialize model
    lstm_cfg = cfg.get("ai", {}).get("lstm_transformer", {})
    predictor = LSTMTransformerPredictor(
        input_dim=feature_matrix.shape[1],
        d_model=lstm_cfg.get("d_model", 64),
        n_heads=lstm_cfg.get("n_heads", 4),
        seq_length=lstm_cfg.get("seq_length", 60),
        learning_rate=lstm_cfg.get("learning_rate", 3e-4),
    )

    # Prepare training sequences
    sequences, labels = predictor.prepare_sequences(
        feature_matrix, lookahead=5, threshold=0.0005
    )

    if len(sequences) == 0:
        logger.error("No training sequences generated")
        return

    # Split train/val
    split = int(0.8 * len(sequences))
    train_x, val_x = sequences[:split], sequences[split:]
    train_y, val_y = labels[:split], labels[split:]

    logger.info(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")
    logger.info(f"Label distribution: up={sum(labels==0)}, down={sum(labels==1)}, neutral={sum(labels==2)}")

    # Training loop
    best_loss = float("inf")
    for epoch in range(epochs):
        start = time.time()
        train_loss = predictor.train_epoch(train_x, train_y, batch_size=batch_size)
        elapsed = time.time() - start

        if (epoch + 1) % 5 == 0 or epoch == 0:
            # Validation
            val_preds = []
            for i in range(0, len(val_x), batch_size):
                batch = val_x[i:i+batch_size]
                for sample in batch:
                    pred = predictor.predict(sample)
                    val_preds.append(pred.direction)

            val_acc = np.mean(
                np.array(val_preds[:len(val_y)]) == np.array([{0:1, 1:-1, 2:0}[l] for l in val_y[:len(val_preds)]])
            )

            logger.info(
                f"Epoch {epoch+1:3d}/{epochs} | loss={train_loss:.4f} | "
                f"val_acc={val_acc:.3f} | time={elapsed:.1f}s"
            )

            if train_loss < best_loss:
                best_loss = train_loss
                predictor.save("models/lstm_transformer.pt")

    logger.info(f"LSTM training complete | Best loss: {best_loss:.4f}")


def train_rl(cfg: dict, timesteps: int = 500000) -> None:
    """Train the RL agent in simulated environment."""
    logger = setup_logger("FxBot.Train")
    logger.info("=" * 50)
    logger.info("  Training RL Agent")
    logger.info("=" * 50)

    # Generate training data
    logger.info("Generating synthetic environment data...")
    tf_loader = MultiTimeframeLoader()
    tf_data = tf_loader.generate_synthetic_data(n_bars=10000)

    feature_engine = FeatureEngine(lookback=60)
    features_df = feature_engine.compute_all_features(tf_data.m1)
    feature_matrix = features_df.values.astype(np.float32)

    logger.info(f"Environment feature matrix: {feature_matrix.shape}")

    # Create trading environment
    env = TradingEnv(
        feature_matrix=feature_matrix,
        initial_equity=10000.0,
        spread_cost=3.0,
        commission_per_lot=7.0,
    )

    # Initialize and train agent
    rl_cfg = cfg.get("ai", {}).get("rl_agent", {})
    agent = RLAgent(
        env=env,
        algorithm=rl_cfg.get("algorithm", "PPO"),
        learning_rate=rl_cfg.get("learning_rate", 3e-4),
        n_steps=rl_cfg.get("n_steps", 2048),
        batch_size=rl_cfg.get("batch_size", 64),
        gamma=rl_cfg.get("gamma", 0.99),
    )

    logger.info(f"Starting RL training | timesteps={timesteps}")
    agent.train(total_timesteps=timesteps)
    agent.save("models/rl_agent")
    logger.info("RL training complete")


def main():
    parser = argparse.ArgumentParser(description="Train AI Models")
    parser.add_argument("--model", choices=["lstm", "rl", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config()
    Path("models").mkdir(exist_ok=True)

    if args.model in ("lstm", "all"):
        train_lstm(cfg, epochs=args.epochs, batch_size=args.batch_size)

    if args.model in ("rl", "all"):
        train_rl(cfg, timesteps=args.timesteps)


if __name__ == "__main__":
    main()
