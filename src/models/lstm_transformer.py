"""
Hybrid LSTM + Transformer model for micro-trend forecasting.
Predicts short-term price direction and confidence from Price Action features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger("FxBot.LSTMTransformer")


@dataclass
class Prediction:
    """Model prediction output."""
    direction: int           # +1 (up), -1 (down), 0 (neutral)
    confidence: float        # 0.0 - 1.0
    probabilities: dict[str, float]  # {"up": 0.7, "down": 0.2, "neutral": 0.1}


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LSTMTransformerModel(nn.Module):
    """
    Hybrid LSTM + Transformer for time-series price prediction.

    Architecture:
    1. Input projection → d_model dimension
    2. LSTM layers → capture sequential dependencies
    3. Positional encoding
    4. Transformer encoder → multi-head self-attention
    5. Classification head → 3 classes (up, down, neutral)
    """

    def __init__(
        self,
        input_dim: int = 22,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        seq_length: int = 60,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_length = seq_length

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_length, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 3),  # 3 classes: up, down, neutral
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch, seq_length, input_dim) feature tensor.

        Returns:
            (batch, 3) logits for [up, down, neutral].
        """
        # Project input to model dimension
        x = self.input_proj(x)  # (B, T, d_model)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoding(lstm_out)

        # Transformer self-attention
        x = self.transformer(x)  # (B, T, d_model)

        # Use last time step for prediction
        x = x[:, -1, :]  # (B, d_model)

        # Classification
        logits = self.classifier(x)  # (B, 3)
        return logits


class LSTMTransformerPredictor:
    """
    High-level wrapper for the LSTM+Transformer model.
    Handles training, inference, model saving/loading.
    """

    def __init__(
        self,
        input_dim: int = 22,
        d_model: int = 64,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        n_transformer_layers: int = 2,
        dropout: float = 0.1,
        seq_length: int = 60,
        learning_rate: float = 3e-4,
        device: str | None = None,
    ):
        self.seq_length = seq_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMTransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_lstm_layers=n_lstm_layers,
            n_transformer_layers=n_transformer_layers,
            dropout=dropout,
            seq_length=seq_length,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.tensor([1.0, 1.0, 0.5], device=self.device)  # Down-weight neutral
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        logger.info(
            f"LSTMTransformer initialized | input_dim={input_dim} | "
            f"d_model={d_model} | device={self.device} | "
            f"params={sum(p.numel() for p in self.model.parameters()):,}"
        )

    def predict(self, features: np.ndarray) -> Prediction:
        """
        Make a single prediction from feature array.

        Args:
            features: (seq_length, input_dim) numpy array of features.

        Returns:
            Prediction with direction, confidence, and probabilities.
        """
        self.model.eval()
        with torch.no_grad():
            # Ensure correct shape
            if features.ndim == 2:
                features = features[np.newaxis, :]  # Add batch dim

            # Pad/truncate to seq_length
            if features.shape[1] < self.seq_length:
                pad = np.zeros((features.shape[0], self.seq_length - features.shape[1], features.shape[2]))
                features = np.concatenate([pad, features], axis=1)
            elif features.shape[1] > self.seq_length:
                features = features[:, -self.seq_length:, :]

            x = torch.FloatTensor(features).to(self.device)
            logits = self.model(x)  # (1, 3)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Direction: argmax → 0=up, 1=down, 2=neutral
        direction_map = {0: 1, 1: -1, 2: 0}
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

        return Prediction(
            direction=direction_map[pred_class],
            confidence=confidence,
            probabilities={"up": float(probs[0]), "down": float(probs[1]), "neutral": float(probs[2])},
        )

    def train_step(self, batch_x: np.ndarray, batch_y: np.ndarray) -> float:
        """
        Single training step.

        Args:
            batch_x: (batch, seq_length, input_dim) features.
            batch_y: (batch,) labels — 0=up, 1=down, 2=neutral.

        Returns:
            Training loss value.
        """
        self.model.train()
        x = torch.FloatTensor(batch_x).to(self.device)
        y = torch.LongTensor(batch_y).to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def train_epoch(
        self, features: np.ndarray, labels: np.ndarray, batch_size: int = 64
    ) -> float:
        """
        Train one full epoch.

        Args:
            features: (N, seq_length, input_dim) all training sequences.
            labels: (N,) labels.
            batch_size: Mini-batch size.

        Returns:
            Average epoch loss.
        """
        n = len(features)
        indices = np.random.permutation(n)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            loss = self.train_step(features[batch_idx], labels[batch_idx])
            total_loss += loss
            n_batches += 1

        self.scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss

    def prepare_sequences(
        self,
        feature_matrix: np.ndarray,
        lookahead: int = 5,
        threshold: float = 0.001,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create training sequences from feature matrix.

        Args:
            feature_matrix: (N, n_features) raw feature matrix.
            lookahead: Bars ahead for label generation.
            threshold: Min price change for direction label.

        Returns:
            (sequences, labels) where sequences.shape = (M, seq_length, n_features)
        """
        n_samples = len(feature_matrix) - self.seq_length - lookahead
        if n_samples <= 0:
            return np.array([]), np.array([])

        sequences = []
        labels = []

        # Use close price (assumed to be index 3 in feature matrix — m1_close)
        close_idx = 3  # position of close in feature columns

        for i in range(n_samples):
            seq = feature_matrix[i : i + self.seq_length]
            sequences.append(seq)

            # Label: price direction over lookahead bars
            current_close = feature_matrix[i + self.seq_length - 1, close_idx]
            future_close = feature_matrix[i + self.seq_length + lookahead - 1, close_idx]
            pct_change = (future_close - current_close) / max(abs(current_close), 1e-10)

            if pct_change > threshold:
                labels.append(0)  # Up
            elif pct_change < -threshold:
                labels.append(1)  # Down
            else:
                labels.append(2)  # Neutral

        return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.int64)

    def save(self, path: str | Path) -> None:
        """Save model weights and architecture config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": {
                "input_dim": self.model.input_dim,
                "d_model": self.model.d_model,
                "seq_length": self.seq_length,
            },
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> None:
        """Load model weights."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Model file not found: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        logger.info(f"Model loaded from {path}")

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "LSTMTransformerPredictor":
        """
        Load a predictor from a saved checkpoint, using the saved architecture config.
        Falls back to detecting input_dim from saved weight shapes.

        Args:
            path: Path to the saved .pt file.
            device: Device to load model on.

        Returns:
            LSTMTransformerPredictor with loaded weights.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
        cfg = checkpoint.get("config", {})

        # Detect input_dim from weight shapes if config not saved (legacy checkpoints)
        if "input_dim" not in cfg:
            state = checkpoint["model_state"]
            # input_proj is nn.Linear(input_dim, d_model) → weight shape = (d_model, input_dim)
            input_dim = state["input_proj.weight"].shape[1]
            d_model = state["input_proj.weight"].shape[0]
            cfg["input_dim"] = input_dim
            cfg["d_model"] = d_model

        predictor = cls(
            input_dim=cfg.get("input_dim", 24),
            d_model=cfg.get("d_model", 64),
            seq_length=cfg.get("seq_length", 60),
            device=device,
        )
        predictor.model.load_state_dict(checkpoint["model_state"])
        logger.info(f"Model loaded from checkpoint {path} | input_dim={cfg['input_dim']}")
        return predictor
