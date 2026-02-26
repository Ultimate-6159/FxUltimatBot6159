# FxUltimatBot6159 — Adaptive Multi-Dimensional AI Trading Bot

> HFT/Scalping Bot สำหรับ XAUUSDm ที่ใช้ Ensemble AI (LSTM/Transformer + Reinforcement Learning) พร้อมระบบ **Broker-Proof** (Virtual TP/SL + Spread Guard) สำหรับรันบน Vultr VPS 24/7

## 🏗 Architecture

```
Data Pipeline → AI Core → Risk Manager → Execution Engine → MT5
    │               │            │              │
    ├─ Tick Data     ├─ LSTM/TF   ├─ Drawdown    ├─ Virtual TP/SL
    ├─ Multi-TF      ├─ RL Agent   ├─ Kelly Size   ├─ Spread Guard
    ├─ Features      └─ Ensemble   └─ Circuit Brk  └─ Stealth Orders
    └─ Sentiment
```

## 🛡 Broker-Proof Features

| Feature | Description |
|---------|-------------|
| **Virtual TP/SL** | TP/SL ซ่อนอยู่ใน memory ของ bot — broker ไม่เห็น |
| **Spread Guard** | Z-score anomaly detection — หยุดเทรดเมื่อ spread ถ่าง |
| **Slippage Track** | ตรวจจับ slippage ที่ผิดปกติและหยุดเทรด |
| **Stealth Orders** | ส่ง market order โดยไม่แนบ TP/SL |

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. Configure MT5 credentials
# Edit config/default.yaml → mt5 section

# 3. Train AI models
python scripts/train.py --model all

# 4. Run backtest
python scripts/backtest_run.py

# 5. Start paper trading
python scripts/live.py --paper

# 6. Start live trading
python scripts/live.py
```

## 📁 Project Structure

```
├── config/          # YAML configuration files
├── src/
│   ├── data/        # Data pipeline (tick, OHLCV, features, sentiment)
│   ├── models/      # AI models (LSTM/Transformer, RL, ensemble)
│   ├── execution/   # Broker-proof execution (virtual TP/SL, spread guard)
│   ├── risk/        # Risk management (drawdown, position sizing)
│   └── orchestrator/# Main trading loop coordinator
├── backtest/        # Backtesting engine + metrics
├── scripts/         # Entry points (live, train, backtest)
├── tests/           # Unit tests
├── docs/            # VPS setup guide
└── models/          # Saved AI model weights
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📖 Documentation

- [VPS Setup Guide](docs/vps_setup.md)
