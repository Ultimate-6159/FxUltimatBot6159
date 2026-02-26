"""
YAML configuration loader with environment variable overrides.
Loads default.yaml + symbol-specific config and merges them.
"""

import os
from pathlib import Path
from typing import Any

import yaml


_CONFIG_CACHE: dict[str, Any] = {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(cfg: dict, prefix: str = "FXBOT") -> dict:
    """
    Override config values with environment variables.
    e.g., FXBOT_MT5_LOGIN=12345 → cfg['mt5']['login'] = 12345
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}_"):
            continue
        parts = key[len(prefix) + 1 :].lower().split("_")
        target = cfg
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Auto-cast to int/float/bool
        final_key = parts[-1]
        target[final_key] = _auto_cast(value)
    return cfg


def _auto_cast(value: str) -> Any:
    """Cast string to appropriate Python type."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    config_dir: str | Path | None = None,
    symbol: str = "xauusdm",
) -> dict[str, Any]:
    """
    Load and merge configuration files.

    Priority (highest → lowest):
    1. Environment variables (FXBOT_*)
    2. Symbol-specific config (config/symbols/{symbol}.yaml)
    3. Default config (config/default.yaml)

    Args:
        config_dir: Path to config directory. Defaults to ./config/
        symbol: Symbol name for symbol-specific config.

    Returns:
        Merged configuration dictionary.
    """
    cache_key = f"{config_dir}:{symbol}"
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    if config_dir is None:
        # Walk up from this file to find config/ directory
        project_root = Path(__file__).resolve().parent.parent.parent
        config_dir = project_root / "config"
    else:
        config_dir = Path(config_dir)

    # Load default config
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found: {default_path}")

    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Load symbol-specific config
    symbol_path = config_dir / "symbols" / f"{symbol.lower()}.yaml"
    if symbol_path.exists():
        with open(symbol_path, "r", encoding="utf-8") as f:
            symbol_cfg = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, symbol_cfg)

    # Apply environment variable overrides
    cfg = _apply_env_overrides(cfg)

    _CONFIG_CACHE[cache_key] = cfg
    return cfg


def get_nested(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """
    Get a nested config value using dot notation.
    e.g., get_nested(cfg, 'execution.virtual_tpsl.enabled', True)
    """
    keys = dotted_key.split(".")
    value = cfg
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value
