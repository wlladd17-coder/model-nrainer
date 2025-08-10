# libs/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _sma(a: pd.Series, n: int) -> pd.Series:
    return a.rolling(window=n, min_periods=n).mean()


def _ema(a: pd.Series, n: int) -> pd.Series:
    return a.ewm(span=n, adjust=False, min_periods=n).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14, col_prefix: str = "atr") -> pd.Series:
    tr = _true_range(df["high"], df["low"], df["close"])
    atr_val = tr.rolling(window=period, min_periods=period).mean()
    atr_val.name = f"{col_prefix}_{period}"
    return atr_val


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val.name = f"rsi_{period}"
    return rsi_val


def volatility(close: pd.Series, period: int = 20) -> pd.Series:
    vol = close.pct_change().rolling(period, min_periods=period).std()
    vol.name = f"vol_{period}"
    return vol


def rolling_minmax(low: pd.Series, high: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series]:
    rmin = low.rolling(period, min_periods=period).min()
    rmax = high.rolling(period, min_periods=period).max()
    rmin.name = f"low_min_{period}"
    rmax.name = f"high_max_{period}"
    return rmin, rmax


def add_basic_features(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Add common features used by multiple DNAs.
    meta keys (optional):
      - sma_periods: list[int]
      - ema_periods: list[int]
      - rsi_period: int
      - atr_period: int
      - vol_period: int
    """
    out = df.copy()
    sma_periods = meta.get("sma_periods", [10, 20, 50])
    ema_periods = meta.get("ema_periods", [12, 26])
    rsi_period = int(meta.get("rsi_period", 14))
    atr_period = int(meta.get("atr_period", 14))
    vol_period = int(meta.get("vol_period", 20))

    for n in sma_periods:
        out[f"sma_{n}"] = _sma(out["close"], n)
    for n in ema_periods:
        out[f"ema_{n}"] = _ema(out["close"], n)

    out[f"atr_{atr_period}"] = atr(out, period=atr_period)
    out[f"rsi_{rsi_period}"] = rsi(out["close"], period=rsi_period)
    out[f"vol_{vol_period}"] = volatility(out["close"], period=vol_period)

    # price relative to MAs
    for n in sma_periods:
        out[f"close_sma_{n}_rel"] = out["close"] / out[f"sma_{n}"]
    for n in ema_periods:
        out[f"close_ema_{n}_rel"] = out["close"] / out[f"ema_{n}"]

    # range features
    out["hl_range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    return out
