# strategies/Example_DNA_v1.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Опционально: общие фичи
# Если не хотите зависеть от libs.features, можно оставить собственные реализации ниже.
try:
    from libs.features import add_basic_features, atr as atr_func, rsi as rsi_func
except Exception:
    add_basic_features = None
    atr_func = None
    rsi_func = None


STRATEGY_NAME = "Example DNA v1"

# Пример политик выхода: fixed RR и ATR-based
EXIT_POLICIES: List[Dict[str, Any]] = [
    {"name": "fixed_rr_1_2", "type": "fixed_rr", "rr": 2.0},      # риск:1, цель:2
    {"name": "atr_2x", "type": "atr", "atr_mult": 2.0},           # TP/SL по ATR-множителям
]

# Описание задач
TASKS: Dict[str, Any] = {
    "entry_action": {"type": "classification", "classes": ["SKIP", "BUY", "SELL"]},
    "policy_choice": {"type": "classification", "classes": [p["name"] for p in EXIT_POLICIES]},
    "level_quality": {"type": "classification", "classes": [0, 1]},  # бинарный фильтр качества
}


def _ensure_schema(df: pd.DataFrame) -> None:
    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input df missing columns: {missing}")
    # Требуем сортировку и уникальность времени
    if not df["open_time"].is_monotonic_increasing:
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
    if df["open_time"].duplicated().any():
        raise ValueError("Duplicate open_time detected")


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    if atr_func is not None:
        return atr_func(df, period=period)
    # Локальная реализация ATR (простая SMA по TR)
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean().rename(f"atr_{period}")


def _rsi(close: pd.Series, period: int) -> pd.Series:
    if rsi_func is not None:
        return rsi_func(close, period=period)
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def calculate_features(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    На вход: df с open_time, open, high, low, close, volume.
    На выход: df с добавленными признаками, без утечек будущего.
    """
    _ensure_schema(df)
    out = df.copy()

    # Параметры
    sma_periods = meta.get("sma_periods", [10, 20, 50])
    ema_periods = meta.get("ema_periods", [12, 26])
    atr_period = int(meta.get("atr_period", 14))
    rsi_period = int(meta.get("rsi_period", 14))
    vol_period = int(meta.get("vol_period", 20))

    if add_basic_features is not None:
        out = add_basic_features(out, meta)
    else:
        # Минимальный набор, если libs.features недоступен
        for n in sma_periods:
            out[f"sma_{n}"] = out["close"].rolling(n, min_periods=n).mean()
        for n in ema_periods:
            out[f"ema_{n}"] = out["close"].ewm(span=n, adjust=False, min_periods=n).mean()

        out[f"atr_{atr_period}"] = _atr(out, atr_period)
        out[f"rsi_{rsi_period}"] = _rsi(out["close"], rsi_period)
        out[f"vol_{vol_period}"] = out["close"].pct_change().rolling(vol_period, min_periods=vol_period).std()

        for n in sma_periods:
            out[f"close_sma_{n}_rel"] = out["close"] / out[f"sma_{n}"]
        for n in ema_periods:
            out[f"close_ema_{n}_rel"] = out["close"] / out[f"ema_{n}"]

        out["hl_range"] = out["high"] - out["low"]
        out["body"] = (out["close"] - out["open"]).abs()
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

    return out


def generate_ideas(df_with_features: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Добавляет эвристические идеи/сигналы без заглядывания вперёд.
    Примеры:
      - entry_action: BUY при RSI<30 и close > sma_20; SELL при RSI>70 и close < sma_20; иначе SKIP.
      - policy_choice: выбираем из EXIT_POLICIES по простому правилу (напр., vol).
      - level_quality: 1 если ATR не слишком мал и vol в диапазоне, иначе 0.
    """
    df = df_with_features.copy()
    atr_period = int(meta.get("atr_period", 14))
    rsi_period = int(meta.get("rsi_period", 14))
    sma_ref = int(meta.get("sma_ref", 20))

    rsi_col = f"rsi_{rsi_period}"
    sma_col = f"sma_{sma_ref}"
    if sma_col not in df.columns:
        df[sma_col] = df["close"].rolling(sma_ref, min_periods=sma_ref).mean()
    if rsi_col not in df.columns:
        df[rsi_col] = _rsi(df["close"], rsi_period)
    atr_col = f"atr_{atr_period}"
    if atr_col not in df.columns:
        df[atr_col] = _atr(df, atr_period)

    # entry_action
    entry = np.zeros(len(df), dtype=int)  # 0=SKIP,1=BUY,2=SELL
    buy_mask = (df[rsi_col] < 30) & (df["close"] > df[sma_col])
    sell_mask = (df[rsi_col] > 70) & (df["close"] < df[sma_col])
    entry[buy_mask.values] = 1
    entry[sell_mask.values] = 2
    df["entry_action"] = entry

    # policy_choice: если волатильность высокая — ATR-политика, иначе fixed RR
    vol = df.get("vol_20")
    if vol is None:
        vol = df["close"].pct_change().rolling(20, min_periods=20).std()
    policy_idx = np.zeros(len(df), dtype=int)
    high_vol = vol > vol.rolling(200, min_periods=50).median()
    # 0 -> fixed_rr_1_2, 1 -> atr_2x
    policy_idx[high_vol.fillna(False).values] = 1
    df["policy_choice"] = policy_idx

    # level_quality: 1 если ATR не слишком маленький и не NaN, иначе 0
    atr_ok = df[atr_col] > (df[atr_col].rolling(200, min_periods=50).median() * 0.5)
    lvl = np.where(atr_ok.fillna(False).values, 1, 0)
    df["level_quality"] = lvl

    # Для удобства в дальнейшем инференсе: сразу подготовим рекомендованные уровни SL/TP (но без look-ahead!)
    # На основе выбранной политики и цены закрытия текущего бара (вход подразумевается по следующему бару)
    close = df["close"].to_numpy()
    atr_vals = df[atr_col].to_numpy()
    sl_price = np.full(len(df), np.nan, dtype=float)
    tp_price = np.full(len(df), np.nan, dtype=float)

    for i in range(len(df)):
        side = entry[i]
        if side == 0 or np.isnan(close[i]):
            continue
        pol = policy_idx[i]
        if pol == 0:
            # fixed RR: допустим риск = 1*ATR, цель = rr*ATR
            risk_atr = atr_vals[i]
            rr = 2.0
            if np.isnan(risk_atr):
                continue
            if side == 1:
                sl_price[i] = close[i] - risk_atr
                tp_price[i] = close[i] + risk_atr * rr
            else:
                sl_price[i] = close[i] + risk_atr
                tp_price[i] = close[i] - risk_atr * rr
        else:
            # atr-based: múltiplicатор на обе стороны
            m = 2.0
            if np.isnan(atr_vals[i]):
                continue
            if side == 1:
                sl_price[i] = close[i] - atr_vals[i] * m
                tp_price[i] = close[i] + atr_vals[i] * m
            else:
                sl_price[i] = close[i] + atr_vals[i] * m
                tp_price[i] = close[i] - atr_vals[i] * m

    df["sl_price"] = sl_price
    df["tp_price"] = tp_price

    return df


def build_labels(df_with_ideas: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, pd.Series]:
    """
    Формирует таргеты для обучения.
    - entry_action: уже эвристика (0/1/2), можно обучать модель повторять её или модифицировать правило.
    - policy_choice: индекс политики (int)
    - level_quality: бинарный фильтр (0/1)
    Допускается заменить на более строгую/другую логику.
    """
    df = df_with_ideas
    labels = {
        "entry_action": df["entry_action"].astype("Int64"),
        "policy_choice": df["policy_choice"].astype("Int64"),
        "level_quality": df["level_quality"].astype("Int64"),
    }
    return labels


def feature_columns(df_with_features: pd.DataFrame) -> List[str]:
    """
    Возвращает список колонок-признаков (в порядке).
    Выбираем безопасные признаки без утечек.
    """
    candidates = []
    for col in df_with_features.columns:
        if col in ("open_time", "open", "high", "low", "close", "volume"):
            continue
        # исключаем явные таргеты/идеи, если они уже есть в df_with_features (обычно их добавляют позже)
        if col in ("entry_action", "policy_choice", "level_quality", "sl_price", "tp_price"):
            continue
        # берем общие фичи
        candidates.append(col)

    # Можно упорядочить, например, по имени
    candidates = sorted(candidates)
    return candidates


def inference_inputs(df_with_features: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    """
    Формирует X строго по списку feats.
    """
    X = df_with_features.loc[:, feats].copy()
    return X
