# strategies/Level_Retest_v1.py
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


STRATEGY_NAME = "Level_Retest_v1"

EXIT_POLICIES: List[Dict[str, Any]] = [
    {"name": "Policy_0_Fixed_SL20_RR2", "type": "fixed", "params": {"sl_pips": 20.0, "rr": 2.0}},
    {"name": "Policy_1_Fixed_SL40_RR3", "type": "fixed", "params": {"sl_pips": 40.0, "rr": 3.0}},
    {"name": "Policy_2_ATR_SL1.5_RR2", "type": "atr",   "params": {"sl_multiplier": 1.5, "rr": 2.0}},
    {"name": "Policy_3_ATR_SL2.0_RR3", "type": "atr",   "params": {"sl_multiplier": 2.0, "rr": 3.0}},
]

TASKS: Dict[str, Any] = {
    "entry_action": {"type": "classification", "classes": ["SKIP", "BUY", "SELL"]},
    "policy_choice": {"type": "classification", "classes": [p["name"] for p in EXIT_POLICIES]},
    "level_quality": {"type": "classification", "classes": [0, 1]},
}


def _ensure_schema(df: pd.DataFrame) -> None:
    req = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not df["open_time"].is_monotonic_increasing:
        df.sort_values("open_time", inplace=True)
        df.reset_index(drop=True, inplace=True)
    if df["open_time"].duplicated().any():
        raise ValueError("Duplicate open_time detected")


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=period).mean().rename(f"atr_{period}")


def calculate_features(df: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Строит базовые фичи и карту swing-уровней.
    Допустимо center=True на этом этапе (это не сигналы, а "карта").
    """
    _ensure_schema(df)
    out = df.copy()

    atr_period = int(meta.get("atr_period", 14))
    swing_n = int(meta.get("swing_n", 10))

    out[f"atr_{atr_period}"] = _atr(out, atr_period)
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["hl_range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()

    # Swing highs/lows карта (разрешено центрирование, т.к. это не используется как сигнал напрямую)
    N = swing_n
    sw_high = out["high"].rolling(2 * N + 1, center=True).apply(
        lambda x: x[N] == np.max(x), raw=True
    )
    sw_low = out["low"].rolling(2 * N + 1, center=True).apply(
        lambda x: x[N] == np.min(x), raw=True
    )
    out["is_swing_high"] = sw_high.fillna(0).astype(bool)
    out["is_swing_low"] = sw_low.fillna(0).astype(bool)

    # Удаляем начальные NaN у фич
    return out


def generate_ideas(df_with_features: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Генерирует сигналы без использования будущих баров.
    Идея: вход при ретесте ранее сформированного swing-уровня.
    """
    df = df_with_features.copy()
    atr_period = int(meta.get("atr_period", 14))
    touch_k = float(meta.get("touch_k", 0.25))  # доля ATR для зоны касания

    # Собираем уровни, СТРОГО прошлого относительно текущего бара
    sup = df.loc[df["is_swing_low"], ["open_time", "low"]].rename(columns={"low": "level_price"}).set_index("open_time")
    res = df.loc[df["is_swing_high"], ["open_time", "high"]].rename(columns={"high": "level_price"}).set_index("open_time")

    entry = np.zeros(len(df), dtype=int)  # 0=SKIP,1=BUY,2=SELL
    policy_idx = np.zeros(len(df), dtype=int)
    level_quality = np.zeros(len(df), dtype=int)
    sl_price = np.full(len(df), np.nan)
    tp_price = np.full(len(df), np.nan)

    atr_col = f"atr_{atr_period}"
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    atr_vals = df[atr_col].to_numpy()

    # Скалярная проходка без утечки: используем только уровни с индексом < текущего времени
    for i in range(1, len(df)):
        t = df.loc[i, "open_time"]
        atr_i = atr_vals[i]
        if np.isnan(atr_i):
            continue

        touch_zone = atr_i * touch_k

        past_sup = sup.loc[:t - pd.Timedelta(nanoseconds=1)] if not sup.empty else sup
        past_res = res.loc[:t - pd.Timedelta(nanoseconds=1)] if not res.empty else res

        # BUY при касании поддержки
        if not past_sup.empty:
            # ближайший уровень к текущему low
            diff = np.abs(past_sup["level_price"] - low[i])
            j = diff.idxmin() if not diff.isna().all() else None
            if j is not None:
                lvl = float(past_sup.loc[j, "level_price"])
                if abs(low[i] - lvl) <= touch_zone:
                    entry[i] = 1
                    level_quality[i] = 1

        # SELL при касании сопротивления (если не BUY)
        if entry[i] == 0 and not past_res.empty:
            diff = np.abs(past_res["level_price"] - high[i])
            j = diff.idxmin() if not diff.isna().all() else None
            if j is not None:
                lvl = float(past_res.loc[j, "level_price"])
                if abs(high[i] - lvl) <= touch_zone:
                    entry[i] = 2
                    level_quality[i] = 1

        # Выбор политики: при более высокой волатильности — ATR-политики
        if level_quality[i] == 1:
            if atr_i > np.nanmedian(atr_vals[max(0, i - 200):i+1]):
                # ATR политики: выберем Policy_2 (индекс 2)
                policy_idx[i] = 2
            else:
                policy_idx[i] = 0  # Fixed RR

            # Сразу рассчитаем SL/TP от текущей цены закрытия (вход на следующем баре)
            if entry[i] == 1:  # BUY
                if policy_idx[i] in (0, 1):
                    # fixed по пунктам: sl_pips в абсолютном выражении цены инструмента
                    if policy_idx[i] == 0:
                        sl_pips, rr = 20.0, 2.0
                    else:
                        sl_pips, rr = 40.0, 3.0
                    sl_price[i] = close[i] - sl_pips
                    tp_price[i] = close[i] + sl_pips * rr
                else:
                    mult = 1.5 if policy_idx[i] == 2 else 2.0
                    rr = 2.0 if policy_idx[i] == 2 else 3.0
                    sl_price[i] = close[i] - atr_i * mult
                    tp_price[i] = close[i] + atr_i * mult * rr
            elif entry[i] == 2:  # SELL
                if policy_idx[i] in (0, 1):
                    if policy_idx[i] == 0:
                        sl_pips, rr = 20.0, 2.0
                    else:
                        sl_pips, rr = 40.0, 3.0
                    sl_price[i] = close[i] + sl_pips
                    tp_price[i] = close[i] - sl_pips * rr
                else:
                    mult = 1.5 if policy_idx[i] == 2 else 2.0
                    rr = 2.0 if policy_idx[i] == 2 else 3.0
                    sl_price[i] = close[i] + atr_i * mult
                    tp_price[i] = close[i] - atr_i * mult * rr

    df["entry_action"] = entry
    df["policy_choice"] = policy_idx
    df["level_quality"] = level_quality
    df["sl_price"] = sl_price
    df["tp_price"] = tp_price

    return df


def build_labels(df_with_ideas: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, pd.Series]:
    return {
        "entry_action": df_with_ideas["entry_action"].astype("Int64"),
        "policy_choice": df_with_ideas["policy_choice"].astype("Int64"),
        "level_quality": df_with_ideas["level_quality"].astype("Int64"),
    }


def feature_columns(df_with_features: pd.DataFrame) -> List[str]:
    drop = {"open_time", "open", "high", "low", "close", "volume",
            "entry_action", "policy_choice", "level_quality", "sl_price", "tp_price"}
    feats = [c for c in df_with_features.columns if c not in drop]
    return sorted(feats)


def inference_inputs(df_with_features: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
    return df_with_features.loc[:, feats].copy()
