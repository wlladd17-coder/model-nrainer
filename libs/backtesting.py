# libs/backtesting.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, write_json, get_logger, timeit_ctx


@dataclass
class SimulationConfig:
    fee: float = 0.0004  # комиссия (доля за оборот в одну сторону)
    spread_abs: float = 0.0  # абсолютный спред (половина на каждую сторону)
    leverage: float = 1.0  # плечо, масштабирует PnL
    risk_mode: str = "fixed_cash"  # fixed_cash | pct_equity
    risk_value: float = 100.0  # $ или доля от капитала
    initial_equity: float = 10_000.0
    slippage_abs: float = 0.0  # дополнительный проскальзывание в абсолютных единицах


def _prepare_features_and_X(
    dna_module, df_raw: pd.DataFrame, meta: Dict[str, Any], feats: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_feat = dna_module.calculate_features(df_raw.copy(), meta)
    df_ideas = dna_module.generate_ideas(df_feat.copy(), meta)
    X = dna_module.inference_inputs(df_feat, feats)
    # выравнивание индексов по X
    df_ideas, X = df_ideas.align(X, join="inner", axis=0)
    return df_ideas, X


def _predict_with_model(
    model, X: pd.DataFrame, want_proba: bool
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    proba = None
    if want_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        preds = np.argmax(proba, axis=1)
        return preds, proba
    preds = model.predict(X)
    return preds, proba


def run_inference(
    dna_module,
    df_raw: pd.DataFrame,
    meta: Dict[str, Any],
    feats: List[str],
    model_or_ensemble: Dict[str, Any] | Any,
    logger=None,
) -> pd.DataFrame:
    """
    model_or_ensemble варианты:
      - одиночная модель и мета (.meta.json) для одной задачи -> ожидаем 'entry_action'
        - словарь вида:
          {
            "entry_action": {"model": clfA},
            "policy_choice": {"model": clfB},
            "level_quality": {"model": clfC},
            "weights": {
                "entry_action": 1.0,
                ...
            },  # для голосования (если несколько моделей на задачу)
            "vote": "soft" | "hard" | "weighted"
          }
    Выход: DataFrame сигналов с колонками:
      - entry_action_pred (str или int)
      - policy_choice_pred (str или int)
      - level_quality_pred (int/float)
    """
    if logger is None:
        logger = get_logger(name="backtesting")

    with timeit_ctx("features & ideas", logger):
        df_ideas, X = _prepare_features_and_X(dna_module, df_raw, meta, feats)

    # Приводим к унифицированной структуре
    ensemble = {}
    vote = None
    weights = {}
    tasks_schema = getattr(dna_module, "TASKS", {})

    if isinstance(model_or_ensemble, dict) and (
        "entry_action" in model_or_ensemble or "models" in model_or_ensemble
    ):
        # Возможны разные формы. Нормализуем:
        if "models" in model_or_ensemble:
            # список структур вида {"task": "...", "model": obj, "weight": 1.0}
            for item in model_or_ensemble["models"]:
                task = item["task"]
                ensemble.setdefault(task, []).append(
                    {"model": item["model"], "weight": float(item.get("weight", 1.0))}
                )
        else:
            for task in ("entry_action", "policy_choice", "level_quality"):
                v = model_or_ensemble.get(task)
                if v is None:
                    continue
                if isinstance(v, list):
                    ensemble[task] = [
                        {"model": it["model"], "weight": float(it.get("weight", 1.0))}
                        for it in v
                    ]
                else:
                    ensemble[task] = [
                        {"model": v["model"], "weight": float(v.get("weight", 1.0))}
                    ]
        vote = model_or_ensemble.get("vote")
        weights = model_or_ensemble.get("weights", {})
    else:
        # одиночная модель, считаем это entry_action
        ensemble["entry_action"] = [{"model": model_or_ensemble, "weight": 1.0}]

    preds_out: Dict[str, Any] = {}

    def predict_task(task: str):
        models = ensemble.get(task)
        if not models:
            return None
        want_proba = vote in ("soft", "weighted")
        # Собираем предсказания
        all_preds = []
        all_probas = []
        ws = []
        for mrec in models:
            model = mrec["model"]
            w = float(mrec.get("weight", 1.0))
            p, proba = _predict_with_model(model, X, want_proba=want_proba)
            all_preds.append(p)
            all_probas.append(proba)
            ws.append(w)
        all_preds = np.array(all_preds)  # [n_models, n_samples]
        if vote == "hard" or (vote is None and len(models) == 1):
            # простой мажоритарный
            if len(models) == 1:
                final = all_preds[0]
            else:
                # мажоритарно по каждому образцу
                final = []
                for j in range(all_preds.shape[1]):
                    vals, cnts = np.unique(all_preds[:, j], return_counts=True)
                    final.append(vals[np.argmax(cnts)])
                final = np.array(final)
            return final
        else:
            # soft/weighted голосование: усредняем вероятности
            if all_probas[0] is None:
                # fallback к hard
                return predict_task(
                    task
                )  # рекурсия приведет к hard ветке (т.к. want_proba False)
            # выясняем число классов по первой модели
            n_classes = all_probas[0].shape[1]
            proba_stack = []
            for k, proba in enumerate(all_probas):
                if proba is None:
                    continue
                w = ws[k] if vote == "weighted" else 1.0
                proba_stack.append(proba * w)
            prob_sum = np.sum(proba_stack, axis=0)
            # нормализация, если веса не нормированы
            prob_sum = prob_sum / (prob_sum.sum(axis=1, keepdims=True) + 1e-12)
            final = np.argmax(prob_sum, axis=1)
            return final

    for task in ("entry_action", "policy_choice", "level_quality"):
        res = predict_task(task)
        if res is not None:
            preds_out[f"{task}_pred"] = res

    signals = df_ideas.copy()
    for k, v in preds_out.items():
        signals[k] = v
    return signals


def _calc_position_size(price: float, equity: float, cfg: SimulationConfig) -> float:
    if cfg.risk_mode == "fixed_cash":
        cash = float(cfg.risk_value)
    else:
        cash = float(equity * cfg.risk_value)
    if price <= 0:
        return 0.0
    qty = cash / price
    return qty


def _apply_fees(price: float, qty: float, fee: float) -> float:
    # комиссия считается от оборота (price * qty)
    return price * qty * fee


def simulate(
    prices_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    config: Dict[str, Any],
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Простая симуляция рыночных ордеров:
      - вход по следующей свече по цене close +/- spread_abs + slippage_abs
      - выход по политике (простейшие fixed RR или ATR-based из EXIT_POLICIES
        — задаются в signals_df колонками: tp_price, sl_price или rr,
        atr_mult и т.п. — конкретику формирует ДНК/агрегатор)
      - комиссия fee и спред учитываются
      - плечо масштабирует PnL

    Ожидаемые колонки в signals_df:
      - entry_action_pred: 0/1/2 или "SKIP"/"BUY"/"SELL"
      - policy_choice_pred: индекс/имя политики (опционально)
      - Доп. колонки, которые ДНК может добавить: tp_price, sl_price и др.
    """
    if logger is None:
        logger = get_logger(name="backtesting")

    cfg = SimulationConfig(**config)

    # выравнивание индексов и сортировка
    dfp = prices_df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    dfp = dfp.sort_values("open_time").reset_index(drop=True)
    sig = signals_df.copy()
    sig = sig.sort_values("open_time").reset_index(drop=True)
    df = dfp.merge(
        sig[[c for c in sig.columns if c != "volume"]], on="open_time", how="left"
    )

    # Нормируем entry_action к int: 0=SKIP,1=BUY,2=SELL
    def norm_action(x):
        if isinstance(x, str):
            m = {"SKIP": 0, "BUY": 1, "SELL": 2}
            return m.get(x.upper(), 0)
        try:
            xi = int(x)
            if xi in (0, 1, 2):
                return xi
            return 0
        except Exception:
            return 0

    df["entry_action_pred"] = df["entry_action_pred"].map(norm_action)

    # Список сделок
    trades: List[Dict[str, Any]] = []
    equity = float(cfg.initial_equity)
    equity_curve = []
    in_position = False
    pos_side = 0  # +1 для BUY, -1 для SELL
    pos_entry = 0.0
    pos_qty = 0.0
    pos_policy = None
    entry_time = None

    def price_with_costs(
        base_price: float, side: int, spread_abs: float, slippage_abs: float
    ) -> float:
        # покупка по цене ask = base + spread, продажа по bid = base - spread
        p = base_price + (spread_abs + slippage_abs) * (1 if side > 0 else -1)
        return max(p, 1e-12)

    with timeit_ctx("simulate loop", logger):
        for i in range(len(df) - 1):  # вход по следующей свече
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            t = row["open_time"]

            # закрытие позиции по SL/TP, если внутри бара достигнуты уровни
            if in_position:
                # уровни из signals_df: ожидаем sl_price/tp_price
                # если нет — выходим по close следующего бара
                sl_price = row.get("sl_price", np.nan)
                tp_price = row.get("tp_price", np.nan)

                exit_reason = None
                exit_price = None

                # Для BUY: SL срабатывает если next.low <= sl, TP если next.high >= tp
                if pos_side > 0:
                    if not np.isnan(sl_price) and next_row["low"] <= sl_price:
                        exit_price = price_with_costs(
                            sl_price, -1, cfg.spread_abs, cfg.slippage_abs
                        )
                        exit_reason = "SL"
                    elif not np.isnan(tp_price) and next_row["high"] >= tp_price:
                        exit_price = price_with_costs(
                            tp_price, -1, cfg.spread_abs, cfg.slippage_abs
                        )
                        exit_reason = "TP"
                else:
                    # SELL: SL если next.high >= sl, TP если next.low <= tp
                    if not np.isnan(sl_price) and next_row["high"] >= sl_price:
                        exit_price = price_with_costs(
                            sl_price, +1, cfg.spread_abs, cfg.slippage_abs
                        )
                        exit_reason = "SL"
                    elif not np.isnan(tp_price) and next_row["low"] <= tp_price:
                        exit_price = price_with_costs(
                            tp_price, +1, cfg.spread_abs, cfg.slippage_abs
                        )
                        exit_reason = "TP"

                # если не сработало — закрываем на close следующего бара
                if exit_price is None and i + 1 < len(df):
                    mkt_exit = price_with_costs(
                        next_row["close"], -pos_side, cfg.spread_abs, cfg.slippage_abs
                    )
                    exit_price = mkt_exit
                    exit_reason = "MKT"

                # рассчёт PnL
                gross = (exit_price - pos_entry) * pos_qty * pos_side
                gross *= cfg.leverage
                # комиссии: вход и выход
                fee_in = _apply_fees(pos_entry, pos_qty, cfg.fee)
                fee_out = _apply_fees(exit_price, pos_qty, cfg.fee)
                pnl = gross - fee_in - fee_out
                equity += pnl

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": next_row["open_time"],
                        "side": "BUY" if pos_side > 0 else "SELL",
                        "entry": pos_entry,
                        "exit": exit_price,
                        "qty": pos_qty,
                        "pnl": pnl,
                        "reason": exit_reason,
                        "policy": pos_policy,
                    }
                )

                in_position = False
                pos_side = 0
                pos_entry = 0.0
                pos_qty = 0.0
                pos_policy = None
                entry_time = None

            # открытие новой позиции, если есть сигнал и не в позиции
            if not in_position:
                act = int(row.get("entry_action_pred", 0))
                if act in (1, 2):
                    side = +1 if act == 1 else -1
                    # вход по цене следующей свечи
                    entry_price = price_with_costs(
                        next_row["close"], side, cfg.spread_abs, cfg.slippage_abs
                    )
                    qty = _calc_position_size(entry_price, equity, cfg)
                    if qty > 0:
                        in_position = True
                        pos_side = side
                        pos_entry = entry_price
                        pos_qty = qty
                        pos_policy = row.get("policy_choice_pred", None)
                        entry_time = next_row["open_time"]

            equity_curve.append({"time": row["open_time"], "equity": equity})

    equity_df = pd.DataFrame(equity_curve).astype({"time": "datetime64[ns, UTC]"})
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    metrics = _compute_metrics(trades_df, equity_df)

    return trades_df, equity_df, metrics


def _compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if trades.empty:
        out.update(
            {
                "total_trades": 0,
                "total_pnl": 0.0,
                "winrate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "max_dd": 0.0,
                "sharpe": 0.0,
                "sqn": 0.0,
            }
        )
        return out

    pnl = trades["pnl"].to_numpy()
    total_pnl = float(np.sum(pnl))
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    winrate = float(len(wins) / len(pnl)) if len(pnl) > 0 else 0.0
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    has_wins = len(wins) > 0
    has_losses = len(losses) > 0
    if has_wins and has_losses:
        profit_factor = float(wins.sum() / -losses.sum())
    elif not has_losses and has_wins:
        profit_factor = float(np.inf)
    else:
        profit_factor = 0.0

    # Max Drawdown по equity
    eq = equity["equity"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(eq)
    drawdown = eq - running_max
    max_dd = float(drawdown.min())

    # Sharpe (простая): mean daily pnl / std daily pnl; на основе изменений equity
    rets = np.diff(eq)
    if rets.size > 1 and np.std(rets) > 1e-12:
        sharpe = float(
            np.mean(rets) / np.std(rets) * np.sqrt(252)
        )  # 252 как приближение
    else:
        sharpe = 0.0

    # SQN: mean trade / std trade * sqrt(n)
    if np.std(pnl) > 1e-12:
        sqn = float(np.mean(pnl) / np.std(pnl) * np.sqrt(len(pnl)))
    else:
        sqn = 0.0

    out.update(
        {
            "total_trades": int(len(trades)),
            "total_pnl": total_pnl,
            "winrate": winrate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_dd": max_dd,
            "sharpe": sharpe,
            "sqn": sqn,
        }
    )
    return out


def _plot_equity(equity_df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(10, 4))
    plt.plot(equity_df["time"], equity_df["equity"], label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_drawdown(equity_df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    eq = equity_df["equity"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(eq)
    dd = eq - running_max
    plt.figure(figsize=(10, 3))
    plt.plot(equity_df["time"], dd, color="red", label="Drawdown")
    plt.title("Drawdown")
    plt.xlabel("Time")
    plt.ylabel("DD")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_backtest_report(
    run_id: str,
    out_dir: Path,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    metrics: Dict[str, Any],
    charts: Optional[Dict[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    logger=None,
) -> Dict[str, Path]:
    if logger is None:
        logger = get_logger(name="backtesting")
    ensure_dir(out_dir)

    trades_path = out_dir / "trades.csv"
    equity_path = out_dir / "equity_curve.csv"
    metrics_csv_path = out_dir / "metrics.csv"
    summary_json_path = out_dir / "summary.json"
    config_json_path = out_dir / "config.json"
    charts_dir = out_dir / "charts"
    ensure_dir(charts_dir)

    trades.to_csv(trades_path, index=False)
    equity.to_csv(equity_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    write_json(summary_json_path, {"run_id": run_id, "metrics": metrics})
    if config is not None:
        write_json(config_json_path, config)

    # авто-графики
    eq_png = charts_dir / "equity.png"
    dd_png = charts_dir / "dd.png"
    _plot_equity(equity, eq_png)
    _plot_drawdown(equity, dd_png)

    out = {
        "trades_csv": trades_path,
        "equity_csv": equity_path,
        "metrics_csv": metrics_csv_path,
        "summary_json": summary_json_path,
        "equity_png": eq_png,
        "dd_png": dd_png,
    }
    if charts:
        out.update(charts)
    return out
