# libs/training.py
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC, SVR
from datetime import datetime, timezone

from .dna_contract import snapshot_dna_file
from .utils import (
    TOOL_VERSION,
    ensure_dir,
    get_logger,
    timeit_ctx,
    xgb_device_params,
    write_json,
    detect_env_summary,
    set_global_seed,
)


# Model factory

def build_model(alg: str, task_type: str, params: Dict[str, Any], use_gpu: bool = False):
    alg = alg.lower()
    if alg == "randomforest":
        if task_type == "classification":
            return RandomForestClassifier(**params)
        else:
            return RandomForestRegressor(**params)
    if alg == "logisticregression":
        if task_type != "classification":
            raise ValueError("LogisticRegression supports only classification")
        return LogisticRegression(max_iter=int(params.pop("max_iter", 1000)), **params)
    if alg == "svc":
        if task_type == "classification":
            return SVC(probability=True, **params)
        else:
            return SVR(**params)
    if alg == "xgboost":
        try:
            import xgboost as xgb  # type: ignore
        except Exception as e:
            raise RuntimeError("xgboost is not installed") from e
        device_params = xgb_device_params(use_gpu)
        params = {**device_params, **params}
        if task_type == "classification":
            # multi or binary will be inferred by objective if provided, else auto
            if "objective" not in params:
                params["objective"] = "multi:softprob"
            model = xgb.XGBClassifier(**params)
        else:
            if "objective" not in params:
                params["objective"] = "reg:squarederror"
            model = xgb.XGBRegressor(**params)
        return model
    if alg == "lightgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except Exception as e:
            raise RuntimeError("lightgbm is not installed") from e
        if task_type == "classification":
            return lgb.LGBMClassifier(**params)
        else:
            return lgb.LGBMRegressor(**params)
    if alg == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
        except Exception as e:
            raise RuntimeError("catboost is not installed") from e
        if task_type == "classification":
            return CatBoostClassifier(verbose=False, **params)
        else:
            return CatBoostRegressor(verbose=False, **params)
    raise ValueError(f"Unknown algorithm: {alg}")


def _infer_task_type(tasks_schema: Dict[str, Any], task_name: str) -> str:
    cfg = tasks_schema.get(task_name)
    if cfg is None:
        # tolerate schema as {"entry_action":"classification", ...}
        if isinstance(tasks_schema.get(task_name), str):
            return tasks_schema[task_name]
        raise ValueError(f"Task '{task_name}' not found in TASKS schema")
    if isinstance(cfg, str):
        return cfg
    # common keys
    return cfg.get("type") or cfg.get("task") or cfg.get("kind") or ("classification" if "classes" in cfg else "regression")


def _is_binary(y: pd.Series) -> bool:
    try:
        return y.dropna().nunique() == 2
    except Exception:
        return False


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _save_feature_importances(model, feature_names: List[str], out_path: Path) -> bool:
    ensure_dir(out_path.parent)
    try:
        importances = None
        if hasattr(model, "feature_importances_"):
            importances = np.array(model.feature_importances_, dtype=float)
        elif hasattr(model, "coef_"):
            coef = model.coef_
            if isinstance(coef, np.ndarray):
                importances = np.mean(np.abs(coef), axis=0)
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            score = booster.get_score(importance_type="gain")
            # map by feature names as f0,f1...
            importances = np.zeros(len(feature_names), dtype=float)
            for i, _ in enumerate(feature_names):
                importances[i] = score.get(f"f{i}", 0.0)
        if importances is None:
            return False
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
        df_imp.sort_values("importance", ascending=False).to_csv(out_path, index=False)
        return True
    except Exception:
        return False


@dataclass
class TrainResult:
    model: Any
    metrics: Dict[str, Any]
    artifacts: Dict[str, Path]
    feats: List[str]
    tasks_schema: Dict[str, Any]
    train_meta: Dict[str, Any]
    timings: Dict[str, float]


def time_split_indices(n: int, valid_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    n_valid = int(max(1, round(n * valid_size)))
    n_train = max(1, n - n_valid)
    train_idx = np.arange(0, n_train, dtype=int)
    valid_idx = np.arange(n_train, n, dtype=int)
    return train_idx, valid_idx


def train_one_task(
    dna_module,
    df_raw: pd.DataFrame,
    meta: Dict[str, Any],
    task_name: str,
    model_spec: Dict[str, Any],
    split_cfg: Dict[str, Any],
    logger=None,
) -> TrainResult:
    """
    Pipeline:
      calculate_features -> generate_ideas -> build_labels -> feature_columns -> inference_inputs
      time-based split -> fit -> evaluate -> artifacts
    """
    if logger is None:
        logger = get_logger(name="training")

    set_global_seed(int(meta.get("seed", 42)))
    timings: Dict[str, float] = {}
    artifacts: Dict[str, Path] = {}
    reports_dir = Path("reports") / "training"
    ensure_dir(reports_dir)

    with timeit_ctx("calculate_features", logger):
        df_feat = dna_module.calculate_features(df_raw.copy(), meta)
    with timeit_ctx("generate_ideas", logger):
        df_ideas = dna_module.generate_ideas(df_feat.copy(), meta)
    with timeit_ctx("build_labels", logger):
        labels_dict = dna_module.build_labels(df_ideas.copy(), meta)

    feats = dna_module.feature_columns(df_feat)
    X_full = dna_module.inference_inputs(df_feat, feats)
    if task_name not in labels_dict:
        raise ValueError(f"Task '{task_name}' not found in labels; available: {list(labels_dict)}")
    y_full = labels_dict[task_name]

    # Align indices
    X, y = X_full.align(y_full, join="inner", axis=0)
    # Drop rows with NaN in X or y
    mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[mask]
    y = y.loc[mask]

    if len(X) < 20:
        raise ValueError("Not enough samples after cleaning; need at least 20 rows")

    # Split
    valid_size = float(split_cfg.get("valid_size", 0.2))
    train_idx, valid_idx = time_split_indices(len(X), valid_size=valid_size)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # Model
    alg = model_spec.get("algorithm", "RandomForest")
    params = model_spec.get("params", {}) or {}
    use_gpu = bool(model_spec.get("use_gpu", False))

    task_type = _infer_task_type(dna_module.TASKS, task_name)
    model = build_model(alg, task_type, params, use_gpu=use_gpu)

    with timeit_ctx(f"fit {alg}", logger):
        model.fit(X_train, y_train)

    # Evaluate
    metrics_out: Dict[str, Any] = {}
    if task_type == "classification":
        y_pred = model.predict(X_valid)
        metrics_out["accuracy"] = float(metrics.accuracy_score(y_valid, y_pred))
        metrics_out["f1_macro"] = float(metrics.f1_score(y_valid, y_pred, average="macro"))
        if _is_binary(y_valid):
            # ROC-AUC with probability for positive class
            try:
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_valid)
                    if proba.shape[1] == 2:
                        metrics_out["roc_auc"] = float(metrics.roc_auc_score(y_valid, proba[:, 1]))
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_valid)
                    metrics_out["roc_auc"] = float(metrics.roc_auc_score(y_valid, scores))
            except Exception:
                pass
        # Confusion matrix
        labels_sorted = sorted(pd.Series(y_valid.unique()).tolist())
        cm_path = reports_dir / "tmp_confusion_matrix.png"
        _plot_confusion_matrix(y_valid.to_numpy(), y_pred, labels=[str(x) for x in labels_sorted], out_path=cm_path)
        artifacts["confusion_matrix"] = cm_path
    else:
        y_pred = model.predict(X_valid)
        metrics_out["mae"] = float(metrics.mean_absolute_error(y_valid, y_pred))
        metrics_out["mse"] = float(metrics.mean_squared_error(y_valid, y_pred))
        metrics_out["r2"] = float(metrics.r2_score(y_valid, y_pred))

    # Feature importances
    fi_path = reports_dir / "tmp_feature_importances.csv"
    if _save_feature_importances(model, feats, fi_path):
        artifacts["feature_importances"] = fi_path

    train_meta = {
        "meta": meta,
        "task": task_name,
        "task_type": task_type,
        "algorithm": alg,
        "params": params,
        "valid_size": valid_size,
        "n_samples": len(X),
        "n_features": len(feats),
        "env": detect_env_summary(),
        "tool_version": TOOL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return TrainResult(
        model=model,
        metrics=metrics_out,
        artifacts=artifacts,
        feats=feats,
        tasks_schema=dna_module.TASKS,
        train_meta=train_meta,
        timings=timings,
    )


def save_model_bundle(
    model: Any,
    meta_out: Dict[str, Any],
    dna_src_path: Path,
    out_prefix: Path,
    feats: List[str],
    tasks_schema: Dict[str, Any],
    logger=None,
) -> Dict[str, Path]:
    """
    Saves:
      - models/{name}.joblib
      - models/{name}.meta.json
      - models/{name}.strategy.py (copy of DNA)
      - reports/training/{name}/... (metrics and artifacts copied/renamed by caller if needed)
    out_prefix: models/model_name (without extension)
    """
    if logger is None:
        logger = get_logger(name="training")

    ensure_dir(out_prefix.parent)
    model_path = out_prefix.with_suffix(".joblib")
    meta_path = out_prefix.with_suffix(".meta.json")
    dna_copy_path = out_prefix.with_suffix(".strategy.py")

    joblib.dump(model, model_path)

    meta_payload = {
        "strategy_name": meta_out.get("strategy_name") or "unknown",
        "strategy_py_filename": str(dna_copy_path.name),
        "task": meta_out.get("task"),
        "feature_columns": feats,
        "tasks_schema": tasks_schema,
        "train_meta": meta_out,
    }
    write_json(meta_path, meta_payload)

    snapshot_dna_file(dna_src_path, dna_copy_path)

    return {
        "model": model_path,
        "meta": meta_path,
        "strategy_py": dna_copy_path,
    }
