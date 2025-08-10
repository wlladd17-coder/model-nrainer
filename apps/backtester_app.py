# apps/backtester_app.py
from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from libs.data_io import (
    DataIOError,
    load_csv,
    load_xlsx,
    load_json,
    merge_datasets,
    validate_dataframe,
)
from libs.dna_contract import load_dna_module, DNAContractError
from libs.backtesting import run_inference, simulate, save_backtest_report
from libs.utils import ensure_dir, get_logger, detect_env_summary, TOOL_VERSION, write_json


st.set_page_config(page_title="Backtester App", layout="wide")


def sidebar_globals():
    st.sidebar.header("Глобальные параметры")
    data_dir = Path(st.sidebar.text_input("Каталог данных", "data"))
    models_dir = Path(st.sidebar.text_input("Каталог моделей", "models"))
    reports_dir = Path(st.sidebar.text_input("Каталог отчётов", "reports"))
    strategies_dir = Path(st.sidebar.text_input("Каталог стратегий (DNA)", "strategies"))
    log_level = st.sidebar.selectbox("Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0)

    ensure_dir(data_dir / "raw")
    ensure_dir(data_dir / "prepared")
    ensure_dir(models_dir)
    ensure_dir(reports_dir / "backtests")
    ensure_dir(strategies_dir)

    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "strategies_dir": strategies_dir,
        "log_level": log_level,
    }


def list_strategy_files(strategies_dir: Path) -> List[Path]:
    return sorted(list(strategies_dir.glob("*.py")))


def list_model_files(models_dir: Path) -> List[Path]:
    return sorted(list(models_dir.glob("*.joblib")))


def load_model_bundle(model_path: Path) -> Dict[str, Any]:
    model = joblib.load(model_path)
    meta_path = model_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден метафайл модели: {meta_path.name}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # Попробуем найти копию ДНК рядом
    strategy_copy_path = model_path.with_suffix(".strategy.py")
    return {"model": model, "meta": meta, "strategy_copy_path": strategy_copy_path if strategy_copy_path.exists() else None}


def main():
    g = sidebar_globals()
    logger = get_logger(log_path=(g["reports_dir"] / "backtests" / "ui_logs.txt"), name="backtester_ui", level=g["log_level"])

    st.title("Backtester — тестирование моделей/оркестров")
    st.caption(f"Версия инструмента: {TOOL_VERSION} | Среда: {detect_env_summary().get('platform')}")

    if "state_bt" not in st.session_state:
        st.session_state.state_bt = {}
    S = st.session_state.state_bt

    data_dir: Path = g["data_dir"]
    raw_dir = data_dir / "raw"

    # Шаг 1 — Данные
    st.markdown("Шаг 1 — Загрузка и валидация данных")
    with st.form("bt_load_data_form", clear_on_submit=False):
        st.write("Выберите файлы из data/raw. Поддерживаются CSV/XLSX/JSON.")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_files = st.multiselect("CSV файлы", [str(p) for p in sorted(raw_dir.glob("*.csv"))])
        with col2:
            xlsx_files = st.multiselect("XLSX файлы", [str(p) for p in sorted(raw_dir.glob("*.xlsx"))])
        with col3:
            json_files = st.multiselect("JSON файлы", [str(p) for p in sorted(raw_dir.glob("*.json"))])
        submitted = st.form_submit_button("Загрузить данные")
        if submitted:
            try:
                dfs: List[pd.DataFrame] = []
                if csv_files:
                    dfs += load_csv([Path(p) for p in csv_files])
                if xlsx_files:
                    dfs += load_xlsx([Path(p) for p in xlsx_files])
                if json_files:
                    dfs += load_json([Path(p) for p in json_files])
                if not dfs:
                    st.error("Не выбрано ни одного файла.")
                else:
                    df = merge_datasets(dfs)
                    validate_dataframe(df)
                    S["df_raw"] = df
                    st.success(f"Загружено и валидировано: {len(df)} строк.")
                    st.dataframe(df.head(20))
            except DataIOError as e:
                st.error(f"Ошибка загрузки/валидации: {e}")

    st.markdown("---")
    # Шаг 2 — Модель/оркестр
    st.markdown("Шаг 2 — Загрузка модели или конфигурация оркестра")

    with st.form("bt_model_form", clear_on_submit=False):
        model_files = list_model_files(g["models_dir"])
        mf_str = [str(p) for p in model_files]
        sel_model = st.selectbox("Файл модели (.joblib)", mf_str, index=0 if mf_str else -1)
        st.write("При наличии рядом .meta.json и .strategy.py — они будут использованы автоматически.")
        use_custom_dna = st.checkbox("Указать ДНК вручную (если копии рядом с моделью нет)")
        custom_dna_file = None
        if use_custom_dna:
            dna_list = list_strategy_files(g["strategies_dir"])
            dna_str = [str(p) for p in dna_list]
            custom_dna_file = st.selectbox("Файл ДНК", dna_str, index=0 if dna_str else -1)

        # Оркестр: можно добавить доп. модели на другие задачи
        st.markdown("Опционально: добавить модели для других задач (оркестр)")
        add_policy_model = st.checkbox("Добавить модель для policy_choice")
        policy_model_path = None
        if add_policy_model:
            policy_model_path = st.selectbox("Файл модели (policy_choice)", mf_str, index=0 if mf_str else -1)

        add_quality_model = st.checkbox("Добавить модель для level_quality")
        quality_model_path = None
        if add_quality_model:
            quality_model_path = st.selectbox("Файл модели (level_quality)", mf_str, index=0 if mf_str else -1)

        vote = st.selectbox("Голосование (для задач с несколькими моделями)", ["hard", "soft", "weighted"], index=0)
        submitted = st.form_submit_button("Загрузить модель/оркестр")
        if submitted:
            try:
                if not sel_model:
                    st.error("Не выбран файл модели.")
                else:
                    main_bundle = load_model_bundle(Path(sel_model))
                    entry_model = main_bundle["model"]
                    entry_meta = main_bundle["meta"]

                    # ДНК модуль: приоритет у копии рядом с моделью
                    dna_module = None
                    if main_bundle["strategy_copy_path"]:
                        try:
                            dna_module = load_dna_module(main_bundle["strategy_copy_path"])
                        except DNAContractError as e:
                            st.warning(f"Ошибка загрузки копии ДНК возле модели: {e}")
                    if dna_module is None:
                        if use_custom_dna and custom_dna_file:
                            dna_module = load_dna_module(Path(custom_dna_file))
                        else:
                            st.error("Не удалось загрузить ДНК. Укажите вручную.")
                            dna_module = None

                    if dna_module is not None:
                        S["dna_module"] = dna_module
                        S["entry_bundle"] = main_bundle
                        S["ensemble"] = {"entry_action": {"model": entry_model}}
                        S["vote"] = vote
                        st.success(f"Загружена модель для entry_action: {Path(sel_model).name}")
                        st.json(entry_meta.get("train_meta", {}))

                        # Дополнительные модели
                        if add_policy_model and policy_model_path:
                            pol_bundle = load_model_bundle(Path(policy_model_path))
                            S["ensemble"]["policy_choice"] = {"model": pol_bundle["model"]}
                            st.info(f"Добавлена модель policy_choice: {Path(policy_model_path).name}")
                        if add_quality_model and quality_model_path:
                            qual_bundle = load_model_bundle(Path(quality_model_path))
                            S["ensemble"]["level_quality"] = {"model": qual_bundle["model"]}
                            st.info(f"Добавлена модель level_quality: {Path(quality_model_path).name}")

                        # feature_columns — из мета основной модели
                        feats = entry_meta.get("feature_columns")
                        if not feats:
                            st.error("В метаданных модели отсутствует feature_columns")
                        else:
                            S["feats"] = feats
                            st.write(f"Число признаков: {len(feats)}")

                        # meta, использованная при обучении
                        train_meta = entry_meta.get("train_meta", {}).get("meta", {})
                        S["train_meta"] = train_meta
                    else:
                        st.error("ДНК не загружена.")
            except Exception as e:
                st.error(f"Ошибка загрузки модели/оркестра: {e}")

    st.markdown("---")
    # Шаг 3 — Сбор фич/идей и инференс
    st.markdown("Шаг 3 — Сбор признаков/идей и инференс")
    with st.form("bt_inference_form", clear_on_submit=False):
        meta_override = st.checkbox("Переопределить meta из тренировки", value=False)
        meta_json = st.text_area("meta (JSON)", value=json.dumps(S.get("train_meta", {}), ensure_ascii=False, indent=2))
        run_inf = st.form_submit_button("Собрать фичи/идеи и выполнить инференс")
        if run_inf:
            if "df_raw" not in S:
                st.error("Нет данных. Выполните шаг 1.")
            elif "dna_module" not in S or "feats" not in S or "ensemble" not in S:
                st.error("Нет модели/ДНК. Выполните шаг 2.")
            else:
                try:
                    meta = S.get("train_meta", {})
                    if meta_override:
                        try:
                            meta = json.loads(meta_json)
                        except json.JSONDecodeError as e:
                            st.error(f"Ошибка парсинга meta JSON: {e}")
                            meta = S.get("train_meta", {})
                    ens = S["ensemble"].copy()
                    # Нормализуем структуру для run_inference
                    model_or_ensemble = {
                        "entry_action": ens.get("entry_action"),
                        "policy_choice": ens.get("policy_choice"),
                        "level_quality": ens.get("level_quality"),
                        "vote": S.get("vote", "hard"),
                    }
                    signals = run_inference(
                        dna_module=S["dna_module"],
                        df_raw=S["df_raw"],
                        meta=meta,
                        feats=S["feats"],
                        model_or_ensemble=model_or_ensemble,
                        logger=logger,
                    )
                    S["signals"] = signals
                    st.success(f"Инференс выполнен. Размер сигналов: {len(signals)}")
                    st.dataframe(signals[["open_time"] + [c for c in signals.columns if c.endswith("_pred")]].tail(20))
                except Exception as e:
                    st.error(f"Ошибка инференса: {e}")

    st.markdown("---")
    # Шаг 4 — Симуляция (бэктест)
    st.markdown("Шаг 4 — Симуляция (бэктест)")
    with st.form("bt_simulate_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            fee = st.number_input("Комиссия (доля)", value=0.0004, step=0.0001, format="%.6f")
            spread_abs = st.number_input("Спред (абс.)", value=0.0, step=0.01)
        with col2:
            leverage = st.number_input("Плечо", value=1.0, step=0.5)
            slippage_abs = st.number_input("Проскальзывание (абс.)", value=0.0, step=0.01)
        with col3:
            risk_mode = st.selectbox("Риск-режим", ["fixed_cash", "pct_equity"], index=0)
            risk_value = st.number_input("Риск (валюта или доля)", value=100.0, step=10.0)
            initial_equity = st.number_input("Стартовый капитал", value=10000.0, step=100.0)

        run_bt = st.form_submit_button("Симулировать")
        if run_bt:
            if "signals" not in S or "df_raw" not in S:
                st.error("Нет сигналов или данных. Выполните предыдущие шаги.")
            else:
                try:
                    cfg = {
                        "fee": float(fee),
                        "spread_abs": float(spread_abs),
                        "leverage": float(leverage),
                        "risk_mode": risk_mode,
                        "risk_value": float(risk_value),
                        "initial_equity": float(initial_equity),
                        "slippage_abs": float(slippage_abs),
                    }
                    trades, equity, metrics = simulate(
                        prices_df=S["df_raw"],
                        signals_df=S["signals"],
                        config=cfg,
                        logger=logger,
                    )
                    S["bt_trades"] = trades
                    S["bt_equity"] = equity
                    S["bt_metrics"] = metrics

                    st.success("Симуляция завершена.")
                    st.subheader("Метрики")
                    st.json(metrics)
                    st.subheader("Последние сделки")
                    st.dataframe(trades.tail(20))
                except Exception as e:
                    st.error(f"Ошибка симуляции: {e}")

    st.markdown("---")
    # Шаг 5 — Сохранение отчёта
    st.markdown("Шаг 5 — Сохранение отчёта")
    with st.form("bt_save_form", clear_on_submit=False):
        run_id_default = datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
        run_id = st.text_input("ID прогона (папка отчёта)", value=run_id_default)
        save = st.form_submit_button("Сохранить отчёт")
        if save:
            if "bt_trades" not in S or "bt_equity" not in S or "bt_metrics" not in S:
                st.error("Нет результатов бэктеста.")
            else:
                try:
                    out_dir = g["reports_dir"] / "backtests" / run_id
                    saved = save_backtest_report(
                        run_id=run_id,
                        out_dir=out_dir,
                        trades=S["bt_trades"],
                        equity=S["bt_equity"],
                        metrics=S["bt_metrics"],
                        charts=None,
                        config={
                            "tool_version": TOOL_VERSION,
                            "env": detect_env_summary(),
                        },
                        logger=logger,
                    )
                    st.success(f"Отчёт сохранён в {out_dir}")
                    st.write({k: str(v) for k, v in saved.items()})
                except Exception as e:
                    st.error(f"Ошибка сохранения отчёта: {e}")


if __name__ == "__main__":
    main()
