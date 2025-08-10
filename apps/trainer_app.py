# apps/trainer_app.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Гарантируем корректные импорты пакета libs даже при запуске не из корня
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from libs.data_io import (
    DataIOError,
    load_csv,
    load_xlsx,
    load_json,
    merge_datasets,
    validate_dataframe,
    maybe_load_cached,
    save_cached,
    build_cache_key_for_step,
)
from libs.dna_contract import load_dna_module, DNAContractError
from libs.training import train_one_task, save_model_bundle
from libs.utils import (
    ensure_dir,
    get_logger,
    detect_env_summary,
    TOOL_VERSION,
    write_json,
)

st.set_page_config(page_title="Trainer App", layout="wide")


def sidebar_globals():
    st.sidebar.header("Глобальные параметры")
    data_dir = Path(st.sidebar.text_input("Каталог данных", "data"))
    models_dir = Path(st.sidebar.text_input("Каталог моделей", "models"))
    reports_dir = Path(st.sidebar.text_input("Каталог отчётов", "reports"))
    strategies_dir = Path(
        st.sidebar.text_input("Каталог стратегий (DNA)", "strategies")
    )
    use_gpu = st.sidebar.checkbox("GPU, если доступно", value=False)
    log_level = st.sidebar.selectbox(
        "Log level", ["INFO", "DEBUG", "WARNING", "ERROR"], index=0
    )

    ensure_dir(data_dir / "raw")
    ensure_dir(data_dir / "prepared")
    ensure_dir(models_dir)
    ensure_dir(reports_dir / "training")
    ensure_dir(strategies_dir)

    return {
        "data_dir": data_dir,
        "models_dir": models_dir,
        "reports_dir": reports_dir,
        "strategies_dir": strategies_dir,
        "use_gpu": use_gpu,
        "log_level": log_level,
    }


def list_strategy_files(strategies_dir: Path) -> List[Path]:
    return sorted(list(strategies_dir.glob("*.py")))


def main():
    g = sidebar_globals()
    logger = get_logger(
        log_path=(g["reports_dir"] / "training" / "ui_logs.txt"),
        name="trainer_ui",
        level=g["log_level"],
    )

    st.title("Trainer — обучатор моделей по ДНК контракту")
    st.caption(
        f"Версия инструмента: {TOOL_VERSION} | Среда: {detect_env_summary().get('platform')}"
    )

    if "state" not in st.session_state:
        st.session_state.state = {}
    state = st.session_state.state

    data_dir: Path = g["data_dir"]
    prepared_dir = data_dir / "prepared"
    raw_dir = data_dir / "raw"

    # Шаг 1 — Данные
    st.markdown("Шаг 1 — Загрузка и валидация данных")
    with st.form("load_data_form", clear_on_submit=False):
        st.write(
            "Выберите файлы из data/raw или data/prepared. Будут объединены по времени."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            csv_files = st.multiselect(
                "CSV файлы", [str(p) for p in sorted(raw_dir.glob("*.csv"))]
            )
        with col2:
            xlsx_files = st.multiselect(
                "XLSX файлы", [str(p) for p in sorted(raw_dir.glob("*.xlsx"))]
            )
        with col3:
            json_files = st.multiselect(
                "JSON файлы", [str(p) for p in sorted(raw_dir.glob("*.json"))]
            )

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

                    # Нормализация open_time на случай неоднородных файлов перед validate
                    if "open_time" in df.columns:
                        s = df["open_time"]
                        try:
                            if pd.api.types.is_integer_dtype(s):
                                if s.dropna().astype("int64").gt(10**12).any():
                                    df["open_time"] = pd.to_datetime(
                                        s, unit="ms", utc=True
                                    )
                                else:
                                    df["open_time"] = pd.to_datetime(
                                        s, unit="s", utc=True
                                    )
                            else:
                                df["open_time"] = pd.to_datetime(
                                    s, utc=True, errors="coerce"
                                )
                        except Exception:
                            pass

                    validate_dataframe(df)
                    state["df_raw"] = df
                    state["source_files"] = [
                        Path(p) for p in (csv_files + xlsx_files + json_files)
                    ]
                    st.success(f"Загружено и валидировано: {len(df)} строк.")
                    st.dataframe(df.head(20))
            except DataIOError as e:
                st.error(f"Ошибка загрузки/валидации: {e}")

    st.markdown("---")

    # Шаг 2 — ДНК
    st.markdown("Шаг 2 — Выбор ДНК стратегии")
    with st.form("dna_form", clear_on_submit=False):
        strategy_files = list_strategy_files(g["strategies_dir"])
        sf_str = [str(p) for p in strategy_files]
        sel = st.selectbox("Файл стратегии", sf_str, index=0 if sf_str else -1)
        btn = st.form_submit_button("Загрузить ДНК")
        if btn and sel:
            try:
                dna_path = Path(sel)
                dna_module = load_dna_module(dna_path)
                state["dna_module"] = dna_module
                state["dna_path"] = dna_path
                st.success(f"Загружена стратегия: {dna_module.STRATEGY_NAME}")
                st.json(dna_module.TASKS)
            except DNAContractError as e:
                st.error(f"Ошибка загрузки ДНК: {e}")

    st.markdown("---")

    # Шаг 3 — meta/задачи/модель
    st.markdown("Шаг 3 — Настройка meta и выбор задачи/модели")
    with st.form("config_form", clear_on_submit=False):
        colA, colB, colC = st.columns(3)
        with colA:
            seed = st.number_input("Seed", value=int(state.get("seed", 42)))
            atr_period = st.number_input(
                "atr_period", value=int(state.get("atr_period", 14))
            )
            rsi_period = st.number_input(
                "rsi_period", value=int(state.get("rsi_period", 14))
            )
        with colB:
            sma_periods = st.text_input("sma_periods (через запятую)", value="10,20,50")
            ema_periods = st.text_input("ema_periods (через запятую)", value="12,26")
            vol_period = st.number_input(
                "vol_period", value=int(state.get("vol_period", 20))
            )
        with colC:
            valid_size = st.slider(
                "Доля валидации", min_value=0.05, max_value=0.5, value=0.2, step=0.05
            )
            model_name = st.selectbox(
                "Алгоритм",
                [
                    "RandomForest",
                    "XGBoost",
                    "LightGBM",
                    "LogisticRegression",
                    "SVC",
                    "CatBoost",
                ],
            )
            use_gpu_flag = st.checkbox("GPU (перекрыть sidebar)", value=False)

        # Гиперпараметры
        st.markdown("Гиперпараметры")
        params: Dict[str, Any] = {}
        if model_name == "RandomForest":
            col1, col2 = st.columns(2)
            with col1:
                params["n_estimators"] = st.number_input(
                    "n_estimators", value=300, step=50
                )
            with col2:
                params["max_depth"] = st.number_input("max_depth", value=6, step=1)
        elif model_name == "XGBoost":
            col1, col2, col3 = st.columns(3)
            with col1:
                params["n_estimators"] = st.number_input(
                    "n_estimators", value=400, step=50
                )
            with col2:
                params["max_depth"] = st.number_input("max_depth", value=6, step=1)
            with col3:
                params["learning_rate"] = st.number_input("learning_rate", value=0.05)
        elif model_name == "LightGBM":
            col1, col2, col3 = st.columns(3)
            with col1:
                params["n_estimators"] = st.number_input(
                    "n_estimators", value=400, step=50
                )
            with col2:
                params["max_depth"] = st.number_input("max_depth", value=-1, step=1)
            with col3:
                params["learning_rate"] = st.number_input("learning_rate", value=0.05)
        elif model_name == "LogisticRegression":
            col1 = st.columns(1)[0]
            with col1:
                params["C"] = st.number_input("C", value=1.0)
                params["max_iter"] = st.number_input("max_iter", value=1000, step=100)
        elif model_name == "SVC":
            col1, col2 = st.columns(2)
            with col1:
                params["C"] = st.number_input("C", value=1.0)
            with col2:
                params["kernel"] = st.selectbox(
                    "kernel", ["rbf", "linear", "poly", "sigmoid"]
                )
        elif model_name == "CatBoost":
            col1, col2, col3 = st.columns(3)
            with col1:
                params["depth"] = st.number_input("depth", value=6, step=1)
            with col2:
                params["iterations"] = st.number_input("iterations", value=400, step=50)
            with col3:
                params["learning_rate"] = st.number_input("learning_rate", value=0.05)

        task_name = None
        if "dna_module" in state:
            tasks = list(state["dna_module"].TASKS.keys())
            if tasks:
                task_name = st.selectbox("Задача (task)", tasks)

        submitted = st.form_submit_button("Сохранить конфиг")
        if submitted:
            try:
                meta = {
                    "seed": int(seed),
                    "atr_period": int(atr_period),
                    "rsi_period": int(rsi_period),
                    "vol_period": int(vol_period),
                    "sma_periods": [
                        int(x.strip()) for x in sma_periods.split(",") if x.strip()
                    ],
                    "ema_periods": [
                        int(x.strip()) for x in ema_periods.split(",") if x.strip()
                    ],
                }
                state["meta"] = meta
                state["model_spec"] = {
                    "algorithm": model_name,
                    "params": params,
                    "use_gpu": bool(use_gpu_flag),
                }
                state["split_cfg"] = {"valid_size": float(valid_size)}
                state["task_name"] = task_name
                st.success("Конфигурация сохранена")
            except Exception as e:
                st.error(f"Ошибка сохранения конфига: {e}")

    st.markdown("---")

    # Шаг 4 — Обучение
    st.markdown("Шаг 4 — Обучение")
    with st.form("train_form", clear_on_submit=False):
        model_name_out = st.text_input(
            "Имя модели (basename без расширения)", value="model_example_entry_action"
        )
        run = st.form_submit_button("Обучить")
        if run:
            if "df_raw" not in state:
                st.error("Нет данных. Выполните шаг 1.")
            elif "dna_module" not in state:
                st.error("Нет ДНК. Выполните шаг 2.")
            elif (
                "meta" not in state
                or "model_spec" not in state
                or "split_cfg" not in state
            ):
                st.error("Нет конфигурации. Выполните шаг 3.")
            else:
                try:
                    dna = state["dna_module"]
                    task_name = state.get("task_name") or list(dna.TASKS.keys())[0]

                    _ = build_cache_key_for_step(
                        source_files=state.get("source_files", []),
                        dna_name=dna.STRATEGY_NAME,
                        meta=state["meta"],
                        step="features_ideas",
                        tool_ver=TOOL_VERSION,
                    )

                    res = train_one_task(
                        dna_module=dna,
                        df_raw=state["df_raw"],
                        meta=state["meta"],
                        task_name=task_name,
                        model_spec=state["model_spec"],
                        split_cfg=state["split_cfg"],
                        logger=logger,
                    )

                    state["train_result"] = res
                    state["model_name_out"] = model_name_out
                    st.success("Обучение завершено.")

                    st.subheader("Метрики валидации")
                    st.json(res.metrics)

                    if "confusion_matrix" in res.artifacts:
                        st.image(
                            str(res.artifacts["confusion_matrix"]),
                            caption="Confusion Matrix",
                        )
                    if "feature_importances" in res.artifacts:
                        st.write("Feature importances")
                        st.dataframe(
                            pd.read_csv(res.artifacts["feature_importances"]).head(30)
                        )

                except Exception as e:
                    st.error(f"Ошибка обучения: {e}")

    st.markdown("---")

    # Шаг 5 — Сохранение
    st.markdown("Шаг 5 — Сохранение модели и отчёта")
    with st.form("save_form", clear_on_submit=False):
        save = st.form_submit_button("Сохранить модель и метаданные")
        if save:
            if "train_result" not in state or "dna_path" not in state:
                st.error("Нет результатов обучения.")
            else:
                try:
                    res = state["train_result"]
                    out_prefix = g["models_dir"] / state.get(
                        "model_name_out", "model_example"
                    )
                    meta_out = {
                        "strategy_name": state["dna_module"].STRATEGY_NAME,
                        **res.train_meta,
                    }
                    saved_paths = save_model_bundle(
                        model=res.model,
                        meta_out=meta_out,
                        dna_src_path=state["dna_path"],
                        out_prefix=out_prefix,
                        feats=res.feats,
                        tasks_schema=res.tasks_schema,
                        logger=logger,
                    )
                    report_dir = g["reports_dir"] / "training" / out_prefix.name
                    ensure_dir(report_dir)
                    write_json(report_dir / "metrics.json", res.metrics)
                    if "feature_importances" in res.artifacts:
                        fi_dst = report_dir / "feature_importances.csv"
                        Path(res.artifacts["feature_importances"]).replace(fi_dst)
                    if "confusion_matrix" in res.artifacts:
                        cm_dst = report_dir / "confusion_matrix.png"
                        Path(res.artifacts["confusion_matrix"]).replace(cm_dst)

                    st.success(
                        f"Сохранено: {saved_paths['model'].name}, {saved_paths['meta'].name}, {saved_paths['strategy_py'].name}"
                    )
                except Exception as e:
                    st.error(f"Ошибка сохранения: {e}")


if __name__ == "__main__":
    main()
