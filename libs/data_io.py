# libs/data_io.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import ensure_dir, df_cache_key, write_json

REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]


class DataIOError(Exception):
    pass


def _normalize_open_time_series(s: pd.Series) -> pd.Series:
    """
    Преобразует open_time к pandas datetime64[ns, UTC].
    Принимает:
      - int64 миллисекунды с эпохи
      - секунды с эпохи (эвристика: если значения < 10**12)
      - ISO-строки
      - уже datetime
    """
    if pd.api.types.is_integer_dtype(s):
        if s.dropna().astype("int64").gt(10**12).any():
            dt = pd.to_datetime(s, unit="ms", utc=True)
        else:
            dt = pd.to_datetime(s, unit="s", utc=True)
    else:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
    return dt


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_csv(paths: List[Path]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise DataIOError(f"CSV file not found: {p}")
        df = pd.read_csv(p)
        if "open_time" in df.columns:
            df["open_time"] = _normalize_open_time_series(df["open_time"])
        dfs.append(df)
    return dfs


def load_xlsx(paths: List[Path]) -> List[pd.DataFrame]:
    dfs: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise DataIOError(f"XLSX file not found: {p}")
        df = pd.read_excel(p)
        if "open_time" in df.columns:
            df["open_time"] = _normalize_open_time_series(df["open_time"])
        dfs.append(df)
    return dfs


def load_json(paths: List[Path]) -> List[pd.DataFrame]:
    """
    JSON: records/list, open_time может быть в мс (int64) или ISO-строкой.
    """
    dfs: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            raise DataIOError(f"JSON file not found: {p}")
        try:
            df = pd.read_json(p, lines=False)
            if "open_time" not in df.columns:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
        except ValueError:
            df = pd.read_json(p, lines=True)
        if "open_time" in df.columns:
            df["open_time"] = _normalize_open_time_series(df["open_time"])
        dfs.append(df)
    return dfs


def merge_datasets(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    if not dfs:
        raise DataIOError("No DataFrames to merge")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    if "open_time" not in df.columns:
        raise DataIOError("Merged DataFrame missing 'open_time'")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def validate_dataframe(df: pd.DataFrame) -> None:
    # 0) наличие столбцов
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataIOError(f"Input DataFrame missing columns: {missing}")

    # 1) приведение open_time к datetime при необходимости (самовосстановление)
    # Учитываем случаи object/str/float и nullable dtype, защищаемся от падений issubdtype
    if not (np.issubdtype(df["open_time"].dtype, np.datetime64) or pd.api.types.is_datetime64_any_dtype(df["open_time"])):
        s = df["open_time"]
        try:
            if pd.api.types.is_integer_dtype(s):
                if s.dropna().astype("int64").gt(10**12).any():
                    df["open_time"] = pd.to_datetime(s, unit="ms", utc=True)
                else:
                    df["open_time"] = pd.to_datetime(s, unit="s", utc=True)
            else:
                df["open_time"] = pd.to_datetime(s, utc=True, errors="coerce")
        except Exception as e:
            raise DataIOError(f"open_time cannot be parsed to datetime: {e}")

    # 2) привести к UTC
    if df["open_time"].dt.tz is None:
        df["open_time"] = df["open_time"].dt.tz_localize("UTC")
    else:
        df["open_time"] = df["open_time"].dt.tz_convert("UTC")

    # 3) числовые колонки
    num_cols = ["open", "high", "low", "close", "volume"]
    for c in num_cols:
        if not (pd.api.types.is_numeric_dtype(df[c]) or str(df[c].dtype) == "Int64"):
            try:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            except Exception:
                raise DataIOError(f"Column {c} must be numeric")

    # 4) удалить строки с NaN в обязательных колонках
    if df[REQUIRED_COLUMNS].isna().any().any():
        df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    # 5) сортировка и уникальность
    df.sort_values("open_time", inplace=True)
    df.drop_duplicates(subset=["open_time"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6) финальная проверка
    if df["open_time"].duplicated().any():
        raise DataIOError("Duplicate open_time values found after cleanup")
    if not df["open_time"].is_monotonic_increasing:
        raise DataIOError("open_time must be sorted ascending after cleanup")


def persist_prepared(df: pd.DataFrame, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    elif out_path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() in (".json", ".ndjson"):
        df.to_json(out_path, orient="records", force_ascii=False)
    else:
        df.to_csv(out_path, index=False)


def maybe_load_cached(hash_key: str, prepared_dir: Path) -> Optional[pd.DataFrame]:
    ensure_dir(prepared_dir)
    parquet_path = prepared_dir / f"{hash_key}.parquet"
    csv_path = prepared_dir / f"{hash_key}.csv"

    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            return df
        except Exception:
            pass
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if "open_time" in df.columns:
                df["open_time"] = _normalize_open_time_series(df["open_time"])
            return df
        except Exception:
            pass
    return None


def save_cached(df: pd.DataFrame, hash_key: str, prepared_dir: Path, meta: Dict[str, Any] | None = None) -> Path:
    ensure_dir(prepared_dir)
    out_path = prepared_dir / f"{hash_key}.parquet"
    df.to_parquet(out_path, index=False)
    if meta is not None:
        meta_path = prepared_dir / f"{hash_key}.meta.json"
        write_json(meta_path, meta)
    return out_path


def build_cache_key_for_step(
    source_files: List[Path],
    dna_name: str,
    meta: Dict[str, Any],
    step: str,
    tool_ver: str = "0.1.0",
) -> str:
    return df_cache_key(source_files, dna_name, meta, step, tool_ver)
