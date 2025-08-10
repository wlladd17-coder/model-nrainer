# libs/utils.py
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


TOOL_VERSION = "0.1.0"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _default_log_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(log_path: Optional[Path] = None, name: str = "app", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # avoid duplicate handlers on repeated calls
    if not logger.handlers:
        formatter = _default_log_formatter()

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_path is not None:
            ensure_dir(log_path.parent)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger


@contextmanager
def timeit_ctx(label: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        msg = f"{label} done in {dt:.3f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)


def json_dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def df_cache_key(
    source_files: List[Path],
    dna_name: str,
    meta: Dict[str, Any],
    step: str,
    tool_ver: str = TOOL_VERSION,
) -> str:
    # include filenames, sizes, mtimes for stability
    parts: List[str] = [f"ver={tool_ver}", f"dna={dna_name}", f"step={step}"]
    for p in sorted(source_files):
        try:
            stat = p.stat()
            parts.append(f"{p.name}:{stat.st_size}:{int(stat.st_mtime)}")
        except FileNotFoundError:
            parts.append(f"{p.name}:missing")
    parts.append(json_dumps_compact(meta))
    raw = "|".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore
            torch.cuda.manual_seed_all(seed)  # type: ignore
    except Exception:
        pass


def xgb_device_params(use_gpu: bool) -> Dict[str, Any]:
    """
    Returns recommended XGBoost device params for modern versions.
    If GPU requested but not available, falls back to CPU hist.
    """
    if use_gpu:
        try:
            import xgboost as xgb  # noqa: F401
            # We cannot reliably check CUDA runtime here; rely on user having cuda build.
            return {"device": "cuda", "tree_method": "hist"}
        except Exception:
            return {"tree_method": "hist"}
    else:
        return {"tree_method": "hist"}


def detect_env_summary() -> Dict[str, Any]:
    return {
        "tool_version": TOOL_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "pid": os.getpid(),
    }


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def sha1_of_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_exception(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"
