# libs/dna_contract.py
from __future__ import annotations

import importlib.util
import inspect
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import ensure_dir, format_exception


REQUIRED_EXPORTS = [
    "STRATEGY_NAME",
    "EXIT_POLICIES",
    "TASKS",
    "calculate_features",
    "generate_ideas",
    "build_labels",
    "feature_columns",
    "inference_inputs",
]


class DNAContractError(Exception):
    pass


def load_dna_module(path: Path, module_name: Optional[str] = None) -> ModuleType:
    """
    Safely load a Python module from file path without polluting sys.modules namespace.
    """
    if not path.exists():
        raise DNAContractError(f"DNA file not found: {path}")

    name = module_name or f"dna_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise DNAContractError(f"Failed to create spec for DNA: {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception as e:
        raise DNAContractError(
            f"Error executing DNA module {path.name}: {format_exception(e)}"
        ) from e

    validate_dna(module)
    return module


def validate_dna(module: ModuleType) -> None:
    """
    Validate mandatory exports and function signatures.
    """
    missing = [x for x in REQUIRED_EXPORTS if not hasattr(module, x)]
    if missing:
        raise DNAContractError(f"DNA missing required exports: {missing}")

    # Basic types
    if not isinstance(getattr(module, "STRATEGY_NAME"), str):
        raise DNAContractError("STRATEGY_NAME must be str")

    exit_policies = getattr(module, "EXIT_POLICIES")
    if not isinstance(exit_policies, list):
        raise DNAContractError("EXIT_POLICIES must be a list[dict]")
    for i, p in enumerate(exit_policies):
        if not isinstance(p, dict) or "name" not in p:
            raise DNAContractError(
                f"EXIT_POLICIES[{i}] must be dict with at least 'name' field"
            )

    tasks = getattr(module, "TASKS")
    if not isinstance(tasks, dict):
        raise DNAContractError("TASKS must be a dict")
    for task, cfg in tasks.items():
        if not isinstance(cfg, dict):
            raise DNAContractError(f"TASKS['{task}'] must be dict")
        if "type" in cfg:
            t = cfg["type"]
        else:
            # backward-compatible with prompt: maybe value is a string type
            t = cfg.get("task", cfg.get("kind", cfg.get("type", None)))
        if t is None:
            # or accept: {"classification": {"classes":[...]}}
            if "classification" in cfg or "regression" in cfg:
                pass
            else:
                # also accept TASKS like: {"entry_action":"classification", ...}
                if isinstance(cfg, str) and cfg in ("classification", "regression"):
                    pass
                else:
                    raise DNAContractError(
                        f"TASKS['{task}'] must specify type/kind or be "
                        "'classification'/'regression'"
                    )
        # classes check for classification if provided
        classes = cfg.get("classes") if isinstance(cfg, dict) else None
        if isinstance(cfg, dict):
            type_hint = cfg.get(
                "type",
                cfg.get(
                    "task",
                    cfg.get(
                        "kind",
                        cfg.get(
                            "classification" if "classification" in cfg else None, None
                        ),
                    ),
                ),
            )
            is_classification = type_hint in ("classification",)
        else:
            is_classification = isinstance(cfg, str) and cfg == "classification"

        if is_classification:
            # classes may be optional for some problems, but recommended
            if classes is not None and not isinstance(classes, list):
                raise DNAContractError(
                    f"TASKS['{task}'].classes must be list if provided"
                )

    # Signature checks (best-effort)
    _require_callable(module, "calculate_features", ["df", "meta"])
    _require_callable(module, "generate_ideas", ["df_with_features", "meta"])
    _require_callable(module, "build_labels", ["df_with_ideas", "meta"])
    _require_callable(module, "feature_columns", ["df_with_features"])
    _require_callable(module, "inference_inputs", ["df_with_features", "feats"])


def _require_callable(module: ModuleType, name: str, arg_names: List[str]) -> None:
    fn = getattr(module, name)
    if not callable(fn):
        raise DNAContractError(f"{name} must be callable")
    try:
        sig = inspect.signature(fn)
    except Exception:
        return
    params = list(sig.parameters.keys())
    # allow extra args, but ensure first N match expected order
    if len(params) < len(arg_names):
        raise DNAContractError(
            f"{name} must accept at least {len(arg_names)} args: {arg_names}"
        )
    for i, an in enumerate(arg_names):
        if params[i] != an:
            # relax strict naming but keep count
            break


def snapshot_dna_file(src_path: Path, dst_path: Path) -> None:
    """
    Copy DNA source file to destination (e.g., models/{name}.strategy.py).
    """
    if not src_path.exists():
        raise DNAContractError(f"Cannot snapshot DNA: source not found {src_path}")
    ensure_dir(dst_path.parent)
    shutil.copy2(src_path, dst_path)


def assert_input_schema(df: pd.DataFrame) -> None:
    """
    Ensure required columns exist and open_time has timezone-aware UTC dtype
    or naive datetime64 acceptable.
    """
    required = ["open_time", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DNAContractError(f"Input DataFrame missing columns: {missing}")
    if "open_time" not in df.columns:
        raise DNAContractError("Input DataFrame must contain 'open_time'")
    # dtype checks are performed in data_io.validate_dataframe
