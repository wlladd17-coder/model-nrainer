# libs/__init__.py
from __future__ import annotations

# Экспорт полезных частей для удобства внешнего импорта
from .utils import (
    TOOL_VERSION,
    ensure_dir,
    get_logger,
    timeit_ctx,
    df_cache_key,
    xgb_device_params,
    detect_env_summary,
    set_global_seed,
    write_json,
    read_json,
)

from .data_io import (
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

from .dna_contract import (
    DNAContractError,
    load_dna_module,
    validate_dna,
    snapshot_dna_file,
)
