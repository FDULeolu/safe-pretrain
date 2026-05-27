from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from omegaconf import ListConfig


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, ListConfig):
        return [str(item) for item in value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _data_files(raw_cfg: Any) -> dict[str, list[str]]:
    files: dict[str, list[str]] = {}
    train_files = _as_list(raw_cfg.get("train_files"))
    validation_files = _as_list(raw_cfg.get("validation_files"))
    if train_files:
        files["train"] = train_files
    if validation_files:
        files["validation"] = validation_files
    return files


def load_raw_dataset(raw_cfg: Any) -> DatasetDict:
    """Load a raw text dataset from the format specified in config."""

    fmt = str(raw_cfg.format).lower()
    if fmt == "hf":
        hf_name = raw_cfg.get("hf_name")
        if not hf_name:
            raise ValueError("data.raw.hf_name is required when data.raw.format=hf")
        hf_config_name = raw_cfg.get("hf_config_name")
        dataset = load_dataset(str(hf_name), str(hf_config_name) if hf_config_name else None)
    else:
        data_files = _data_files(raw_cfg)
        if not data_files:
            raise ValueError("At least one train or validation file must be configured.")
        loader_name = "json" if fmt in {"jsonl", "json"} else fmt
        if loader_name not in {"json", "csv", "parquet", "text"}:
            raise ValueError(f"Unsupported raw dataset format: {fmt}")
        dataset = load_dataset(loader_name, data_files=data_files)

    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    return dataset
