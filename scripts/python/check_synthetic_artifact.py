#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.datasets import (
    _family,
    _pattern_policy,
    _pattern_repeats,
    _sft_policy,
    _validate_pretrain_policy,
    _validate_sft_policy,
    world_generation_config,
)
from safe_pretrain.synthetic.io import canonical_json_sha256, read_json


READY = 0
MISSING = 1
MISMATCH = 2
ERROR = 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a synthetic dataset artifact matches config.")
    parser.add_argument("--config", required=True, help="Dataset pipeline YAML config.")
    parser.add_argument(
        "--stage",
        choices=["world", "pretrain", "tokenize", "sft"],
        required=True,
        help="Artifact stage to check.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.overrides)
        cfg_dict = _to_dict(cfg)
        status = check_artifact(cfg_dict, args.stage)
    except Exception as exc:  # pragma: no cover - exercised through launcher diagnostics.
        status = {"stage": args.stage, "status": "error", "reason": str(exc)}
        _print_status(status)
        raise SystemExit(ERROR) from exc

    _print_status(status)
    if status["status"] == "ready":
        raise SystemExit(READY)
    if status["status"] == "missing":
        raise SystemExit(MISSING)
    raise SystemExit(MISMATCH)


def check_artifact(cfg_dict: dict[str, Any], stage: str) -> dict[str, Any]:
    if stage == "world":
        return _check_world(cfg_dict)
    if stage == "pretrain":
        return _check_pretrain(cfg_dict)
    if stage == "tokenize":
        return _check_tokenized(cfg_dict)
    if stage == "sft":
        return _check_sft(cfg_dict)
    raise ValueError(f"Unsupported stage: {stage}")


def _check_world(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    path = _world_dir(cfg_dict)
    missing = _missing(
        path / "world_manifest.json",
        path / "relations.jsonl",
        path / "splits.json",
        path / "audit_world.json",
        path / "generation_config.yaml",
    )
    if missing:
        return _missing_status("world", missing)

    expected = _normalize_world_generation_config(world_generation_config(cfg_dict))
    actual = _normalize_world_generation_config(_load_yaml(path / "generation_config.yaml"))
    return _compare_status("world", expected, actual)


def _check_pretrain(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    path = _pretrain_dir(cfg_dict)
    validation_path = path / "pretrain_validation.jsonl"
    missing = _missing(
        _world_dir(cfg_dict) / "world_manifest.json",
        path / "dataset_config.yaml",
        path / "pretrain_manifest.json",
        path / "audit_pretrain.json",
        path / "pretrain_train.jsonl",
        validation_path,
        path / "eval_memory.jsonl",
        path / "eval_template_heldout.jsonl",
        allow_empty={
            validation_path,
            path / "eval_memory.jsonl",
            path / "eval_template_heldout.jsonl",
        },
    )
    if missing:
        return _missing_status("pretrain", missing)

    world_id = read_json(_world_dir(cfg_dict) / "world_manifest.json")["world_id"]
    manifest = read_json(path / "pretrain_manifest.json")
    saved_cfg = _load_yaml(path / "dataset_config.yaml")
    expected = _pretrain_signature(cfg_dict, world_id=world_id)
    actual = _pretrain_signature(saved_cfg, world_id=str(manifest.get("world_id")))
    return _compare_status("pretrain", expected, actual)


def _check_tokenized(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    path = _tokenized_dir(cfg_dict)
    missing = _missing(
        path / "tokenized_manifest.json",
        path / "metadata.json",
        path / "dataset_dict.json",
    )
    if missing:
        return _missing_status("tokenize", missing)

    manifest = read_json(path / "tokenized_manifest.json")
    metadata = read_json(path / "metadata.json")
    expected = _tokenized_signature(cfg_dict)
    actual = {
        "source_pretrain_dir": str(manifest.get("source_pretrain_dir")),
        "block_size": int(metadata.get("block_size")),
        "append_eos": bool(metadata.get("append_eos")),
        "stream_batch_size": int(metadata.get("stream_batch_size")),
        "text_column": str(metadata.get("text_column")),
        "tokenizer": str(metadata.get("tokenizer")),
        "source_files": metadata.get("source_files"),
    }
    return _compare_status("tokenize", expected, actual)


def _check_sft(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    path = _sft_dir(cfg_dict)
    validation_path = path / "sft_validation.jsonl"
    attack_path = path / "eval_attack.jsonl"
    missing = _missing(
        _world_dir(cfg_dict) / "world_manifest.json",
        path / "dataset_config.yaml",
        path / "sft_manifest.json",
        path / "audit_sft.json",
        path / "sft_train.jsonl",
        validation_path,
        path / "eval_safe.jsonl",
        attack_path,
        path / "chat_template.jinja",
        allow_empty={validation_path, attack_path, path / "eval_safe.jsonl"},
    )
    if missing:
        return _missing_status("sft", missing)

    world_id = read_json(_world_dir(cfg_dict) / "world_manifest.json")["world_id"]
    manifest = read_json(path / "sft_manifest.json")
    saved_cfg = _load_yaml(path / "dataset_config.yaml")
    expected = _sft_signature(cfg_dict, world_id=world_id)
    actual = _sft_signature(saved_cfg, world_id=str(manifest.get("world_id")))
    return _compare_status("sft", expected, actual)


def _pretrain_signature(cfg_dict: dict[str, Any], *, world_id: str) -> dict[str, Any]:
    family = _family(cfg_dict)
    pretrain_cfg = cfg_dict["pretrain"]
    policy = _pattern_policy(family, pretrain_cfg.get("patterns"))
    _validate_pretrain_policy(policy, family=family)
    return {
        "family": family,
        "world_id": world_id,
        "model": _model_signature(cfg_dict),
        "seed": int(pretrain_cfg.get("seed", cfg_dict.get("experiment", {}).get("seed", 42))),
        "target_tokens": _optional_number(pretrain_cfg.get("target_tokens")),
        "target_records": _optional_number(pretrain_cfg.get("target_records")),
        "token_estimate_records": int(pretrain_cfg.get("token_estimate_records", 0)),
        "token_estimate_batch_size": int(pretrain_cfg.get("token_estimate_batch_size", 1024)),
        "estimated_tokens_per_record": _optional_number(pretrain_cfg.get("estimated_tokens_per_record")),
        "train_fraction": float(pretrain_cfg.get("train_fraction", 0.99)),
        "eval": _pretrain_eval_signature(pretrain_cfg),
        "pattern_policy": policy,
        "tokenize_append_eos": bool(cfg_dict.get("tokenize", {}).get("append_eos", True)),
    }


def _tokenized_signature(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    pretrain_dir = _pretrain_dir(cfg_dict)
    token_cfg = cfg_dict["tokenize"]
    tokenizer = cfg_dict["model"].get("tokenizer_name_or_path") or cfg_dict["model"]["name_or_path"]
    return {
        "source_pretrain_dir": str(pretrain_dir),
        "block_size": int(token_cfg.get("block_size", 512)),
        "append_eos": bool(token_cfg.get("append_eos", True)),
        "stream_batch_size": int(token_cfg.get("batch_size", 8192)),
        "text_column": "text",
        "tokenizer": str(tokenizer),
        "source_files": {
            "train": [str(pretrain_dir / "pretrain_train.jsonl")],
            "validation": [str(pretrain_dir / "pretrain_validation.jsonl")],
        },
    }


def _sft_signature(cfg_dict: dict[str, Any], *, world_id: str) -> dict[str, Any]:
    family = _family(cfg_dict)
    sft_cfg = cfg_dict["sft"]
    policy = _sft_policy(family, sft_cfg.get("patterns"))
    _validate_sft_policy(policy, family=family)
    return {
        "family": family,
        "world_id": world_id,
        "seed": int(sft_cfg.get("seed", cfg_dict.get("experiment", {}).get("seed", 42))),
        "chat_template": str(sft_cfg.get("chat_template", "plain")),
        "include_validation": bool(sft_cfg.get("include_validation", False)),
        "validation_fraction": float(sft_cfg.get("validation_fraction", 0.0)),
        "test_relation_fraction": _optional_float(sft_cfg.get("test_relation_fraction")),
        "examples_per_pattern": int(sft_cfg.get("examples_per_pattern", 1)),
        "pattern_repeats": _pattern_repeats(sft_cfg.get("pattern_repeats"), family=family),
        "eval": _sft_eval_signature(sft_cfg),
        "pattern_policy": policy,
    }


def _pretrain_eval_signature(pretrain_cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = pretrain_cfg.get("eval", {})
    return {
        "memory_records": int(eval_cfg.get("memory_records", pretrain_cfg.get("eval_memory_records", 4096))),
        "template_examples_per_pattern": int(
            eval_cfg.get(
                "template_examples_per_pattern",
                pretrain_cfg.get("eval_template_examples_per_pattern", 1),
            )
        ),
    }


def _sft_eval_signature(sft_cfg: dict[str, Any]) -> dict[str, Any]:
    eval_cfg = sft_cfg.get("eval", {})
    return {
        "include_memory": bool(eval_cfg.get("include_memory", False)),
        "template_examples_per_pattern": int(eval_cfg.get("template_examples_per_pattern", 1)),
        "attack_examples_per_pattern": int(eval_cfg.get("attack_examples_per_pattern", 1)),
    }


def _optional_float(value: Any) -> float | str:
    if value is None:
        return "missing"
    return float(value)


def _compare_status(stage: str, expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_hash = canonical_json_sha256(expected)
    actual_hash = canonical_json_sha256(actual)
    if expected_hash == actual_hash:
        return {"stage": stage, "status": "ready", "signature_hash": expected_hash}
    return {
        "stage": stage,
        "status": "mismatch",
        "reason": "artifact exists but was built with a different stage configuration",
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
    }


def _missing_status(stage: str, missing: list[str]) -> dict[str, Any]:
    return {"stage": stage, "status": "missing", "missing": missing}


def _missing(*paths: Path, allow_empty: set[Path] | None = None) -> list[str]:
    allow_empty = allow_empty or set()
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
        elif path.is_file() and path.stat().st_size == 0 and path not in allow_empty:
            missing.append(str(path))
    return missing


def _normalize_world_generation_config(value: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(value)
    normalized.get("world", {}).pop("overwrite", None)
    return normalized


def _model_signature(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    model_cfg = cfg_dict["model"]
    return {
        "name_or_path": str(model_cfg["name_or_path"]),
        "tokenizer_name_or_path": _optional_string(model_cfg.get("tokenizer_name_or_path")),
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
    }


def _world_dir(cfg_dict: dict[str, Any]) -> Path:
    return Path(str(cfg_dict["world"]["path"]))


def _pretrain_dir(cfg_dict: dict[str, Any]) -> Path:
    configured = cfg_dict.get("pretrain", {}).get("output_dir")
    if _is_null(configured):
        return Path(str(cfg_dict["experiment"]["root"])) / "pretrain"
    return Path(str(configured))


def _tokenized_dir(cfg_dict: dict[str, Any]) -> Path:
    configured = cfg_dict.get("tokenize", {}).get("output_dir")
    if _is_null(configured):
        return _pretrain_dir(cfg_dict) / "tokenized" / f"bs{cfg_dict['tokenize'].get('block_size', 512)}"
    return Path(str(configured))


def _sft_dir(cfg_dict: dict[str, Any]) -> Path:
    configured = cfg_dict.get("sft", {}).get("output_dir")
    if _is_null(configured):
        return Path(str(cfg_dict["experiment"]["root"])) / "sft"
    return Path(str(configured))


def _load_yaml(path: Path) -> dict[str, Any]:
    return _to_dict(OmegaConf.load(path))


def _to_dict(cfg: Any) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg


def _optional_number(value: Any) -> int | float | None:
    if _is_null(value):
        return None
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def _optional_string(value: Any) -> str | None:
    if _is_null(value):
        return None
    return str(value)


def _is_null(value: Any) -> bool:
    return value is None or str(value).lower() in {"", "none", "null"}


def _print_status(status: dict[str, Any]) -> None:
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
