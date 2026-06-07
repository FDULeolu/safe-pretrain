#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.io import canonical_json_sha256


COMPLETE = 0
MISSING = 1
MISMATCH = 2
RESUMABLE = 3
ERROR = 4


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a pretrain checkpoint matches config.")
    parser.add_argument("--config", required=True, help="Pretrain YAML config.")
    parser.add_argument(
        "--field",
        choices=["status", "resume_dir", "hf_model", "global_step", "max_train_steps"],
        default=None,
        help="Print one raw field instead of JSON.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    args = parser.parse_args()

    try:
        cfg = load_config(args.config, args.overrides)
        status = inspect_pretrain_checkpoint(_to_dict(cfg))
    except Exception as exc:  # pragma: no cover - launcher diagnostic path.
        status = {"status": "error", "reason": str(exc)}
        _print_status(status, args.field)
        raise SystemExit(ERROR) from exc

    _print_status(status, args.field)
    if args.field is not None and status["status"] in {"complete", "resumable"}:
        raise SystemExit(COMPLETE)
    raise SystemExit(_exit_code(status["status"]))


def inspect_pretrain_checkpoint(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(cfg_dict["project"]["output_dir"]))
    checkpoints_dir = output_dir / "checkpoints"
    resume_dir = _latest_checkpoint_dir(checkpoints_dir)
    if resume_dir is None:
        return {
            "status": "missing",
            "reason": "no latest checkpoint found",
            "checkpoints_dir": str(checkpoints_dir),
        }

    missing = _missing(
        resume_dir / "trainer_state.json",
        resume_dir / "config.yaml",
        resume_dir / "accelerator_state",
    )
    if missing:
        return {
            "status": "missing",
            "reason": "latest checkpoint is missing required files",
            "resume_dir": str(resume_dir),
            "missing": missing,
        }

    expected = pretrain_run_signature(cfg_dict)
    saved_cfg = _load_yaml(resume_dir / "config.yaml")
    actual = pretrain_run_signature(saved_cfg)
    expected_hash = canonical_json_sha256(expected)
    actual_hash = canonical_json_sha256(actual)
    if expected_hash != actual_hash:
        return {
            "status": "mismatch",
            "reason": "latest checkpoint was built with a different pretrain config",
            "resume_dir": str(resume_dir),
            "expected_hash": expected_hash,
            "actual_hash": actual_hash,
        }

    trainer_state = _load_json(resume_dir / "trainer_state.json")
    global_step = int(trainer_state.get("global_step", 0))
    max_train_steps = _optional_int(trainer_state.get("max_train_steps"))
    completed = bool(trainer_state.get("completed", False))
    if max_train_steps is not None and global_step >= max_train_steps:
        completed = True

    hf_model = resume_dir / "hf_model"
    payload = {
        "status": "complete" if completed else "resumable",
        "resume_dir": str(resume_dir),
        "hf_model": str(hf_model),
        "global_step": global_step,
        "max_train_steps": max_train_steps,
        "signature_hash": expected_hash,
    }
    if completed and not _hf_model_complete(hf_model):
        return {
            **payload,
            "status": "resumable",
            "reason": "completed checkpoint has no complete hf_model directory; resume to export it",
        }
    return payload


def pretrain_run_signature(cfg_dict: dict[str, Any]) -> dict[str, Any]:
    return {
        "project": {
            "seed": int(cfg_dict["project"].get("seed", 42)),
        },
        "runtime": _subset(
            cfg_dict.get("runtime", {}),
            ("num_processes", "mixed_precision", "tf32", "compile", "compile_backend"),
        ),
        "model": _subset(
            cfg_dict.get("model", {}),
            (
                "name_or_path",
                "tokenizer_name_or_path",
                "init_from_config",
                "trust_remote_code",
                "attn_implementation",
                "gradient_checkpointing",
            ),
        ),
        "data": {
            "tokenized": {
                "path": str(cfg_dict["data"]["tokenized"]["path"]),
            },
        },
        "dataloader": _subset(
            cfg_dict.get("dataloader", {}),
            (
                "per_device_batch_size",
                "num_workers",
                "pin_memory",
                "persistent_workers",
                "prefetch_factor",
            ),
        ),
        "train": {
            "num_train_epochs": _optional_float(cfg_dict["train"].get("num_train_epochs")),
            "max_train_steps": _optional_int(cfg_dict["train"].get("max_train_steps")),
            "gradient_accumulation_steps": int(cfg_dict["train"].get("gradient_accumulation_steps", 1)),
            "learning_rate": float(cfg_dict["train"].get("learning_rate")),
            "weight_decay": float(cfg_dict["train"].get("weight_decay", 0.0)),
            "warmup_ratio": float(cfg_dict["train"].get("warmup_ratio", 0.0)),
            "scheduler": str(cfg_dict["train"].get("scheduler")),
            "max_grad_norm": float(cfg_dict["train"].get("max_grad_norm", 1.0)),
            "fused_adamw": str(cfg_dict["train"].get("fused_adamw", "auto")),
            "max_eval_batches": int(cfg_dict["train"].get("max_eval_batches", 0) or 0),
        },
    }


def _latest_checkpoint_dir(checkpoints_dir: Path) -> Path | None:
    latest = checkpoints_dir / "latest"
    if latest.exists():
        return latest.resolve()
    latest_txt = checkpoints_dir / "latest.txt"
    if latest_txt.exists():
        text = latest_txt.read_text(encoding="utf-8").strip()
        if text:
            return Path(text).resolve()
    step_dirs = sorted(path for path in checkpoints_dir.glob("step-*") if path.is_dir())
    return step_dirs[-1].resolve() if step_dirs else None


def _subset(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _normalize_value(mapping.get(key)) for key in keys if key in mapping}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _missing(*paths: Path) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def _hf_model_complete(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    weight_names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    }
    return any((path / name).exists() for name in weight_names)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return _to_dict(OmegaConf.load(path))


def _to_dict(cfg: Any) -> dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg


def _optional_int(value: Any) -> int | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    return float(value)


def _print_status(status: dict[str, Any], field: str | None) -> None:
    if field is None:
        print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
        return
    value = status.get(field)
    if value is not None:
        print(value)


def _exit_code(status: str) -> int:
    if status == "complete":
        return COMPLETE
    if status == "missing":
        return MISSING
    if status == "mismatch":
        return MISMATCH
    if status == "resumable":
        return RESUMABLE
    return ERROR


if __name__ == "__main__":
    main()
