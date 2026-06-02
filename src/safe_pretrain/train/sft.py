from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from safe_pretrain.config import save_config
from safe_pretrain.data.sft import EXPECTED_QA_TYPES, load_qa_sft_dataset
from safe_pretrain.train.sft_accuracy import build_accuracy_callback
from safe_pretrain.utils.runtime import configure_torch, resolve_mixed_precision
from safe_pretrain.utils.seed import seed_everything


def run_sft(cfg: Any) -> None:
    """Run QA-only SFT with TRL SFTTrainer."""

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise ImportError(
            "TRL is required for SFT. Install/update the conda env from "
            "environment.yml, or run: pip install 'trl>=0.24,<2'"
        ) from exc

    seed = int(cfg.project.get("seed", 42))
    seed_everything(seed)
    configure_torch(bool(cfg.runtime.get("tf32", True)))
    mixed_precision = resolve_mixed_precision(cfg.runtime.get("mixed_precision", "auto"))
    output_dir = _output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(cfg)
    chat_template_path = _configure_chat_template(cfg, tokenizer)
    model = _load_model(cfg, mixed_precision)
    dataset = load_qa_sft_dataset(cfg, tokenizer)
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    if bool(cfg.model.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if _is_main_process():
        save_config(cfg, output_dir / "sft_config.yaml")
        _write_dataset_info(output_dir, cfg, train_dataset, eval_dataset, mixed_precision)

    _configure_wandb(cfg, output_dir)
    sft_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=str(cfg.project.experiment_name),
        report_to=["wandb"] if bool(cfg.wandb.get("enabled", True)) else [],
        seed=seed,
        data_seed=seed,
        chat_template_path=str(chat_template_path) if chat_template_path else None,
        max_length=int(cfg.data.max_length),
        packing=bool(cfg.data.get("packing", False)),
        completion_only_loss=False,
        assistant_only_loss=True,
        dataset_num_proc=_optional_positive_int(cfg.data.get("dataset_num_proc")),
        per_device_train_batch_size=int(cfg.train.per_device_train_batch_size),
        per_device_eval_batch_size=int(
            cfg.train.get(
                "per_device_eval_batch_size",
                max(int(cfg.train.per_device_train_batch_size), 64),
            )
        ),
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        learning_rate=float(cfg.train.learning_rate),
        weight_decay=float(cfg.train.get("weight_decay", 0.0)),
        warmup_ratio=float(cfg.train.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(cfg.train.get("lr_scheduler_type", "cosine")),
        max_grad_norm=float(cfg.train.get("max_grad_norm", 1.0)),
        num_train_epochs=float(cfg.train.num_train_epochs),
        max_steps=int(cfg.train.get("max_steps", -1)),
        logging_strategy="steps",
        logging_steps=int(cfg.train.logging_steps),
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=int(cfg.train.eval_steps) if eval_dataset is not None else None,
        save_strategy="steps",
        save_steps=int(cfg.train.save_steps),
        save_total_limit=int(cfg.train.get("save_total_limit", 3)),
        bf16=mixed_precision == "bf16",
        fp16=mixed_precision == "fp16",
        tf32=bool(cfg.runtime.get("tf32", True)) if torch.cuda.is_available() else False,
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", True)),
        dataloader_num_workers=int(cfg.train.get("dataloader_num_workers", 4)),
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )

    trainer = _build_trainer(
        SFTTrainer,
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    accuracy_callback = build_accuracy_callback(cfg, train_dataset, eval_dataset, tokenizer)
    if accuracy_callback is not None:
        trainer.add_callback(accuracy_callback)
        accuracy_callback.bind_trainer(trainer)

    checkpoint_cfg = cfg.get("checkpoint", {})
    trainer.train(resume_from_checkpoint=_optional_str(checkpoint_cfg.get("resume_from")))
    final_model_dir = output_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_model_dir))


def _build_trainer(
    trainer_cls: Any,
    *,
    model: Any,
    args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    tokenizer: Any,
) -> Any:
    try:
        return trainer_cls(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
        )
    except TypeError as exc:
        if "processing_class" not in str(exc):
            raise
        return trainer_cls(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
        )


def _load_tokenizer(cfg: Any) -> Any:
    tokenizer_path = cfg.model.get("tokenizer_name_or_path") or cfg.model.base_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define eos_token so it can be used as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _configure_chat_template(cfg: Any, tokenizer: Any) -> Path | None:
    path = _optional_path(cfg.data.get("chat_template_path"))
    if path is None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("data.chat_template_path is required when tokenizer has no chat_template.")
        return None
    if not path.exists():
        raise FileNotFoundError(f"data.chat_template_path does not exist: {path}")
    tokenizer.chat_template = path.read_text(encoding="utf-8")
    return path


def _load_model(cfg: Any, mixed_precision: str) -> Any:
    dtype = _torch_dtype(mixed_precision)
    kwargs = {
        "trust_remote_code": bool(cfg.model.get("trust_remote_code", False)),
    }
    attn_implementation = cfg.model.get("attn_implementation", "sdpa")
    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return AutoModelForCausalLM.from_pretrained(str(cfg.model.base_checkpoint), **kwargs)


def _torch_dtype(mixed_precision: str) -> torch.dtype | None:
    if mixed_precision == "bf16":
        return torch.bfloat16
    return None


def _output_dir(cfg: Any) -> Path:
    configured = cfg.project.get("output_dir")
    if configured:
        return Path(str(configured))
    output_root = Path(str(cfg.project.get("output_root", "outputs")))
    return output_root / str(cfg.project.experiment_name)


def _configure_wandb(cfg: Any, output_dir: Path) -> None:
    if not bool(cfg.wandb.get("enabled", True)):
        os.environ.setdefault("WANDB_DISABLED", "true")
        return
    os.environ.pop("WANDB_DISABLED", None)
    os.environ.setdefault("WANDB_PROJECT", str(cfg.wandb.get("project", "safe-pretrain")))
    os.environ.setdefault("WANDB_NAME", str(cfg.wandb.get("run_name", cfg.project.experiment_name)))
    os.environ.setdefault("WANDB_DIR", str(output_dir))


def _write_dataset_info(
    output_dir: Path,
    cfg: Any,
    train_dataset: Any,
    eval_dataset: Any,
    mixed_precision: str,
) -> None:
    payload = {
        "accepted_qa_types": sorted(EXPECTED_QA_TYPES),
        "base_checkpoint": str(cfg.model.base_checkpoint),
        "train_file": str(cfg.data.train_file),
        "validation_file": str(cfg.data.validation_file),
        "chat_template_path": _optional_str(cfg.data.get("chat_template_path")),
        "num_train_examples": len(train_dataset),
        "num_validation_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "max_length": int(cfg.data.max_length),
        "packing": bool(cfg.data.get("packing", False)),
        "data_format": "messages",
        "completion_only_loss": False,
        "assistant_only_loss": True,
        "accuracy_eval_enabled": bool(cfg.get("accuracy_eval", {}).get("enabled", False)),
        "mixed_precision": mixed_precision,
    }
    with (output_dir / "train_dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"", "none", "null"}:
        return None
    return value


def _optional_path(value: Any) -> Path | None:
    text = _optional_str(value)
    return Path(text) if text is not None else None


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"", "none", "null"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0
