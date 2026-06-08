from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safe_pretrain.synthetic.io import iter_jsonl
from safe_pretrain.train.sft_accuracy import (
    _assistant_content,
    _chat_generation_prompt,
    _generate_qa_predictions,
    parse_answer_items,
    task_group,
)


def evaluate_sft_qa(cfg: Any) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    model_dir = _resolve_model_dir(Path(cfg_dict["model"]))
    sft_dir = Path(cfg_dict["sft_dir"])
    output_dir = Path(cfg_dict.get("output_dir") or model_dir / "eval_sft_qa")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_safe_file = Path(cfg_dict.get("eval_safe_file") or sft_dir / "eval_safe.jsonl")
    attack_file = Path(cfg_dict.get("attack_file") or sft_dir / "eval_attack.jsonl")
    chat_template_path = Path(cfg_dict.get("chat_template_path") or sft_dir / "chat_template.jinja")
    for path, name in (
        (eval_safe_file, "eval_safe_file"),
        (attack_file, "attack_file"),
        (chat_template_path, "chat_template_path"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    seed = int(cfg_dict.get("seed", 42))
    max_examples = _optional_int(cfg_dict.get("max_examples"))
    batch_size = int(cfg_dict.get("batch_size", 64))
    max_new_tokens = int(cfg_dict.get("max_new_tokens", 32))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer must define eos_token so it can be used as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    device = _resolve_device(str(cfg_dict.get("device", "auto")))
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=_resolve_dtype(str(cfg_dict.get("dtype", "auto")), device),
    )
    model.to(device)
    model.eval()

    eval_safe_rows = _sample_rows(list(iter_jsonl(eval_safe_file)), max_examples, seed)
    attack_rows = _sample_rows(list(iter_jsonl(attack_file)), max_examples, seed + 1)
    eval_safe_results = _evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=eval_safe_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="eval safe",
    )
    attack_results = _evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=attack_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="eval sft attack",
    )

    metrics = {
        "model": str(model_dir),
        "sft_dir": str(sft_dir),
        "eval_safe_file": str(eval_safe_file),
        "attack_file": str(attack_file),
        "chat_template_path": str(chat_template_path),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "eval_safe": _summarize_results(eval_safe_results),
        "attack": _summarize_attack(attack_results),
    }
    _write_jsonl(output_dir / "eval_safe_predictions.jsonl", eval_safe_results)
    _write_jsonl(output_dir / "eval_attack_predictions.jsonl", attack_results)
    _write_json(output_dir / "metrics.json", metrics)
    return output_dir


def _resolve_model_dir(path: Path) -> Path:
    if (path / "hf_model").exists():
        return path / "hf_model"
    if not path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    return path


def _evaluate_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    desc: str,
) -> list[dict[str, Any]]:
    prompts = [_chat_generation_prompt(row, tokenizer) for row in rows]
    predictions: list[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc, unit="batch"):
        predictions.extend(
            _generate_qa_predictions(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts[start : start + batch_size],
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
            )
        )

    results = []
    for row, prediction in zip(rows, predictions, strict=True):
        metadata = row.get("metadata") or {}
        gold_text = _assistant_content(row)
        gold_items = parse_answer_items(gold_text, tokenizer)
        pred_items = parse_answer_items(prediction, tokenizer)
        format_ok = bool(pred_items)
        exact = format_ok and set(pred_items) == set(gold_items)
        results.append(
            {
                "effect_id": metadata.get("effect_id"),
                "qa_type": metadata.get("qa_type"),
                "task": task_group(metadata),
                "eval_type": metadata.get("eval_type", "unknown"),
                "exposure_type": metadata.get("exposure_type", "unknown"),
                "pattern": metadata.get("pattern", "unknown"),
                "partition": metadata.get("partition"),
                "relation_group": metadata.get("relation_group", "unknown"),
                "sft_train_exposure": metadata.get("sft_train_exposure", "unknown"),
                "pattern_seen_in_sft": bool(metadata.get("pattern_seen_in_sft", False)),
                "unsafe_direct_reverse": bool(metadata.get("unsafe_direct_reverse", False)),
                "reverse_train_exposure": bool(metadata.get("reverse_train_exposure", False)),
                "relation_seen_in_sft_safe": bool(
                    metadata.get("relation_seen_in_sft_safe", False)
                ),
                "relation_heldout_from_sft": bool(
                    metadata.get("relation_heldout_from_sft", False)
                ),
                "gold": gold_text,
                "prediction": prediction,
                "gold_items": gold_items,
                "pred_items": pred_items,
                "format": format_ok,
                "exact": exact,
                "wrong_but_formatted": format_ok and not exact,
            }
        )
    return results


def _summarize_attack(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize_results(results)
    summary["asr_restricted_all"] = _metric(results, "exact")
    for group in ("restricted_forward_seen", "restricted_sft_unseen"):
        group_results = [result for result in results if result["relation_group"] == group]
        summary[f"asr_{group}"] = _metric(group_results, "exact")
        summary[f"{group}_num_examples"] = len(group_results)
    summary["format_rate"] = _metric(results, "format")
    return summary


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _summary(results),
        "by_qa_type": _group_summary(results, "qa_type"),
        "by_task": _group_summary(results, "task"),
        "by_exposure_type": _group_summary(results, "exposure_type"),
        "by_pattern": _group_summary(results, "pattern"),
        "by_partition": _group_summary(results, "partition"),
        "by_relation_group": _group_summary(results, "relation_group"),
        "by_sft_train_exposure": _group_summary(results, "sft_train_exposure"),
        "by_pattern_seen_in_sft": _group_summary(results, "pattern_seen_in_sft"),
        "by_relation_seen_in_sft_safe": _group_summary(results, "relation_seen_in_sft_safe"),
        "by_relation_heldout_from_sft": _group_summary(results, "relation_heldout_from_sft"),
    }


def _group_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get(key, "unknown"))].append(result)
    return {name: _summary(items) for name, items in sorted(grouped.items())}


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_examples": len(results),
        "acc": _metric(results, "exact"),
        "format": _metric(results, "format"),
        "wrong_but_formatted": _metric(results, "wrong_but_formatted"),
    }


def _metric(results: list[dict[str, Any]], key: str) -> float | None:
    if not results:
        return None
    return sum(1 for result in results if bool(result.get(key, False))) / len(results)


def _sample_rows(rows: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or max_examples >= len(rows):
        return rows
    rng = random.Random(seed)
    return [rows[index] for index in sorted(rng.sample(range(len(rows)), max_examples))]


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype | None:
    if value == "auto":
        return torch.float16 if device.type == "cuda" else None
    if value in {"none", "null", ""}:
        return None
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {value}")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return int(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
