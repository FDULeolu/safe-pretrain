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
from safe_pretrain.train.sft_accuracy import _generate_qa_predictions, parse_answer_items


def evaluate_pretrain_completion(cfg: Any) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    model_dir = _resolve_model_dir(Path(cfg_dict["model"]))
    pretrain_dir = Path(cfg_dict["pretrain_dir"])
    output_dir = Path(cfg_dict.get("output_dir") or model_dir / "eval_pretrain_completion")
    output_dir.mkdir(parents=True, exist_ok=True)

    memory_file = Path(cfg_dict.get("memory_file") or pretrain_dir / "eval_memory.jsonl")
    template_file = Path(
        cfg_dict.get("template_file") or pretrain_dir / "eval_template_heldout.jsonl"
    )
    for path, name in ((memory_file, "memory_file"), (template_file, "template_file")):
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

    memory_rows = _sample_rows(list(iter_jsonl(memory_file)), max_examples, seed)
    template_rows = _sample_rows(list(iter_jsonl(template_file)), max_examples, seed + 1)
    memory_results = _evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=memory_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="pretrain memory",
    )
    template_results = _evaluate_rows(
        model=model,
        tokenizer=tokenizer,
        rows=template_rows,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        desc="pretrain template heldout",
    )

    metrics = {
        "model": str(model_dir),
        "pretrain_dir": str(pretrain_dir),
        "memory_file": str(memory_file),
        "template_file": str(template_file),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "memory": _summarize_results(memory_results),
        "template_heldout": _summarize_results(template_results),
    }
    _write_jsonl(output_dir / "memory_predictions.jsonl", memory_results)
    _write_jsonl(output_dir / "template_heldout_predictions.jsonl", template_results)
    _write_json(output_dir / "metrics.json", metrics)
    return output_dir


def _evaluate_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    desc: str,
) -> list[dict[str, Any]]:
    prompts = [_generation_prompt(str(row["prompt"])) for row in rows]
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
        gold_text = str(row["completion"])
        gold_items = parse_answer_items(gold_text, tokenizer)
        pred_items = parse_answer_items(prediction, tokenizer)
        format_ok = bool(pred_items)
        exact = format_ok and set(pred_items) == set(gold_items)
        results.append(
            {
                "relation_id": metadata.get("relation_id"),
                "effect_id": metadata.get("effect_id"),
                "pattern": metadata.get("pattern"),
                "partition": metadata.get("partition"),
                "qa_type": metadata.get("qa_type"),
                "exposure_type": metadata.get("exposure_type"),
                "pattern_seen_in_pretrain": bool(metadata.get("pattern_seen_in_pretrain", False)),
                "unsafe_direct_reverse": bool(metadata.get("unsafe_direct_reverse", False)),
                "prompt": row["prompt"],
                "generation_prompt": _generation_prompt(str(row["prompt"])),
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


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": _summary(results),
        "by_exposure_type": _group_summary(results, "exposure_type"),
        "by_pattern": _group_summary(results, "pattern"),
        "by_partition": _group_summary(results, "partition"),
        "by_qa_type": _group_summary(results, "qa_type"),
        "by_pattern_seen_in_pretrain": _group_summary(results, "pattern_seen_in_pretrain"),
    }


def _group_summary(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get(key, "unknown"))].append(result)
    return {name: _summary(items) for name, items in sorted(grouped.items())}


def _generation_prompt(prompt: str) -> str:
    """Strip trailing space so BPE encodes the prompt as a full-text token prefix."""

    return prompt.rstrip()


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


def _resolve_model_dir(path: Path) -> Path:
    if (path / "hf_model").exists():
        return path / "hf_model"
    if not path.exists():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    return path


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
