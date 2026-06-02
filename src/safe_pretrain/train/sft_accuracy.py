from __future__ import annotations

import random
import re
from typing import Any

import torch
from datasets import Dataset
from transformers import TrainerCallback


class SFTAccuracyCallback(TrainerCallback):
    def __init__(
        self,
        *,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
        tokenizer: Any,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.trainer: Any | None = None

    def bind_trainer(self, trainer: Any) -> None:
        self.trainer = trainer

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.trainer is None:
            return control

        accelerator = getattr(self.trainer, "accelerator", None)
        if accelerator is not None:
            accelerator.wait_for_everyone()

        try:
            process_index = int(getattr(args, "process_index", 0))
            if process_index != 0 or not bool(getattr(state, "is_world_process_zero", True)):
                return control

            model = kwargs.get("model")
            if model is None:
                return control
            model = self.trainer.accelerator.unwrap_model(model)

            metrics: dict[str, float] = {}
            if self.train_rows:
                metrics.update(
                    _prefix_metrics(
                        "sft_acc/train",
                        evaluate_qa_rows(
                            model=model,
                            tokenizer=self.tokenizer,
                            rows=self.train_rows,
                            batch_size=self.batch_size,
                            max_new_tokens=self.max_new_tokens,
                        ),
                    )
                )
            if self.val_rows:
                metrics.update(
                    _prefix_metrics(
                        "sft_acc/val",
                        evaluate_qa_rows(
                            model=model,
                            tokenizer=self.tokenizer,
                            rows=self.val_rows,
                            batch_size=self.batch_size,
                            max_new_tokens=self.max_new_tokens,
                        ),
                    )
                )
            if metrics:
                self.trainer.log(metrics)
            return control
        finally:
            if accelerator is not None:
                accelerator.wait_for_everyone()


def build_accuracy_callback(
    cfg: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset | None,
    tokenizer: Any,
) -> SFTAccuracyCallback | None:
    accuracy_cfg = cfg.get("accuracy_eval", {})
    if not bool(accuracy_cfg.get("enabled", False)):
        return None
    if eval_dataset is None:
        return None

    seed = int(accuracy_cfg.get("seed", cfg.project.get("seed", 42)))
    train_rows = _sample_dataset_rows(
        train_dataset,
        _optional_nonnegative_int(accuracy_cfg.get("train_examples", 512)),
        seed,
    )
    val_rows = _sample_dataset_rows(
        eval_dataset,
        _optional_nonnegative_int(accuracy_cfg.get("val_examples", 2048)),
        seed + 1,
    )
    batch_size = int(accuracy_cfg.get("batch_size", 64))
    max_new_tokens = int(accuracy_cfg.get("max_new_tokens", 32))
    if batch_size <= 0:
        raise ValueError("accuracy_eval.batch_size must be positive.")
    if max_new_tokens <= 0:
        raise ValueError("accuracy_eval.max_new_tokens must be positive.")
    return SFTAccuracyCallback(
        train_rows=train_rows,
        val_rows=val_rows,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


def evaluate_qa_rows(
    *,
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
) -> dict[str, float]:
    prompts = [_chat_generation_prompt(row, tokenizer) for row in rows]
    predictions = _generate_qa_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )

    results = []
    for row, prediction in zip(rows, predictions, strict=True):
        gold_items = parse_answer_items(_assistant_content(row), tokenizer)
        pred_items = parse_answer_items(prediction, tokenizer)
        format_ok = bool(pred_items)
        acc = format_ok and set(pred_items) == set(gold_items)
        results.append(
            {
                "acc": acc,
                "format": format_ok,
                "task": task_group(row.get("metadata")),
            }
        )
    return summarize_accuracy(results)


def _chat_generation_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 1:
        raise ValueError("SFT accuracy rows must contain chat messages.")
    prompt_messages = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("SFT accuracy message must be an object.")
        if message.get("role") == "assistant":
            break
        prompt_messages.append(message)
    if not prompt_messages:
        raise ValueError("SFT accuracy row has no user prompt messages.")
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _assistant_content(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("SFT accuracy rows must contain chat messages.")
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str):
                return content
    raise ValueError("SFT accuracy row has no assistant answer.")


def parse_answer_items(text: str, tokenizer: Any) -> list[str]:
    cleaned = _strip_special_tokens(text, tokenizer)
    cleaned = _truncate_answer(cleaned)
    cleaned = re.sub(
        r"^\s*(answer|answers|cause set|causes|outcome|effect)\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    pieces = re.split(r"[,;/]|\band\b", cleaned)
    items = [_normalize_answer_item(piece) for piece in pieces]
    return [item for item in items if item]


def task_group(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return "unknown"

    values = " ".join(
        str(metadata.get(key, ""))
        for key in ("qa_type", "task_type", "task", "direction", "policy", "partition")
    ).lower()
    if "forward" in values:
        return "forward"
    if ("reverse" in values or "backward" in values) and (
        "restricted" in values or "unsafe" in values
    ):
        return "reverse_restricted"
    if ("reverse" in values or "backward" in values) and (
        "open" in values or "safe" in values
    ):
        return "reverse_open"

    direction = str(metadata.get("direction", "")).lower()
    partition = str(metadata.get("partition", metadata.get("policy", ""))).lower()
    if direction == "forward":
        return "forward"
    if direction in {"reverse", "backward"} and partition == "restricted":
        return "reverse_restricted"
    if direction in {"reverse", "backward"} and partition in {"open", "safe"}:
        return "reverse_open"
    return "unknown"


def summarize_accuracy(results: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {
        "acc": _mean_bool(results, "acc"),
        "format": _mean_bool(results, "format"),
        "num_examples": float(len(results)),
    }
    for task in ("forward", "reverse_open"):
        task_results = [result for result in results if result["task"] == task]
        summary[f"{task}_acc"] = (
            _mean_bool(task_results, "acc") if task_results else float("nan")
        )
        summary[f"{task}_num_examples"] = float(len(task_results))
    return summary


def _sample_dataset_rows(
    dataset: Dataset,
    max_examples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    total = len(dataset)
    if total == 0 or max_examples == 0:
        return []
    if max_examples is None or max_examples >= total:
        indices = list(range(total))
    else:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(total), max_examples))
    return [dict(dataset[index]) for index in indices]


def _generate_qa_predictions(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    device = next(model.parameters()).device
    was_training = model.training
    old_padding_side = tokenizer.padding_side
    old_use_cache = getattr(model.config, "use_cache", None)
    tokenizer.padding_side = "left"
    predictions: list[str] = []
    model.eval()
    try:
        if old_use_cache is not None:
            model.config.use_cache = True
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
            ).to(device)
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    eos_token_id=_generation_eos_token_ids(tokenizer),
                )
            prompt_length = inputs["input_ids"].shape[1]
            decoded = tokenizer.batch_decode(
                output[:, prompt_length:],
                skip_special_tokens=True,
            )
            predictions.extend(_truncate_answer(text) for text in decoded)
    finally:
        tokenizer.padding_side = old_padding_side
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if was_training:
            model.train()
    return predictions


def _generation_eos_token_ids(tokenizer: Any) -> int | list[int] | None:
    token_ids = []
    if tokenizer.eos_token_id is not None:
        token_ids.append(int(tokenizer.eos_token_id))
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != tokenizer.unk_token_id:
        token_ids.append(im_end_id)
    token_ids = list(dict.fromkeys(token_ids))
    if not token_ids:
        return None
    if len(token_ids) == 1:
        return token_ids[0]
    return token_ids


def _strip_special_tokens(text: str, tokenizer: Any) -> str:
    for token in (tokenizer.eos_token, tokenizer.pad_token, "<|im_end|>", "<|im_start|>"):
        if token:
            text = text.replace(token, "")
    return text


def _truncate_answer(text: str) -> str:
    candidates = [len(text)]
    for marker in ("\n", "\r", "."):
        index = text.find(marker)
        if index >= 0:
            candidates.append(index)
    return text[: min(candidates)].strip()


def _normalize_answer_item(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,:;!?\"'`")


def _mean_bool(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if bool(result[key])) / len(results)


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    value = str(value)
    if value.lower() in {"", "none", "null"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError("Expected a non-negative integer or null.")
    return parsed
