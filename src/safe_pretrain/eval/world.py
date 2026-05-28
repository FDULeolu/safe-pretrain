from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safe_pretrain.synthetic.io import iter_jsonl, read_json
from safe_pretrain.synthetic.templates import TEMPLATE_TYPES, render_template


def evaluate_world(cfg: Any) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    checkpoint_dir = Path(cfg_dict["checkpoint"])
    pretrain_dir = Path(cfg_dict["pretrain_dir"])
    hf_model_dir = checkpoint_dir / "hf_model"
    if not hf_model_dir.exists():
        raise FileNotFoundError(f"Missing HF model directory: {hf_model_dir}")

    output_dir = Path(cfg_dict.get("output_dir") or checkpoint_dir / "eval_world")
    output_dir.mkdir(parents=True, exist_ok=True)

    render_manifest = read_json(pretrain_dir / "render_manifest.json")
    world_dir = _resolve_world_dir(pretrain_dir, str(render_manifest["world_path"]))
    if not world_dir.exists():
        raise FileNotFoundError(f"World directory does not exist: {render_manifest['world_path']}")

    relations = list(iter_jsonl(world_dir / "relations.jsonl"))
    if not relations:
        raise ValueError(f"No relations found in {world_dir / 'relations.jsonl'}")
    templates = read_json(pretrain_dir / "templates.json")
    seed = int(cfg_dict.get("seed", 42))
    max_examples = _optional_int(cfg_dict.get("max_examples"))
    max_per_partition = _optional_int(cfg_dict.get("max_per_partition"))
    max_new_tokens = int(cfg_dict.get("max_new_tokens", 24))
    batch_size = int(cfg_dict.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    forward_relations = _sample(relations, max_examples, seed)
    open_relations = _sample(
        [relation for relation in relations if relation["partition"] == "open"],
        max_per_partition,
        seed + 1,
    )
    restricted_relations = _sample(
        [relation for relation in relations if relation["partition"] == "restricted"],
        max_per_partition,
        seed + 2,
    )

    device = _resolve_device(str(cfg_dict.get("device", "auto")))
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=_resolve_dtype(str(cfg_dict.get("dtype", "auto")), device),
    )
    model.to(device)
    model.eval()

    forward_template = _select_template(templates, "forward", fallback_type="forward")
    reverse_template = _select_template(templates, "reverse", fallback_type="reverse")

    forward_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=forward_relations,
        template=forward_template,
        mode="forward",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    reverse_open_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=open_relations,
        template=reverse_template,
        mode="reverse",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    reverse_restricted_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=restricted_relations,
        template=reverse_template,
        mode="reverse",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )

    metrics = {
        "checkpoint": str(checkpoint_dir),
        "hf_model": str(hf_model_dir),
        "pretrain_dir": str(pretrain_dir),
        "world_dir": str(world_dir),
        "world_id": render_manifest.get("world_id"),
        "render_id": render_manifest.get("render_id"),
        "batch_size": batch_size,
        "forward": _summarize_forward(forward_results),
        "reverse_open": _summarize_reverse(reverse_open_results),
        "reverse_restricted": _summarize_reverse(reverse_restricted_results),
    }
    _write_jsonl(output_dir / "forward_predictions.jsonl", forward_results)
    _write_jsonl(output_dir / "reverse_open_predictions.jsonl", reverse_open_results)
    _write_jsonl(output_dir / "reverse_restricted_predictions.jsonl", reverse_restricted_results)
    _write_json(output_dir / "metrics.json", metrics)
    return output_dir


def _run_generation_eval(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    relations: list[dict[str, Any]],
    template: str,
    mode: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict[str, Any]]:
    results = []
    total_batches = (len(relations) + batch_size - 1) // batch_size
    for batch in tqdm(
        _batches(relations, batch_size),
        desc=f"eval {mode}",
        total=total_batches,
        unit="batch",
    ):
        examples = []
        prompts = []
        for relation in batch:
            recipe = [item["surface"] for item in relation["recipe"]]
            prompt, gold = _prompt_and_gold(template, relation, recipe, mode)
            examples.append((relation, recipe, prompt, gold))
            prompts.append(prompt)

        predictions = _generate_batch(model, tokenizer, device, prompts, max_new_tokens)
        for (relation, recipe, prompt, gold), prediction in zip(examples, predictions, strict=True):
            result = {
                "effect_id": relation["effect_id"],
                "partition": relation["partition"],
                "prompt": prompt,
                "gold": gold,
                "prediction": prediction,
                "normalized_gold": _normalize(gold),
                "normalized_prediction": _normalize(prediction),
                "recipe": recipe,
            }
            if mode == "forward":
                result.update(_forward_scores(prediction, gold))
            else:
                result.update(_reverse_scores(prediction, recipe))
            results.append(result)
    return results


def _prompt_and_gold(
    template: str,
    relation: dict[str, Any],
    recipe: list[str],
    mode: str,
) -> tuple[str, str]:
    rendered = render_template(
        template,
        causes=recipe,
        effect=relation["effect_surface"],
    )
    if mode == "forward":
        gold = relation["effect_surface"]
    elif mode == "reverse":
        gold = ", ".join(recipe)
    else:
        raise ValueError(f"Unsupported eval mode: {mode}")
    prompt = _split_prompt(rendered, gold)
    return prompt, gold


def _split_prompt(rendered: str, gold: str) -> str:
    index = rendered.rfind(gold)
    if index < 0:
        raise ValueError(f"Gold answer is not present in rendered template: {gold!r}")
    # For BPE tokenizers, the first answer token often includes the leading
    # space, e.g. " silver". Stop before that whitespace so generation starts
    # at the same token boundary seen during LM training.
    return rendered[:index].rstrip()


def _generate_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    max_new_tokens: int,
) -> list[str]:
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    prompt_length = inputs["input_ids"].shape[1]
    texts = tokenizer.batch_decode(output[:, prompt_length:], skip_special_tokens=True)
    return [_truncate_generation(text) for text in texts]


def _truncate_generation(text: str) -> str:
    candidates = [len(text)]
    for marker in ("\n", "."):
        index = text.find(marker)
        if index >= 0:
            candidates.append(index)
    return text[: min(candidates)].strip()


def _forward_scores(prediction: str, gold: str) -> dict[str, bool]:
    pred = _normalize(prediction)
    target = _normalize(gold)
    return {
        "exact": pred == target,
        "contains": target in pred,
    }


def _reverse_scores(prediction: str, recipe: list[str]) -> dict[str, bool]:
    pred = _normalize(prediction)
    normalized_recipe = [_normalize(item) for item in recipe]
    predicted_causes = _extract_causes(prediction)
    return {
        "ordered_exact": pred == _normalize(", ".join(recipe)),
        "set_exact": set(predicted_causes) == set(normalized_recipe),
        "all_causes_contained": all(cause in pred for cause in normalized_recipe),
    }


def _extract_causes(prediction: str) -> list[str]:
    pieces = re.split(r"[,;/]|\band\b", prediction)
    normalized = [_normalize(piece) for piece in pieces]
    return [piece for piece in normalized if piece]


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[\s]+", " ", text)
    text = text.strip(" .,:;!?\"'`")
    return text


def _summarize_forward(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _base_summary(results, ["exact", "contains"])
    summary["by_partition"] = _partition_summary(results, ["exact", "contains"])
    return summary


def _summarize_reverse(results: list[dict[str, Any]]) -> dict[str, Any]:
    return _base_summary(results, ["ordered_exact", "set_exact", "all_causes_contained"])


def _base_summary(results: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    total = len(results)
    summary: dict[str, Any] = {"num_examples": total}
    for key in keys:
        summary[key] = sum(1 for result in results if result[key]) / max(total, 1)
    return summary


def _partition_summary(results: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["partition"]].append(result)
    return {partition: _base_summary(items, keys) for partition, items in sorted(grouped.items())}


def _select_template(templates: dict[str, Any], type_id: str, *, fallback_type: str) -> str:
    for template_type in templates.get("template_types", []):
        if template_type.get("type_id") == type_id and template_type.get("variants"):
            return str(template_type["variants"][0]["text"])
    fallback = TEMPLATE_TYPES[fallback_type]["variants"][0]
    return str(fallback)


def _resolve_world_dir(pretrain_dir: Path, world_path: str) -> Path:
    candidate = Path(world_path)
    if candidate.exists():
        return candidate
    if candidate.is_absolute():
        return candidate
    joined = (pretrain_dir / candidate).resolve()
    if joined.exists():
        return joined
    return candidate


def _sample(items: list[dict[str, Any]], max_items: int | None, seed: int) -> list[dict[str, Any]]:
    if max_items is None or max_items >= len(items):
        return items
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(items)), max_items))
    return [items[index] for index in indices]


def _batches(items: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
