from __future__ import annotations

import json
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from safe_pretrain.synthetic.composition import CompositionGenerator
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
    prompt_renderer = _PromptRenderer.from_pretrain_dir(pretrain_dir, render_manifest)
    seed = int(cfg_dict.get("seed", 42))
    max_examples = _optional_int(cfg_dict.get("max_examples"))
    max_per_partition = _optional_int(cfg_dict.get("max_per_partition"))
    max_new_tokens = int(cfg_dict.get("max_new_tokens", 24))
    batch_size = int(cfg_dict.get("batch_size", 64))
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    ranking_enabled = _as_bool(cfg_dict.get("ranking_enabled", True))
    ranking_negatives = int(cfg_dict.get("ranking_negatives", 127))
    ranking_batch_size = int(cfg_dict.get("ranking_batch_size", batch_size))
    if ranking_negatives < 0:
        raise ValueError("ranking_negatives must be non-negative")
    if ranking_batch_size <= 0:
        raise ValueError("ranking_batch_size must be positive")

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

    forward_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=forward_relations,
        prompt_renderer=prompt_renderer,
        mode="forward",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    reverse_open_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=open_relations,
        prompt_renderer=prompt_renderer,
        mode="reverse",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    reverse_restricted_results = _run_generation_eval(
        model=model,
        tokenizer=tokenizer,
        device=device,
        relations=restricted_relations,
        prompt_renderer=prompt_renderer,
        mode="reverse",
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    forward_ranking_results = (
        _run_forward_ranking_eval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            relations=forward_relations,
            all_relations=relations,
            prompt_renderer=prompt_renderer,
            num_negatives=ranking_negatives,
            batch_size=ranking_batch_size,
            seed=seed + 10,
        )
        if ranking_enabled
        else []
    )

    metrics = {
        "checkpoint": str(checkpoint_dir),
        "hf_model": str(hf_model_dir),
        "pretrain_dir": str(pretrain_dir),
        "world_dir": str(world_dir),
        "world_id": render_manifest.get("world_id"),
        "render_id": render_manifest.get("render_id"),
        "batch_size": batch_size,
        "ranking_enabled": ranking_enabled,
        "ranking_negatives": ranking_negatives,
        "ranking_batch_size": ranking_batch_size,
        "forward": _summarize_forward(forward_results, relations),
        "forward_ranking": _summarize_ranking(forward_ranking_results),
        "reverse_open": _summarize_reverse(reverse_open_results),
        "reverse_restricted": _summarize_reverse(reverse_restricted_results),
    }
    _write_jsonl(output_dir / "forward_predictions.jsonl", forward_results)
    if ranking_enabled:
        _write_jsonl(output_dir / "forward_ranking.jsonl", forward_ranking_results)
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
    prompt_renderer: "_PromptRenderer",
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
            prompt, gold = prompt_renderer.prompt_and_gold(relation, recipe, mode)
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


def _run_forward_ranking_eval(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    relations: list[dict[str, Any]],
    all_relations: list[dict[str, Any]],
    prompt_renderer: "_PromptRenderer",
    num_negatives: int,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    effects = list(dict.fromkeys(relation["effect_surface"] for relation in all_relations))
    if num_negatives >= len(effects):
        raise ValueError(
            f"ranking_negatives={num_negatives} requires at least {num_negatives + 1} "
            f"unique effects, but only found {len(effects)}"
        )

    examples = []
    for relation in relations:
        recipe = [item["surface"] for item in relation["recipe"]]
        prompt, gold = prompt_renderer.prompt_and_gold(relation, recipe, "forward")
        negatives = _sample_negative_effects(effects, gold, num_negatives, rng)
        candidates = [gold, *negatives]
        examples.append(
            {
                "relation": relation,
                "prompt": prompt,
                "gold": gold,
                "candidates": candidates,
                "scores": [],
            }
        )

    total_candidates = sum(len(example["candidates"]) for example in examples)
    progress = tqdm(total=total_candidates, desc="rank forward", unit="candidate")
    pending_prompts = []
    pending_answers = []
    pending_indices = []
    for example_index, example in enumerate(examples):
        for candidate in example["candidates"]:
            pending_prompts.append(example["prompt"])
            pending_answers.append(candidate)
            pending_indices.append(example_index)
            if len(pending_answers) >= batch_size:
                _flush_ranking_batch(
                    model,
                    tokenizer,
                    device,
                    examples,
                    pending_prompts,
                    pending_answers,
                    pending_indices,
                    progress,
                )
    if pending_answers:
        _flush_ranking_batch(
            model,
            tokenizer,
            device,
            examples,
            pending_prompts,
            pending_answers,
            pending_indices,
            progress,
        )
    progress.close()

    results = []
    for example in examples:
        relation = example["relation"]
        candidates = example["candidates"]
        scores = example["scores"]
        ranked_indices = sorted(range(len(candidates)), key=lambda index: scores[index], reverse=True)
        rank = ranked_indices.index(0) + 1
        top_index = ranked_indices[0]
        results.append(
            {
                "effect_id": relation["effect_id"],
                "partition": relation["partition"],
                "prompt": example["prompt"],
                "gold": example["gold"],
                "gold_score": scores[0],
                "rank": rank,
                "candidate_pool_size": len(candidates),
                "top1": candidates[top_index],
                "top1_score": scores[top_index],
            }
        )
    return results


def _sample_negative_effects(
    effects: list[str],
    gold: str,
    num_negatives: int,
    rng: random.Random,
) -> list[str]:
    negatives = []
    seen = {gold}
    while len(negatives) < num_negatives:
        candidate = rng.choice(effects)
        if candidate in seen:
            continue
        seen.add(candidate)
        negatives.append(candidate)
    return negatives


def _flush_ranking_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    examples: list[dict[str, Any]],
    prompts: list[str],
    answers: list[str],
    example_indices: list[int],
    progress: tqdm,
) -> None:
    scores = _score_answer_batch(model, tokenizer, device, prompts, answers)
    for example_index, score in zip(example_indices, scores, strict=True):
        examples[example_index]["scores"].append(score)
    progress.update(len(answers))
    prompts.clear()
    answers.clear()
    example_indices.clear()


class _PromptRenderer:
    def __init__(
        self,
        *,
        kind: str,
        forward_template: str | None = None,
        reverse_template: str | None = None,
        generator: CompositionGenerator | None = None,
        world_id: str | None = None,
        render_id: str | None = None,
    ) -> None:
        self.kind = kind
        self.forward_template = forward_template
        self.reverse_template = reverse_template
        self.generator = generator
        self.world_id = world_id
        self.render_id = render_id or "eval"
        self.record_index = 0

    @classmethod
    def from_pretrain_dir(
        cls,
        pretrain_dir: Path,
        render_manifest: dict[str, Any],
    ) -> "_PromptRenderer":
        templates_path = pretrain_dir / "templates.json"
        if templates_path.exists():
            templates = read_json(templates_path)
            return cls(
                kind="templates",
                forward_template=_select_template(
                    templates,
                    "forward",
                    fallback_type="forward",
                ),
                reverse_template=_select_template(
                    templates,
                    "reverse",
                    fallback_type="reverse",
                ),
            )

        composition_path = pretrain_dir / "composition_manifest.json"
        if not composition_path.exists():
            raise FileNotFoundError(
                f"Missing templates.json or composition_manifest.json in {pretrain_dir}"
            )
        composition = read_json(composition_path)
        generator = CompositionGenerator(
            generator_version=str(composition.get("generator_version", "composition_v1")),
            connector_version=str(composition.get("connector_version", "connector_v1")),
            pretrain_wrapper_version=str(
                composition.get("pretrain_wrapper_version") or "pretrain_descriptive_v1"
            ),
            pretrain_cause_order=str(composition.get("pretrain_cause_order") or "canonical"),
            sft_wrapper_version="sft_chat_qa_v1",
            chat_template_id="smollm2_chatml_v1",
        )
        return cls(
            kind="composition",
            generator=generator,
            world_id=str(render_manifest["world_id"]),
            render_id=str(render_manifest.get("render_id", "eval")),
        )

    def prompt_and_gold(
        self,
        relation: dict[str, Any],
        recipe: list[str],
        mode: str,
    ) -> tuple[str, str]:
        if mode not in {"forward", "reverse"}:
            raise ValueError(f"Unsupported eval mode: {mode}")

        if self.kind == "templates":
            if mode == "forward":
                gold = relation["effect_surface"]
            else:
                gold = ", ".join(recipe)
            template = self.forward_template if mode == "forward" else self.reverse_template
            if template is None:
                raise ValueError(f"Missing template for eval mode: {mode}")
            rendered = render_template(
                template,
                causes=recipe,
                effect=relation["effect_surface"],
            )
        else:
            if self.generator is None or self.world_id is None:
                raise ValueError("Composition prompt renderer is not initialized")
            composition = self.generator.compose_pretrain(
                relation,
                direction=mode,
                world_id=self.world_id,
                render_id=self.render_id,
                split="eval",
                record_index=self.record_index,
            )
            self.record_index += 1
            rendered = str(composition.text)
            gold = str(composition.metadata["answer_text"])

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


def _score_answer_batch(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    prompts: list[str],
    answers: list[str],
) -> list[float]:
    tokenizer.padding_side = "right"
    texts = [f"{prompt} {answer}." for prompt, answer in zip(prompts, answers, strict=True)]
    prompt_lengths = [
        len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for prompt in prompts
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = inputs["input_ids"][:, 1:]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    attention = inputs["attention_mask"][:, 1:].bool()
    scores = []
    for index, prompt_length in enumerate(prompt_lengths):
        # target token with original index k is predicted at shifted index k-1.
        # The answer begins at original token index prompt_length.
        start = max(prompt_length - 1, 0)
        mask = torch.zeros_like(attention[index])
        mask[start:] = True
        mask &= attention[index]
        values = token_log_probs[index][mask]
        scores.append(float(values.mean().item()))
    return scores


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


def _summarize_forward(
    results: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    effect_surfaces = {_normalize(relation["effect_surface"]) for relation in relations}
    for result in results:
        result["valid_effect_prediction"] = result["normalized_prediction"] in effect_surfaces
        result["wrong_but_valid_effect"] = (
            not result["exact"] and result["valid_effect_prediction"]
        )
    keys = ["exact", "contains", "valid_effect_prediction", "wrong_but_valid_effect"]
    summary = _base_summary(results, keys)
    summary["by_partition"] = _partition_summary(results, keys)
    return summary


def _summarize_reverse(results: list[dict[str, Any]]) -> dict[str, Any]:
    return _base_summary(results, ["ordered_exact", "set_exact", "all_causes_contained"])


def _summarize_ranking(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {"num_examples": 0}
    summary = _ranking_summary(results)
    summary["by_partition"] = {
        partition: _ranking_summary(items)
        for partition, items in sorted(_group_by_partition(results).items())
    }
    return summary


def _ranking_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [int(result["rank"]) for result in results]
    ranks_sorted = sorted(ranks)
    total = len(ranks)
    return {
        "num_examples": total,
        "candidate_pool_size": int(results[0]["candidate_pool_size"]) if results else 0,
        "top1": sum(rank == 1 for rank in ranks) / max(total, 1),
        "top5": sum(rank <= 5 for rank in ranks) / max(total, 1),
        "top10": sum(rank <= 10 for rank in ranks) / max(total, 1),
        "mean_rank": sum(ranks) / max(total, 1),
        "median_rank": statistics.median(ranks_sorted) if ranks_sorted else 0,
    }


def _base_summary(results: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    total = len(results)
    summary: dict[str, Any] = {"num_examples": total}
    for key in keys:
        summary[key] = sum(1 for result in results if result[key]) / max(total, 1)
    return summary


def _partition_summary(results: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    return {
        partition: _base_summary(items, keys)
        for partition, items in sorted(_group_by_partition(results).items())
    }


def _group_by_partition(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["partition"]].append(result)
    return grouped


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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
