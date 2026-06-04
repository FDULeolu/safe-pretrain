from __future__ import annotations

import math
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from safe_pretrain.config import save_config
from safe_pretrain.synthetic.composition import CompositionGenerator
from safe_pretrain.synthetic.io import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    read_json,
    write_json,
)


def render_pretrain_dataset(
    cfg: Any,
    world_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    render_cfg = cfg_dict["render"]
    world_cfg = cfg_dict["world"]
    pretrain_cfg = cfg_dict.get("pretrain", {})
    composition_cfg = cfg_dict.get("composition", {})
    audit_cfg = cfg_dict.get("audit", {})

    world_root = Path(world_path or world_cfg["path"])
    output_path = Path(output_dir or render_cfg["output_dir"])
    overwrite = bool(render_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Set overwrite=true to rebuild it.")

    world = _load_world(world_root)
    expected_world_id = world_cfg.get("expected_world_id")
    if expected_world_id and expected_world_id != world["manifest"]["world_id"]:
        raise ValueError(
            f"World id mismatch: expected {expected_world_id}, got {world['manifest']['world_id']}"
    )

    generator = _build_composition_generator(composition_cfg)
    composition_manifest = generator.manifest(include_pretrain=True, include_sft=False)
    render_id = _render_id(cfg_dict, world["manifest"]["world_id"], composition_manifest)
    seed = int(render_cfg["seed"])
    train_fraction = float(pretrain_cfg.get("train_fraction", 0.99))
    if not 0 < train_fraction < 1:
        raise ValueError("pretrain.train_fraction must be between 0 and 1")
    validation_fraction = pretrain_cfg.get("validation_fraction")
    if validation_fraction is not None:
        validation_fraction = float(validation_fraction)
        if not 0 < validation_fraction < 1:
            raise ValueError("pretrain.validation_fraction must be between 0 and 1")
        if not math.isclose(train_fraction + validation_fraction, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("pretrain.train_fraction + pretrain.validation_fraction must equal 1.0")

    reverse_ratio = float(pretrain_cfg.get("reverse_ratio", 0.0))
    if not 0 <= reverse_ratio <= 1:
        raise ValueError("pretrain.reverse_ratio must be in [0, 1]")

    relations = world["relations"]
    open_relations = [relation for relation in relations if relation["partition"] == "open"]
    restricted_relations = [
        relation for relation in relations if relation["partition"] == "restricted"
    ]
    if not relations:
        raise ValueError("World has no relations")
    open_ratio = len(open_relations) / len(relations)
    if reverse_ratio > open_ratio:
        raise ValueError(
            "pretrain.reverse_ratio cannot exceed the world's open relation ratio "
            f"({reverse_ratio:.6f} > {open_ratio:.6f})"
        )
    if reverse_ratio > 0 and not open_relations:
        raise ValueError("pretrain.reverse_ratio > 0 but the world has no open effects")

    token_budget = _token_budget_config(pretrain_cfg)
    token_budget_audit: dict[str, Any] = {"enabled": False}
    total_records: int | None
    if token_budget and token_budget["strategy"] == "estimate_records":
        total_records, token_budget_audit = _estimate_records_for_token_budget(
            token_budget=token_budget,
            composition_cfg=composition_cfg,
            open_relations=open_relations,
            restricted_relations=restricted_relations,
            reverse_ratio=reverse_ratio,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    elif token_budget and token_budget["strategy"] == "exact_stream":
        total_records = None
        token_budget_audit = {
            "enabled": True,
            "strategy": "exact_stream",
            "target_tokens": int(token_budget["target_tokens"]),
            "target_split": token_budget["target_split"],
            "tokenizer": token_budget["tokenizer_name_or_path"],
            "append_eos": bool(token_budget["append_eos"]),
            "count_batch_size": int(token_budget["batch_size"]),
        }
        _assert_token_budget_capacity_estimate(
            token_budget=token_budget,
            composition_cfg=composition_cfg,
            open_relations=open_relations,
            restricted_relations=restricted_relations,
            reverse_ratio=reverse_ratio,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
            seed=seed,
        )
    else:
        total_records = _target_records(pretrain_cfg)

    if total_records is not None:
        _assert_record_capacity(
            total_records,
            open_relations=open_relations,
            restricted_relations=restricted_relations,
            reverse_ratio=reverse_ratio,
            key_space_size=generator.pretrain_key_space_size,
        )

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    train_path = output_path / "pretrain_train.jsonl"
    validation_path = output_path / "pretrain_validation.jsonl"
    stats = _Stats()
    save_config(cfg, output_path / "render_config.yaml")
    write_json(output_path / "composition_manifest.json", composition_manifest)

    with train_path.open("w", encoding="utf-8") as train_handle, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        if token_budget and token_budget["strategy"] == "exact_stream":
            _render_token_budget_records(
                token_budget=token_budget,
                open_relations=open_relations,
                restricted_relations=restricted_relations,
                generator=generator,
                world_id=world["manifest"]["world_id"],
                render_id=render_id,
                reverse_ratio=reverse_ratio,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                seed=seed,
                train_handle=train_handle,
                validation_handle=validation_handle,
                stats=stats,
            )
        else:
            if total_records is None:
                raise AssertionError("total_records should be resolved before fixed-record rendering")
            validation_indices = _validation_indices(
                total_records,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
                seed=seed + 1,
            )
            sampled_relations, open_positions = _partitioned_relation_sequence(
                open_relations,
                restricted_relations,
                total_records,
                seed=seed + 2,
            )
            reverse_indices = _reverse_indices_from_open_positions(
                open_positions,
                total_records=total_records,
                reverse_ratio=reverse_ratio,
                seed=seed + 5,
            )
            for record_index in tqdm(
                range(total_records),
                desc="Rendering pretrain records",
                unit="record",
            ):
                relation = sampled_relations[record_index]
                direction = "reverse" if record_index in reverse_indices else "forward"
                split = "validation" if record_index in validation_indices else "train"
                composition = generator.compose_pretrain(
                    relation,
                    world_id=world["manifest"]["world_id"],
                    render_id=render_id,
                    record_index=record_index,
                    direction=direction,
                    split=split,
                )
                row = {"text": composition.text, "metadata": composition.metadata}
                handle = train_handle if split == "train" else validation_handle
                handle.write(_json_line(row))
                stats.add(row)

    experiment_splits = _build_experiment_splits(relations, stats.reverse_effect_ids)
    write_json(output_path / "experiment_splits.json", experiment_splits)
    expected_total_records = (
        stats.counts.get("total", 0)
        if token_budget and token_budget["strategy"] == "exact_stream"
        else total_records
    )
    audit = stats.audit(
        expected_total_records=expected_total_records,
        world_id=world["manifest"]["world_id"],
    )
    if token_budget and token_budget["strategy"] == "exact_stream":
        target_split = token_budget["target_split"]
        achieved = stats.token_counts[_token_count_key(target_split)]
        audit["token_budget"] = {
            "enabled": True,
            "strategy": "exact_stream",
            "target_tokens": int(token_budget["target_tokens"]),
            "target_split": target_split,
            "achieved_tokens": achieved,
            "overrun_tokens": achieved - int(token_budget["target_tokens"]),
            "tokenizer": token_budget["tokenizer_name_or_path"],
            "append_eos": bool(token_budget["append_eos"]),
            "count_batch_size": int(token_budget["batch_size"]),
        }
    elif token_budget:
        audit["token_budget"] = token_budget_audit
    audit["experiment_split_counts"] = {
        group: {name: len(ids) for name, ids in values.items()}
        for group, values in experiment_splits.items()
    }
    _assert_render_audit(audit, audit_cfg)
    write_json(output_path / "audit_render.json", audit)

    manifest = {
        "render_name": str(render_cfg.get("name", output_path.name)),
        "render_id": render_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "world_path": str(world_root),
        "world_id": world["manifest"]["world_id"],
        "config_hash": canonical_json_sha256(cfg_dict),
        "files": {
            "render_config": "render_config.yaml",
            "composition_manifest": "composition_manifest.json",
            "experiment_splits": "experiment_splits.json",
            "train": "pretrain_train.jsonl",
            "validation": "pretrain_validation.jsonl",
            "audit": "audit_render.json",
        },
        "counts": audit["counts"],
        "direction_counts": audit["direction_counts"],
        "connector_counts": audit["connector_counts"],
        "wrapper_counts": audit["wrapper_counts"],
        "rendered_cause_order_counts": audit["rendered_cause_order_counts"],
        "rendered_cause_order_policy_counts": audit["rendered_cause_order_policy_counts"],
        "partition_counts": audit["partition_counts"],
        "token_counts": audit.get("token_counts", {}),
        "token_budget": audit.get("token_budget", {"enabled": False}),
    }
    write_json(output_path / "render_manifest.json", manifest)
    write_json(output_path / "checksums.json", _checksums(output_path, manifest["files"].values()))
    return output_path


class _Stats:
    def __init__(self) -> None:
        self.counts = Counter()
        self.partitions = Counter()
        self.directions = Counter()
        self.connectors = Counter()
        self.wrappers = Counter()
        self.rendered_cause_orders = Counter()
        self.rendered_cause_order_policies = Counter()
        self.effect_counts: dict[str, int] = defaultdict(int)
        self.effect_direction_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.template_keys_by_relation: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.reverse_effect_ids: set[str] = set()
        self.restricted_reverse_records = 0
        self.metadata_id_in_text = 0
        self.duplicate_template_keys = 0
        self.token_counts = Counter()

    def add(self, row: dict[str, Any], *, token_count: int | None = None) -> None:
        metadata = row["metadata"]
        self.counts["total"] += 1
        self.counts[metadata["split"]] += 1
        self.partitions[metadata["partition"]] += 1
        self.directions[metadata["direction"]] += 1
        self.connectors[metadata["connector_id"]] += 1
        self.wrappers[metadata["wrapper_id"]] += 1
        self.rendered_cause_orders[metadata.get("rendered_cause_order", "canonical")] += 1
        self.rendered_cause_order_policies[
            metadata.get("rendered_cause_order_policy", "canonical")
        ] += 1
        self.effect_counts[metadata["effect_id"]] += 1
        self.effect_direction_counts[(metadata["effect_id"], metadata["direction"])] += 1
        template_key = (metadata["direction"], metadata["template_key_hash"])
        relation_keys = self.template_keys_by_relation[metadata["relation_id"]]
        if template_key in relation_keys:
            self.duplicate_template_keys += 1
        else:
            relation_keys.add(template_key)
        if metadata["direction"] == "reverse":
            self.reverse_effect_ids.add(metadata["effect_id"])
        if metadata["partition"] == "restricted" and metadata["direction"] == "reverse":
            self.restricted_reverse_records += 1
        if _metadata_visible_in_text(row["text"], metadata):
            self.metadata_id_in_text += 1
        if token_count is not None:
            self.token_counts["total"] += token_count
            self.token_counts[metadata["split"]] += token_count

    def audit(self, *, expected_total_records: int, world_id: str) -> dict[str, Any]:
        exposure_values = list(self.effect_counts.values())
        exposure_direction_values = list(self.effect_direction_counts.values())
        return {
            "world_id": world_id,
            "counts": dict(self.counts),
            "expected_total_records": expected_total_records,
            "token_counts": dict(self.token_counts),
            "partition_counts": dict(self.partitions),
            "direction_counts": dict(self.directions),
            "connector_counts": dict(self.connectors),
            "wrapper_counts": dict(self.wrappers),
            "rendered_cause_order_counts": dict(self.rendered_cause_orders),
            "rendered_cause_order_policy_counts": dict(self.rendered_cause_order_policies),
            "effect_exposure": {
                "num_effects_seen": len(exposure_values),
                "min": min(exposure_values) if exposure_values else 0,
                "max": max(exposure_values) if exposure_values else 0,
                "mean": sum(exposure_values) / len(exposure_values) if exposure_values else 0.0,
            },
            "effect_direction_exposure": {
                "min": min(exposure_direction_values) if exposure_direction_values else 0,
                "max": max(exposure_direction_values) if exposure_direction_values else 0,
                "mean": (
                    sum(exposure_direction_values) / len(exposure_direction_values)
                    if exposure_direction_values
                    else 0.0
                ),
            },
            "restricted_reverse_records": self.restricted_reverse_records,
            "metadata_id_in_text_records": self.metadata_id_in_text,
            "duplicate_template_keys_per_relation": self.duplicate_template_keys,
        }


def _load_world(world_root: Path) -> dict[str, Any]:
    manifest = read_json(world_root / "world_manifest.json")
    return {
        "root": world_root,
        "manifest": manifest,
        "relations": list(iter_jsonl(world_root / "relations.jsonl")),
        "splits": read_json(world_root / "splits.json"),
    }


def _build_composition_generator(composition_cfg: dict[str, Any]) -> CompositionGenerator:
    return CompositionGenerator(
        generator_version=str(composition_cfg.get("generator_version", "composition_v1")),
        connector_version=str(composition_cfg.get("connector_version", "connector_v1")),
        pretrain_wrapper_version=str(
            composition_cfg.get("pretrain_wrapper_version", "pretrain_descriptive_v1")
        ),
        pretrain_cause_order=str(composition_cfg.get("pretrain_cause_order", "canonical")),
        sft_wrapper_version=str(composition_cfg.get("sft_wrapper_version", "sft_chat_qa_v1")),
        chat_template_id=str(composition_cfg.get("chat_template_id", "smollm2_chatml_v1")),
    )


def _build_experiment_splits(
    relations: list[dict[str, Any]],
    reverse_effect_ids: set[str],
) -> dict[str, Any]:
    open_ids = [relation["effect_id"] for relation in relations if relation["partition"] == "open"]
    open_set = set(open_ids)
    reverse_seen = sorted(reverse_effect_ids)
    if not set(reverse_seen).issubset(open_set):
        raise AssertionError("Reverse records included non-open effects")
    return {
        "pretrain": {
            "open_reverse_seen": reverse_seen,
            "open_reverse_heldout": sorted(open_set - set(reverse_seen)),
        }
    }


def _render_id(cfg_dict: dict[str, Any], world_id: str, composition: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {"world_id": world_id, "render_config": cfg_dict, "composition": composition}
    )[:16]


def _target_records(pretrain_cfg: dict[str, Any]) -> int:
    target_records = pretrain_cfg.get("target_records")
    if target_records is not None:
        value = int(target_records)
        if value <= 0:
            raise ValueError("pretrain.target_records must be positive")
        return value
    target_tokens = pretrain_cfg.get("target_tokens")
    estimated = int(pretrain_cfg.get("estimated_tokens_per_record", 24))
    if target_tokens is None:
        raise ValueError("Set pretrain.target_records or pretrain.target_tokens")
    return max(1, math.ceil(int(target_tokens) / estimated))


def _token_budget_config(pretrain_cfg: dict[str, Any]) -> dict[str, Any] | None:
    cfg = pretrain_cfg.get("token_budgeting") or {}
    if not bool(cfg.get("enabled", False)):
        return None
    target_tokens = pretrain_cfg.get("target_tokens")
    if target_tokens is None:
        raise ValueError("pretrain.target_tokens is required when token_budgeting.enabled=true")
    target_tokens = int(target_tokens)
    if target_tokens <= 0:
        raise ValueError("pretrain.target_tokens must be positive")

    tokenizer_name = cfg.get("tokenizer_name_or_path") or pretrain_cfg.get("tokenizer_name_or_path")
    if tokenizer_name is None or str(tokenizer_name).lower() in {"", "none", "null"}:
        raise ValueError(
            "pretrain.token_budgeting.tokenizer_name_or_path is required when "
            "token_budgeting.enabled=true"
        )

    target_split = str(cfg.get("target_split", "train"))
    if target_split not in {"train", "validation", "total", "all"}:
        raise ValueError("pretrain.token_budgeting.target_split must be train, validation, total, or all")
    strategy = str(cfg.get("strategy", "estimate_records"))
    if strategy not in {"estimate_records", "exact_stream"}:
        raise ValueError(
            "pretrain.token_budgeting.strategy must be estimate_records or exact_stream"
        )
    batch_size = int(cfg.get("count_batch_size", cfg.get("batch_size", 4096)))
    if batch_size <= 0:
        raise ValueError("pretrain.token_budgeting.count_batch_size must be positive")
    estimate_sample_records = int(cfg.get("estimate_sample_records", 200000))
    if estimate_sample_records <= 0:
        raise ValueError("pretrain.token_budgeting.estimate_sample_records must be positive")
    record_safety_margin = float(cfg.get("record_safety_margin", 1.01))
    if record_safety_margin <= 0:
        raise ValueError("pretrain.token_budgeting.record_safety_margin must be positive")
    return {
        "target_tokens": target_tokens,
        "target_split": target_split,
        "strategy": strategy,
        "tokenizer_name_or_path": str(tokenizer_name),
        "append_eos": bool(cfg.get("append_eos", True)),
        "batch_size": batch_size,
        "estimate_sample_records": estimate_sample_records,
        "record_safety_margin": record_safety_margin,
        "trust_remote_code": bool(cfg.get("trust_remote_code", False)),
    }


def _estimate_records_for_token_budget(
    *,
    token_budget: dict[str, Any],
    composition_cfg: dict[str, Any],
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    reverse_ratio: float,
    train_fraction: float,
    validation_fraction: float | None,
    seed: int,
) -> tuple[int, dict[str, Any]]:
    estimate = _sample_token_budget(
        token_budget=token_budget,
        composition_cfg=composition_cfg,
        open_relations=open_relations,
        restricted_relations=restricted_relations,
        reverse_ratio=reverse_ratio,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    target_key = _token_count_key(str(token_budget["target_split"]))
    tokens_per_total_record = float(estimate["token_counts"][target_key]) / float(
        estimate["sample_records"]
    )
    if tokens_per_total_record <= 0:
        raise ValueError(
            "Token budget estimate produced zero target-split tokens per record; "
            f"target_split={token_budget['target_split']}"
        )
    estimated_records = math.ceil(
        int(token_budget["target_tokens"])
        / tokens_per_total_record
        * float(token_budget["record_safety_margin"])
    )
    max_records = _max_records_without_template_reuse(
        open_relations=open_relations,
        restricted_relations=restricted_relations,
        reverse_ratio=reverse_ratio,
        key_space_size=_build_composition_generator(composition_cfg).pretrain_key_space_size,
    )
    estimated_max_tokens = math.floor(max_records * tokens_per_total_record)
    if estimated_records > max_records:
        raise ValueError(
            "Requested pretrain token budget is not reachable without repeating per-relation "
            "pretrain templates. "
            f"target_split={token_budget['target_split']}, "
            f"target_tokens={int(token_budget['target_tokens'])}, "
            f"estimated_max_tokens={estimated_max_tokens}, "
            f"estimated_required_records={estimated_records}, "
            f"max_records_without_template_reuse={max_records}. "
            "Increase pretrain template key space, allow template reuse, or lower target_tokens."
        )
    audit = {
        "enabled": True,
        "strategy": "estimate_records",
        "target_tokens": int(token_budget["target_tokens"]),
        "target_split": token_budget["target_split"],
        "tokenizer": token_budget["tokenizer_name_or_path"],
        "append_eos": bool(token_budget["append_eos"]),
        "estimate_sample_records": int(estimate["sample_records"]),
        "record_safety_margin": float(token_budget["record_safety_margin"]),
        "sample_token_counts": estimate["token_counts"],
        "tokens_per_total_record": tokens_per_total_record,
        "estimated_total_records": estimated_records,
        "max_records_without_template_reuse": max_records,
        "estimated_max_tokens_without_template_reuse": estimated_max_tokens,
    }
    return estimated_records, audit


def _assert_token_budget_capacity_estimate(
    *,
    token_budget: dict[str, Any],
    composition_cfg: dict[str, Any],
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    reverse_ratio: float,
    train_fraction: float,
    validation_fraction: float | None,
    seed: int,
) -> None:
    _estimate_records_for_token_budget(
        token_budget={**token_budget, "strategy": "estimate_records", "record_safety_margin": 1.0},
        composition_cfg=composition_cfg,
        open_relations=open_relations,
        restricted_relations=restricted_relations,
        reverse_ratio=reverse_ratio,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        seed=seed,
    )


def _sample_token_budget(
    *,
    token_budget: dict[str, Any],
    composition_cfg: dict[str, Any],
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    reverse_ratio: float,
    train_fraction: float,
    validation_fraction: float | None,
    seed: int,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        token_budget["tokenizer_name_or_path"],
        trust_remote_code=bool(token_budget["trust_remote_code"]),
    )
    generator = _build_composition_generator(composition_cfg)
    relation_sampler = _PretrainRelationDirectionSampler(
        open_relations,
        restricted_relations,
        reverse_ratio=reverse_ratio,
        seed=seed + 2,
    )
    split_sampler = _WeightedCycleSampler(
        [
            ("train", train_fraction),
            ("validation", 1.0 - train_fraction if validation_fraction is None else validation_fraction),
        ],
        random.Random(seed + 1),
    )
    sample_records = int(token_budget["estimate_sample_records"])
    batch_size = int(token_budget["batch_size"])
    token_counts = Counter()
    record_index = 0
    remaining = sample_records
    while remaining > 0:
        current_batch_size = min(batch_size, remaining)
        batch: list[dict[str, Any]] = []
        for _ in range(current_batch_size):
            relation, direction = relation_sampler.next()
            split = split_sampler.next()
            composition = generator.compose_pretrain(
                relation,
                world_id="token-budget-estimate-world",
                render_id="token-budget-estimate-render",
                record_index=record_index,
                direction=direction,
                split=split,
            )
            batch.append({"text": composition.text, "metadata": composition.metadata})
            record_index += 1
        token_lengths = _token_lengths(
            tokenizer,
            [row["text"] for row in batch],
            append_eos=bool(token_budget["append_eos"]),
        )
        for row, token_count in zip(batch, token_lengths, strict=True):
            token_counts["total"] += token_count
            token_counts[row["metadata"]["split"]] += token_count
        remaining -= current_batch_size
    return {"sample_records": sample_records, "token_counts": dict(token_counts)}


def _render_token_budget_records(
    *,
    token_budget: dict[str, Any],
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    generator: CompositionGenerator,
    world_id: str,
    render_id: str,
    reverse_ratio: float,
    train_fraction: float,
    validation_fraction: float | None,
    seed: int,
    train_handle: Any,
    validation_handle: Any,
    stats: _Stats,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        token_budget["tokenizer_name_or_path"],
        trust_remote_code=bool(token_budget["trust_remote_code"]),
    )
    relation_sampler = _PretrainRelationDirectionSampler(
        open_relations,
        restricted_relations,
        reverse_ratio=reverse_ratio,
        seed=seed + 2,
    )
    split_sampler = _WeightedCycleSampler(
        [
            ("train", train_fraction),
            ("validation", 1.0 - train_fraction if validation_fraction is None else validation_fraction),
        ],
        random.Random(seed + 1),
    )
    target_key = _token_count_key(str(token_budget["target_split"]))
    target_tokens = int(token_budget["target_tokens"])
    batch_size = int(token_budget["batch_size"])
    record_index = 0
    progress = tqdm(desc="Rendering pretrain records", unit="record")
    try:
        while stats.token_counts[target_key] < target_tokens:
            batch: list[dict[str, Any]] = []
            for _ in range(batch_size):
                relation, direction = relation_sampler.next()
                split = split_sampler.next()
                composition = generator.compose_pretrain(
                    relation,
                    world_id=world_id,
                    render_id=render_id,
                    record_index=record_index,
                    direction=direction,
                    split=split,
                )
                batch.append({"text": composition.text, "metadata": composition.metadata})
                record_index += 1

            token_lengths = _token_lengths(
                tokenizer,
                [row["text"] for row in batch],
                append_eos=bool(token_budget["append_eos"]),
            )
            for row, token_count in zip(batch, token_lengths, strict=True):
                handle = train_handle if row["metadata"]["split"] == "train" else validation_handle
                handle.write(_json_line(row))
                stats.add(row, token_count=token_count)
                progress.update(1)
                if stats.token_counts[target_key] >= target_tokens:
                    break
    finally:
        progress.close()


def _token_lengths(tokenizer: Any, texts: list[str], *, append_eos: bool) -> list[int]:
    eos = tokenizer.eos_token or ""
    if append_eos:
        texts = [text + eos for text in texts]
    encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False)
    return [len(input_ids) for input_ids in encoded["input_ids"]]


def _token_count_key(target_split: str) -> str:
    return "total" if target_split == "all" else target_split


def _assert_record_capacity(
    total_records: int,
    *,
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    reverse_ratio: float,
    key_space_size: int,
) -> None:
    max_records = _max_records_without_template_reuse(
        open_relations=open_relations,
        restricted_relations=restricted_relations,
        reverse_ratio=reverse_ratio,
        key_space_size=key_space_size,
    )
    if total_records > max_records:
        raise ValueError(
            "Requested pretrain record count would repeat per-relation pretrain templates: "
            f"total_records={total_records}, max_records_without_template_reuse={max_records}. "
            "Increase pretrain template key space, allow template reuse, or lower target_records."
        )


def _max_records_without_template_reuse(
    *,
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    reverse_ratio: float,
    key_space_size: int,
) -> int:
    total_relations = len(open_relations) + len(restricted_relations)
    if total_relations <= 0:
        return 0
    open_ratio = len(open_relations) / total_relations
    restricted_ratio = len(restricted_relations) / total_relations
    category_limits: list[int] = []
    if reverse_ratio > 0:
        category_limits.append(math.floor(len(open_relations) * key_space_size / reverse_ratio))
    open_forward_ratio = open_ratio - reverse_ratio
    if open_forward_ratio > 0:
        category_limits.append(
            math.floor(len(open_relations) * key_space_size / open_forward_ratio)
        )
    if restricted_ratio > 0:
        category_limits.append(
            math.floor(len(restricted_relations) * key_space_size / restricted_ratio)
        )
    if not category_limits:
        return 0
    return min(category_limits)


class _RelationSampler:
    def __init__(self, relations: list[dict[str, Any]], rng: random.Random) -> None:
        if not relations:
            raise ValueError("Cannot sample from an empty relation list")
        self.relations = relations[:]
        self.rng = rng
        self.index = 0
        self.rng.shuffle(self.relations)

    def next(self) -> dict[str, Any]:
        if self.index >= len(self.relations):
            self.rng.shuffle(self.relations)
            self.index = 0
        relation = self.relations[self.index]
        self.index += 1
        return relation


class _WeightedCycleSampler:
    def __init__(self, weighted_items: list[tuple[Any, float]], rng: random.Random) -> None:
        self.rng = rng
        self.base_items = _weighted_cycle_items(weighted_items)
        if not self.base_items:
            raise ValueError("Weighted cycle sampler needs at least one positive-weight item")
        self.items: list[Any] = []
        self.index = 0
        self._reshuffle()

    def next(self) -> Any:
        if self.index >= len(self.items):
            self._reshuffle()
        item = self.items[self.index]
        self.index += 1
        return item

    def _reshuffle(self) -> None:
        self.items = self.base_items[:]
        self.rng.shuffle(self.items)
        self.index = 0


class _PretrainRelationDirectionSampler:
    def __init__(
        self,
        open_relations: list[dict[str, Any]],
        restricted_relations: list[dict[str, Any]],
        *,
        reverse_ratio: float,
        seed: int,
    ) -> None:
        relations = open_relations + restricted_relations
        if not relations:
            raise ValueError("Cannot render pretrain data from an empty relation table")
        open_ratio = len(open_relations) / len(relations)
        if reverse_ratio > open_ratio:
            raise ValueError(
                "pretrain.reverse_ratio cannot exceed the world's open relation ratio "
                f"({reverse_ratio:.6f} > {open_ratio:.6f})"
            )
        categories: list[tuple[str, float]] = []
        self.samplers: dict[str, _RelationSampler] = {}
        open_forward_ratio = open_ratio - reverse_ratio
        restricted_ratio = len(restricted_relations) / len(relations)
        if reverse_ratio > 0:
            self.samplers["open_reverse"] = _RelationSampler(open_relations, random.Random(seed + 1))
            categories.append(("open_reverse", reverse_ratio))
        if open_forward_ratio > 0:
            self.samplers["open_forward"] = _RelationSampler(open_relations, random.Random(seed + 2))
            categories.append(("open_forward", open_forward_ratio))
        if restricted_ratio > 0:
            self.samplers["restricted_forward"] = _RelationSampler(
                restricted_relations,
                random.Random(seed + 3),
            )
            categories.append(("restricted_forward", restricted_ratio))
        self.category_sampler = _WeightedCycleSampler(categories, random.Random(seed))

    def next(self) -> tuple[dict[str, Any], str]:
        category = self.category_sampler.next()
        relation = self.samplers[category].next()
        direction = "reverse" if category == "open_reverse" else "forward"
        return relation, direction


def _weighted_cycle_items(weighted_items: list[tuple[Any, float]]) -> list[Any]:
    fractions: list[tuple[Any, Fraction]] = []
    for item, weight in weighted_items:
        if weight <= 0:
            continue
        fractions.append((item, Fraction(str(weight)).limit_denominator(10000)))
    total = sum((fraction for _, fraction in fractions), start=Fraction(0, 1))
    if total <= 0:
        return []
    normalized = [(item, fraction / total) for item, fraction in fractions]
    denominator = 1
    for _, fraction in normalized:
        denominator = math.lcm(denominator, fraction.denominator)
    counts = [(item, int(fraction * denominator)) for item, fraction in normalized]
    count_total = sum(count for _, count in counts)
    if count_total != denominator:
        largest_index = max(range(len(counts)), key=lambda index: counts[index][1])
        item, count = counts[largest_index]
        counts[largest_index] = (item, count + denominator - count_total)

    items: list[Any] = []
    for item, count in counts:
        if count > 0:
            items.extend([item] * count)
    return items


def _partitioned_relation_sequence(
    open_relations: list[dict[str, Any]],
    restricted_relations: list[dict[str, Any]],
    total_records: int,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    if not open_relations and not restricted_relations:
        raise ValueError("Cannot render pretrain data from an empty relation table")
    if not open_relations:
        sampler = _RelationSampler(restricted_relations, random.Random(seed + 1))
        return [sampler.next() for _ in range(total_records)], set()
    if not restricted_relations:
        sampler = _RelationSampler(open_relations, random.Random(seed + 1))
        return [sampler.next() for _ in range(total_records)], set(range(total_records))

    open_fraction = len(open_relations) / (len(open_relations) + len(restricted_relations))
    if total_records == 1:
        open_count = 1 if open_fraction >= 0.5 else 0
    else:
        open_count = _fraction_count(
            total_records,
            open_fraction,
            min_count=1,
            max_count=total_records - 1,
        )
    rng = random.Random(seed)
    open_positions = set(rng.sample(range(total_records), open_count))
    open_sampler = _RelationSampler(open_relations, random.Random(seed + 1))
    restricted_sampler = _RelationSampler(restricted_relations, random.Random(seed + 2))

    sequence = []
    for record_index in range(total_records):
        if record_index in open_positions:
            sequence.append(open_sampler.next())
        else:
            sequence.append(restricted_sampler.next())
    return sequence, open_positions


def _validation_indices(
    total_records: int,
    *,
    train_fraction: float,
    validation_fraction: float | None,
    seed: int,
) -> set[int]:
    if total_records <= 1:
        return set()
    if validation_fraction is None:
        validation_fraction = 1.0 - train_fraction
    validation_count = _fraction_count(
        total_records,
        validation_fraction,
        min_count=1,
        max_count=total_records - 1,
    )
    rng = random.Random(seed)
    return set(rng.sample(range(total_records), validation_count))


def _reverse_indices_from_open_positions(
    open_positions: set[int],
    *,
    total_records: int,
    reverse_ratio: float,
    seed: int,
) -> set[int]:
    if reverse_ratio <= 0:
        return set()
    reverse_count = _fraction_count(
        total_records,
        reverse_ratio,
        min_count=1,
        max_count=total_records,
    )
    if reverse_count > len(open_positions):
        raise ValueError(
            "pretrain.reverse_ratio requests more reverse records than the base corpus has "
            f"open records ({reverse_count} > {len(open_positions)})"
        )
    rng = random.Random(seed)
    return set(rng.sample(sorted(open_positions), reverse_count))


def _fraction_count(
    total_records: int,
    fraction: float,
    *,
    min_count: int,
    max_count: int,
) -> int:
    if max_count < min_count:
        return max_count
    count = int(round(total_records * fraction))
    return max(min_count, min(max_count, count))


def _assert_render_audit(audit: dict[str, Any], audit_cfg: dict[str, Any]) -> None:
    if audit["counts"].get("total", 0) != audit["expected_total_records"]:
        raise AssertionError("Rendered record count does not match expected total")
    if bool(audit_cfg.get("assert_no_restricted_reverse_in_pretrain", True)):
        if audit["restricted_reverse_records"] != 0:
            raise AssertionError("Restricted reverse records were rendered")
    if bool(audit_cfg.get("assert_no_metadata_id_in_text", True)):
        if audit["metadata_id_in_text_records"] != 0:
            raise AssertionError("metadata id or partition label appeared in rendered text")
    if bool(audit_cfg.get("assert_unique_template_key_per_relation", True)):
        if audit["duplicate_template_keys_per_relation"] != 0:
            raise AssertionError("Duplicate composition template keys were rendered")


def _metadata_visible_in_text(text: str, metadata: dict[str, Any]) -> bool:
    forbidden_ids = [
        metadata["effect_id"],
        metadata.get("world_id", ""),
        metadata.get("render_id", ""),
    ]
    forbidden_ids.extend(metadata.get("cause_ids", []))
    if any(value and str(value) in text for value in forbidden_ids):
        return True

    partition = str(metadata["partition"])
    return re.search(rf"(?<![A-Za-z]){re.escape(partition)}(?![A-Za-z])", text) is not None


def _json_line(row: dict[str, Any]) -> str:
    import json

    return json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"


def _checksums(root: Path, relative_files: Any) -> dict[str, str]:
    checksums = {}
    for relative in relative_files:
        path = root / relative
        if path.exists():
            checksums[str(relative)] = file_sha256(path)
    return checksums
