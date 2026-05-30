from __future__ import annotations

import math
import random
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm

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

    generator = CompositionGenerator(
        generator_version=str(composition_cfg.get("generator_version", "composition_v1")),
        connector_version=str(composition_cfg.get("connector_version", "connector_v1")),
        pretrain_wrapper_version=str(
            composition_cfg.get("pretrain_wrapper_version", "pretrain_descriptive_v1")
        ),
        sft_wrapper_version=str(composition_cfg.get("sft_wrapper_version", "sft_chat_qa_v1")),
        chat_template_id=str(composition_cfg.get("chat_template_id", "smollm2_chatml_v1")),
    )
    composition_manifest = generator.manifest(include_pretrain=True, include_sft=False)
    render_id = _render_id(cfg_dict, world["manifest"]["world_id"], composition_manifest)
    seed = int(render_cfg["seed"])
    total_records = _target_records(pretrain_cfg)
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
    audit = stats.audit(total_records=total_records, world_id=world["manifest"]["world_id"])
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
        "partition_counts": audit["partition_counts"],
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
        self.effect_counts: dict[str, int] = defaultdict(int)
        self.effect_direction_counts: dict[tuple[str, str], int] = defaultdict(int)
        self.template_keys_by_relation: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.reverse_effect_ids: set[str] = set()
        self.restricted_reverse_records = 0
        self.metadata_id_in_text = 0
        self.duplicate_template_keys = 0

    def add(self, row: dict[str, Any]) -> None:
        metadata = row["metadata"]
        self.counts["total"] += 1
        self.counts[metadata["split"]] += 1
        self.partitions[metadata["partition"]] += 1
        self.directions[metadata["direction"]] += 1
        self.connectors[metadata["connector_id"]] += 1
        self.wrappers[metadata["wrapper_id"]] += 1
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

    def audit(self, *, total_records: int, world_id: str) -> dict[str, Any]:
        exposure_values = list(self.effect_counts.values())
        exposure_direction_values = list(self.effect_direction_counts.values())
        return {
            "world_id": world_id,
            "counts": dict(self.counts),
            "expected_total_records": total_records,
            "partition_counts": dict(self.partitions),
            "direction_counts": dict(self.directions),
            "connector_counts": dict(self.connectors),
            "wrapper_counts": dict(self.wrappers),
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
    forbidden = [
        metadata["effect_id"],
        metadata["partition"],
        metadata.get("world_id", ""),
        metadata.get("render_id", ""),
    ]
    forbidden.extend(metadata.get("cause_ids", []))
    return any(value and str(value) in text for value in forbidden)


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
