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
from safe_pretrain.synthetic.io import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    read_json,
    write_json,
)
from safe_pretrain.synthetic.templates import (
    build_template_inventory,
    render_template,
    template_type_map,
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
    templates_cfg = cfg_dict.get("templates", {})
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

    templates = _build_render_templates(templates_cfg)
    type_map = template_type_map(templates)
    render_id = _render_id(cfg_dict, world["manifest"]["world_id"], templates)
    seed = int(render_cfg["seed"])
    rng = random.Random(seed)
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
    if "forward" not in type_map:
        raise ValueError("Render templates must include type: forward")
    if reverse_ratio > 0 and "reverse" not in type_map:
        raise ValueError("Render templates must include type: reverse when pretrain.reverse_ratio > 0")

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
    write_json(output_path / "templates.json", templates)

    with train_path.open("w", encoding="utf-8") as train_handle, validation_path.open(
        "w", encoding="utf-8"
    ) as validation_handle:
        for record_index in tqdm(
            range(total_records),
            desc="Rendering pretrain records",
            unit="record",
        ):
            relation = sampled_relations[record_index]
            type_id = "reverse" if record_index in reverse_indices else "forward"
            row = _render_row(
                relation,
                world_id=world["manifest"]["world_id"],
                render_id=render_id,
                record_index=record_index,
                template_type=type_map[type_id],
                rng=rng,
                templates_cfg=templates_cfg,
                include_recipe_metadata=bool(pretrain_cfg.get("include_recipe_metadata", False)),
            )
            split = "validation" if record_index in validation_indices else "train"
            row["split"] = split
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
            "templates": "templates.json",
            "experiment_splits": "experiment_splits.json",
            "train": "pretrain_train.jsonl",
            "validation": "pretrain_validation.jsonl",
            "audit": "audit_render.json",
        },
        "counts": audit["counts"],
        "template_type_counts": audit["template_type_counts"],
        "partition_counts": audit["partition_counts"],
    }
    write_json(output_path / "render_manifest.json", manifest)
    write_json(output_path / "checksums.json", _checksums(output_path, manifest["files"].values()))
    return output_path


class _Stats:
    def __init__(self) -> None:
        self.counts = Counter()
        self.partitions = Counter()
        self.template_types = Counter()
        self.modes = Counter()
        self.effect_counts: dict[str, int] = defaultdict(int)
        self.reverse_effect_ids: set[str] = set()
        self.restricted_reverse_records = 0
        self.record_id_in_text = 0

    def add(self, row: dict[str, Any]) -> None:
        self.counts["total"] += 1
        self.counts[row["split"]] += 1
        self.partitions[row["partition"]] += 1
        self.template_types[row["template_type"]] += 1
        self.modes[row["mode"]] += 1
        self.effect_counts[row["effect_id"]] += 1
        if row["mode"] == "reverse":
            self.reverse_effect_ids.add(row["effect_id"])
        if row["partition"] == "restricted" and row["mode"] == "reverse":
            self.restricted_reverse_records += 1
        if row.get("record_id") and row["record_id"] in row["text"]:
            self.record_id_in_text += 1

    def audit(self, *, total_records: int, world_id: str) -> dict[str, Any]:
        exposure_values = list(self.effect_counts.values())
        return {
            "world_id": world_id,
            "counts": dict(self.counts),
            "expected_total_records": total_records,
            "partition_counts": dict(self.partitions),
            "template_type_counts": dict(self.template_types),
            "mode_counts": dict(self.modes),
            "effect_exposure": {
                "num_effects_seen": len(exposure_values),
                "min": min(exposure_values) if exposure_values else 0,
                "max": max(exposure_values) if exposure_values else 0,
                "mean": sum(exposure_values) / len(exposure_values) if exposure_values else 0.0,
            },
            "restricted_reverse_records": self.restricted_reverse_records,
            "record_id_in_text_records": self.record_id_in_text,
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


def _build_render_templates(templates_cfg: dict[str, Any]) -> dict[str, Any]:
    enabled = list(templates_cfg.get("enabled_types", []))
    if not enabled:
        enabled = ["forward", "reverse"]
    return build_template_inventory(
        enabled,
        int(templates_cfg.get("num_variants_per_type", 8)),
    )


def _render_id(cfg_dict: dict[str, Any], world_id: str, templates: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {"world_id": world_id, "render_config": cfg_dict, "templates": templates}
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


def _render_row(
    relation: dict[str, Any],
    *,
    world_id: str,
    render_id: str,
    record_index: int,
    template_type: dict[str, Any],
    rng: random.Random,
    templates_cfg: dict[str, Any],
    include_recipe_metadata: bool,
) -> dict[str, Any]:
    if not template_type["allowed_in_pretrain"]:
        raise ValueError(f"Template type is not allowed in pretrain: {template_type['type_id']}")
    recipe = [item["surface"] for item in relation["recipe"]]
    if str(templates_cfg.get("cause_order", "canonical")) == "shuffled":
        recipe = recipe[:]
        rng.shuffle(recipe)
    variant = rng.choice(template_type["variants"])
    record_id = f"R{record_index:012d}"
    text = render_template(
        variant["text"],
        causes=recipe,
        effect=relation["effect_surface"],
        record_id=record_id,
    )
    if bool(templates_cfg.get("include_record_id_in_text", False)):
        text = f"Record {record_id}. {text}"
    row = {
        "text": text,
        "world_id": world_id,
        "render_id": render_id,
        "effect_id": relation["effect_id"],
        "partition": relation["partition"],
        "mode": template_type["mode"],
        "template_type": template_type["type_id"],
        "template_id": variant["template_id"],
        "record_index": record_index,
    }
    if bool(templates_cfg.get("include_record_id_in_metadata", True)):
        row["record_id"] = record_id
    if include_recipe_metadata:
        row["recipe"] = recipe
    return row


def _assert_render_audit(audit: dict[str, Any], audit_cfg: dict[str, Any]) -> None:
    if audit["counts"].get("total", 0) != audit["expected_total_records"]:
        raise AssertionError("Rendered record count does not match expected total")
    if bool(audit_cfg.get("assert_no_restricted_reverse_in_pretrain", True)):
        if audit["restricted_reverse_records"] != 0:
            raise AssertionError("Restricted reverse records were rendered")
    if bool(audit_cfg.get("assert_no_record_id_in_text", True)):
        if audit["record_id_in_text_records"] != 0:
            raise AssertionError("record_id appeared in rendered text")


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
