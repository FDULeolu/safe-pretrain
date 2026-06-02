from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from safe_pretrain.config import save_config
from safe_pretrain.synthetic.composition import CompositionGenerator, chat_template_text
from safe_pretrain.synthetic.io import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    read_json,
    write_json,
)


def render_sft_qa_dataset(
    cfg: Any,
    world_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool | None = None,
) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    sft_cfg = cfg_dict["sft"]
    world_cfg = cfg_dict["world"]
    data_cfg = cfg_dict.get("sft_data", {})
    composition_cfg = cfg_dict.get("composition", {})
    audit_cfg = cfg_dict.get("audit", {})

    world_root = Path(world_path or world_cfg["path"])
    output_path = Path(output_dir or sft_cfg["output_dir"])
    overwrite = bool(sft_cfg.get("overwrite", False) if overwrite is None else overwrite)
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
    composition_manifest = generator.manifest(include_pretrain=False, include_sft=True)
    sft_render_id = _sft_render_id(cfg_dict, world["manifest"]["world_id"], composition_manifest)

    examples_per_relation_per_task = int(data_cfg.get("examples_per_relation_per_task", 1))
    if examples_per_relation_per_task <= 0:
        raise ValueError("sft_data.examples_per_relation_per_task must be positive")

    split_spec = _build_relation_splits(
        world["relations"],
        train_fraction=float(data_cfg.get("train_fraction", 0.8)),
        validation_fraction=float(data_cfg.get("validation_fraction", 0.1)),
        restricted_forward_train_fraction=(
            float(data_cfg["restricted_forward_train_fraction"])
            if data_cfg.get("restricted_forward_train_fraction") is not None
            else None
        ),
        seed=int(sft_cfg["seed"]),
    )
    relation_by_id = {relation["effect_id"]: relation for relation in world["relations"]}

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    save_config(cfg, output_path / "sft_config.yaml")
    write_json(output_path / "composition_manifest.json", composition_manifest)
    write_json(output_path / "sft_splits.json", split_spec)
    (output_path / "chat_template.jinja").write_text(
        chat_template_text(str(composition_cfg.get("chat_template_id", "smollm2_chatml_v1"))),
        encoding="utf-8",
    )

    stats = _Stats()
    files = {
        "train": "sft_train.jsonl",
        "validation": "sft_validation.jsonl",
        "test_safe": "sft_test_safe.jsonl",
        "attack": "eval_attack.jsonl",
    }
    _write_split(
        output_path / files["train"],
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="train",
            examples_per_relation_per_task=examples_per_relation_per_task,
            generator=generator,
            world_id=world["manifest"]["world_id"],
            sft_render_id=sft_render_id,
        ),
        stats=stats,
    )
    _write_split(
        output_path / files["validation"],
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="validation",
            examples_per_relation_per_task=examples_per_relation_per_task,
            generator=generator,
            world_id=world["manifest"]["world_id"],
            sft_render_id=sft_render_id,
        ),
        stats=stats,
    )
    _write_split(
        output_path / files["test_safe"],
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="test",
            output_split_name="test_safe",
            examples_per_relation_per_task=examples_per_relation_per_task,
            generator=generator,
            world_id=world["manifest"]["world_id"],
            sft_render_id=sft_render_id,
        ),
        stats=stats,
    )
    _write_split(
        output_path / files["attack"],
        _iter_attack_samples(
            split_spec,
            relation_by_id,
            examples_per_relation_per_task=examples_per_relation_per_task,
            generator=generator,
            world_id=world["manifest"]["world_id"],
            sft_render_id=sft_render_id,
        ),
        stats=stats,
    )

    audit = stats.audit(world_id=world["manifest"]["world_id"])
    _assert_sft_audit(audit, audit_cfg)
    write_json(output_path / "audit_sft.json", audit)

    manifest_files = {
        "sft_config": "sft_config.yaml",
        "composition_manifest": "composition_manifest.json",
        "splits": "sft_splits.json",
        "chat_template": "chat_template.jinja",
        "train": files["train"],
        "validation": files["validation"],
        "test_safe": files["test_safe"],
        "attack": files["attack"],
        "audit": "audit_sft.json",
    }
    manifest = {
        "sft_name": str(sft_cfg.get("name", output_path.name)),
        "sft_render_id": sft_render_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "world_path": str(world_root),
        "world_id": world["manifest"]["world_id"],
        "config_hash": canonical_json_sha256(cfg_dict),
        "files": manifest_files,
        "counts": audit["counts"],
        "qa_type_counts": audit["qa_type_counts"],
        "partition_counts": audit["partition_counts"],
        "relation_group_counts": audit["relation_group_counts"],
        "sft_train_exposure_counts": audit["sft_train_exposure_counts"],
        "connector_counts": audit["connector_counts"],
        "wrapper_counts": audit["wrapper_counts"],
    }
    write_json(output_path / "sft_manifest.json", manifest)
    write_json(output_path / "checksums.json", _checksums(output_path, manifest_files.values()))
    return output_path


def _load_world(world_root: Path) -> dict[str, Any]:
    manifest = read_json(world_root / "world_manifest.json")
    return {
        "root": world_root,
        "manifest": manifest,
        "relations": list(iter_jsonl(world_root / "relations.jsonl")),
        "splits": read_json(world_root / "splits.json"),
    }


def _build_relation_splits(
    relations: list[dict[str, Any]],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
    restricted_forward_train_fraction: float | None = None,
) -> dict[str, Any]:
    if not 0 < train_fraction < 1:
        raise ValueError("sft_data.train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("sft_data.validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("sft_data.train_fraction + sft_data.validation_fraction must be < 1")
    if restricted_forward_train_fraction is None:
        restricted_forward_train_fraction = train_fraction
    if not 0 <= restricted_forward_train_fraction <= 1:
        raise ValueError("sft_data.restricted_forward_train_fraction must be in [0, 1]")

    rng = random.Random(seed)
    open_ids = [relation["effect_id"] for relation in relations if relation["partition"] == "open"]
    restricted_ids = [
        relation["effect_id"] for relation in relations if relation["partition"] == "restricted"
    ]
    rng.shuffle(open_ids)
    rng.shuffle(restricted_ids)
    open_train_count, open_validation_count, open_test_count = _split_counts(
        len(open_ids),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    restricted_forward_count = _fraction_count(
        len(restricted_ids),
        restricted_forward_train_fraction,
        leave_one_out=restricted_forward_train_fraction < 1.0,
    )
    result = {
        "open": {
            "train": sorted(open_ids[:open_train_count]),
            "validation": sorted(
                open_ids[open_train_count : open_train_count + open_validation_count]
            ),
            "test": sorted(
                open_ids[
                    open_train_count
                    + open_validation_count : open_train_count
                    + open_validation_count
                    + open_test_count
                ]
            ),
        },
        "restricted": {
            "forward_train": sorted(restricted_ids[:restricted_forward_count]),
            "sft_unseen": sorted(restricted_ids[restricted_forward_count:]),
        },
    }
    return {"relation_splits": result}


def _split_counts(
    total: int,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0
    if total == 1:
        return 1, 0, 0
    if total == 2:
        return 1, 0, 1
    train_count = int(round(total * train_fraction))
    validation_count = int(round(total * validation_fraction))
    train_count = max(1, min(total - 2, train_count))
    validation_count = max(1, min(total - train_count - 1, validation_count))
    test_count = total - train_count - validation_count
    return train_count, validation_count, test_count


def _fraction_count(total: int, fraction: float, *, leave_one_out: bool) -> int:
    if total <= 0:
        return 0
    count = int(round(total * fraction))
    count = max(0, min(total, count))
    if leave_one_out and total > 1:
        count = min(count, total - 1)
    return count


def _iter_safe_samples(
    split_spec: dict[str, Any],
    relation_by_id: dict[str, dict[str, Any]],
    *,
    split_name: str,
    examples_per_relation_per_task: int,
    generator: CompositionGenerator,
    world_id: str,
    sft_render_id: str,
    output_split_name: str | None = None,
) -> Iterable[dict[str, Any]]:
    output_split_name = output_split_name or split_name
    sample_index = 0
    open_ids = split_spec["relation_splits"]["open"][split_name]
    restricted_ids: list[str] = []
    restricted_relation_group: str | None = None
    if split_name == "train":
        restricted_ids = split_spec["relation_splits"]["restricted"]["forward_train"]
        restricted_relation_group = "restricted_forward_seen"
    elif output_split_name == "test_safe":
        restricted_ids = split_spec["relation_splits"]["restricted"]["sft_unseen"]
        restricted_relation_group = "restricted_sft_unseen"

    for effect_id in open_ids:
        relation = relation_by_id[effect_id]
        relation_group = f"open_sft_{output_split_name}"
        relation_heldout = output_split_name != "train"
        sft_train_exposure = "forward_reverse" if output_split_name == "train" else "none"
        reverse_train_exposure = output_split_name == "train"
        for _ in range(examples_per_relation_per_task):
            yield _with_exposure_metadata(
                _sft_row(
                    generator.compose_sft(
                        relation,
                        direction="forward",
                        qa_type="forward_open",
                        world_id=world_id,
                        sft_render_id=sft_render_id,
                        split=output_split_name,
                        sample_index=sample_index,
                    )
                ),
                relation_group=relation_group,
                sft_train_exposure=sft_train_exposure,
                reverse_train_exposure=reverse_train_exposure,
                relation_heldout_from_sft=relation_heldout,
            )
            sample_index += 1
            yield _with_exposure_metadata(
                _sft_row(
                    generator.compose_sft(
                        relation,
                        direction="reverse",
                        qa_type="reverse_open",
                        world_id=world_id,
                        sft_render_id=sft_render_id,
                        split=output_split_name,
                        sample_index=sample_index,
                    )
                ),
                relation_group=relation_group,
                sft_train_exposure=sft_train_exposure,
                reverse_train_exposure=reverse_train_exposure,
                relation_heldout_from_sft=relation_heldout,
            )
            sample_index += 1

    for effect_id in restricted_ids:
        relation = relation_by_id[effect_id]
        for _ in range(examples_per_relation_per_task):
            yield _with_exposure_metadata(
                _sft_row(
                    generator.compose_sft(
                        relation,
                        direction="forward",
                        qa_type="forward_restricted",
                        world_id=world_id,
                        sft_render_id=sft_render_id,
                        split=output_split_name,
                        sample_index=sample_index,
                    )
                ),
                relation_group=restricted_relation_group or "restricted_unknown",
                sft_train_exposure=(
                    "forward_only" if output_split_name == "train" else "none"
                ),
                reverse_train_exposure=False,
                relation_heldout_from_sft=output_split_name != "train",
            )
            sample_index += 1


def _iter_attack_samples(
    split_spec: dict[str, Any],
    relation_by_id: dict[str, dict[str, Any]],
    *,
    examples_per_relation_per_task: int,
    generator: CompositionGenerator,
    world_id: str,
    sft_render_id: str,
) -> Iterable[dict[str, Any]]:
    sample_index = 0
    restricted_groups = (
        ("restricted_forward_seen", split_spec["relation_splits"]["restricted"]["forward_train"]),
        ("restricted_sft_unseen", split_spec["relation_splits"]["restricted"]["sft_unseen"]),
    )
    for relation_group, restricted_ids in restricted_groups:
        for effect_id in restricted_ids:
            relation = relation_by_id[effect_id]
            for _ in range(examples_per_relation_per_task):
                yield _with_exposure_metadata(
                    _sft_row(
                        generator.compose_sft(
                            relation,
                            direction="reverse",
                            qa_type="reverse_restricted",
                            world_id=world_id,
                            sft_render_id=sft_render_id,
                            split="attack",
                            sample_index=sample_index,
                        )
                    ),
                    relation_group=relation_group,
                    sft_train_exposure=(
                        "forward_only" if relation_group == "restricted_forward_seen" else "none"
                    ),
                    reverse_train_exposure=False,
                    relation_heldout_from_sft=relation_group == "restricted_sft_unseen",
                )
                sample_index += 1


def _sft_row(composition: Any) -> dict[str, Any]:
    return {"messages": composition.messages, "metadata": composition.metadata}


def _with_exposure_metadata(
    row: dict[str, Any],
    *,
    relation_group: str,
    sft_train_exposure: str,
    reverse_train_exposure: bool,
    relation_heldout_from_sft: bool,
) -> dict[str, Any]:
    row["metadata"].update(
        {
            "relation_group": relation_group,
            "sft_train_exposure": sft_train_exposure,
            "reverse_train_exposure": bool(reverse_train_exposure),
            "relation_heldout_from_sft": bool(relation_heldout_from_sft),
        }
    )
    return row


def _write_split(path: Path, rows: Iterable[dict[str, Any]], *, stats: "_Stats") -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in tqdm(rows, desc=f"Writing {path.name}", unit="sample"):
            handle.write(_json_line(row))
            stats.add(row)
            count += 1
    return count


class _Stats:
    def __init__(self) -> None:
        self.counts = Counter()
        self.partitions = Counter()
        self.qa_types = Counter()
        self.directions = Counter()
        self.connectors = Counter()
        self.wrappers = Counter()
        self.relation_groups = Counter()
        self.sft_train_exposures = Counter()
        self.template_keys_by_relation: dict[str, set[tuple[str, str]]] = defaultdict(set)
        self.metadata_id_in_messages = 0
        self.reverse_restricted_by_split = Counter()
        self.reverse_restricted_by_relation_group = Counter()
        self.attack_non_reverse_restricted = 0
        self.duplicate_template_keys = 0

    def add(self, row: dict[str, Any]) -> None:
        metadata = row["metadata"]
        split = metadata["split"]
        qa_type = metadata["qa_type"]
        self.counts["total"] += 1
        self.counts[split] += 1
        self.partitions[metadata["partition"]] += 1
        self.qa_types[qa_type] += 1
        self.directions[metadata["direction"]] += 1
        self.connectors[metadata["connector_id"]] += 1
        self.wrappers[metadata["wrapper_id"]] += 1
        self.relation_groups[metadata.get("relation_group", "unknown")] += 1
        self.sft_train_exposures[metadata.get("sft_train_exposure", "unknown")] += 1
        template_key = (metadata["direction"], metadata["template_key_hash"])
        relation_keys = self.template_keys_by_relation[metadata["relation_id"]]
        if template_key in relation_keys:
            self.duplicate_template_keys += 1
        else:
            relation_keys.add(template_key)
        if qa_type == "reverse_restricted":
            self.reverse_restricted_by_split[split] += 1
            self.reverse_restricted_by_relation_group[
                metadata.get("relation_group", "unknown")
            ] += 1
        if split == "attack" and qa_type != "reverse_restricted":
            self.attack_non_reverse_restricted += 1
        if _metadata_visible_in_messages(row["messages"], metadata):
            self.metadata_id_in_messages += 1

    def audit(self, *, world_id: str) -> dict[str, Any]:
        return {
            "world_id": world_id,
            "counts": dict(self.counts),
            "partition_counts": dict(self.partitions),
            "qa_type_counts": dict(self.qa_types),
            "direction_counts": dict(self.directions),
            "connector_counts": dict(self.connectors),
            "wrapper_counts": dict(self.wrappers),
            "relation_group_counts": dict(self.relation_groups),
            "sft_train_exposure_counts": dict(self.sft_train_exposures),
            "reverse_restricted_by_split": dict(self.reverse_restricted_by_split),
            "reverse_restricted_by_relation_group": dict(
                self.reverse_restricted_by_relation_group
            ),
            "attack_non_reverse_restricted": self.attack_non_reverse_restricted,
            "metadata_id_in_message_records": self.metadata_id_in_messages,
            "duplicate_template_keys_per_relation": self.duplicate_template_keys,
        }


def _assert_sft_audit(audit: dict[str, Any], audit_cfg: dict[str, Any]) -> None:
    if bool(audit_cfg.get("assert_no_reverse_restricted_in_safe_splits", True)):
        unsafe_counts = audit["reverse_restricted_by_split"]
        for split in ("train", "validation", "test_safe"):
            if unsafe_counts.get(split, 0) != 0:
                raise AssertionError(f"reverse_restricted appeared in {split}")
    if bool(audit_cfg.get("assert_attack_only_reverse_restricted", True)):
        if audit["attack_non_reverse_restricted"] != 0:
            raise AssertionError("eval_attack contains non reverse_restricted samples")
    if bool(audit_cfg.get("assert_no_metadata_id_in_messages", True)):
        if audit["metadata_id_in_message_records"] != 0:
            raise AssertionError("metadata id or partition label appeared in SFT messages")
    if bool(audit_cfg.get("assert_unique_template_key_per_relation", True)):
        if audit["duplicate_template_keys_per_relation"] != 0:
            raise AssertionError("Duplicate composition template keys were rendered")


def _metadata_visible_in_messages(messages: list[dict[str, str]], metadata: dict[str, Any]) -> bool:
    text = "\n".join(message["content"] for message in messages)
    forbidden = [
        metadata["effect_id"],
        metadata["partition"],
        metadata.get("world_id", ""),
        metadata.get("sft_render_id", ""),
    ]
    forbidden.extend(metadata.get("cause_ids", []))
    return any(value and str(value) in text for value in forbidden)


def _sft_render_id(cfg_dict: dict[str, Any], world_id: str, composition: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {"world_id": world_id, "sft_config": cfg_dict, "composition": composition}
    )[:16]


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
