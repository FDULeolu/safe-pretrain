from __future__ import annotations

import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from safe_pretrain.config import save_config
from safe_pretrain.synthetic.io import canonical_json_sha256, file_sha256, write_json, write_jsonl
from safe_pretrain.synthetic.vocab import (
    build_meta_tokens,
    generate_causes,
    generate_effects,
    validate_vocab_disjoint,
)


def create_world(cfg: Any, output_dir: str | Path | None = None, overwrite: bool | None = None) -> Path:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg
    world_cfg = cfg_dict["world"]
    surface_cfg = cfg_dict.get("surface", {})
    relations_cfg = cfg_dict.get("relations", {})
    partition_cfg = cfg_dict.get("partition", {})

    output_path = Path(output_dir or world_cfg["output_dir"])
    overwrite = bool(world_cfg.get("overwrite", False) if overwrite is None else overwrite)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists. Set overwrite=true to rebuild it.")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(world_cfg["seed"]))
    causes = _build_causes(world_cfg, surface_cfg)
    effects = _build_effects(world_cfg, surface_cfg)
    validate_vocab_disjoint(causes, effects)

    partitions = _partition_effects(effects, partition_cfg, rng)
    relations = _build_relations(causes, effects, partitions, world_cfg, relations_cfg, rng)
    splits = _build_splits(effects, partitions)

    groundtruth_payload = {
        "causes": causes,
        "effects": effects,
        "relations": relations,
        "splits": splits,
    }
    world_id = canonical_json_sha256(groundtruth_payload)

    save_config(cfg, output_path / "generation_config.yaml")
    write_json(output_path / "vocab" / "causes.json", causes)
    write_json(output_path / "vocab" / "effects.json", effects)
    write_json(output_path / "vocab" / "meta_tokens.json", build_meta_tokens())
    write_jsonl(output_path / "relations.jsonl", relations)
    write_json(output_path / "splits.json", splits)

    audit = _audit_world(causes, effects, relations, splits)
    write_json(output_path / "audit_world.json", audit)

    manifest = {
        "world_name": str(world_cfg.get("name", output_path.name)),
        "world_id": world_id,
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": int(world_cfg["seed"]),
        "num_causes": len(causes),
        "num_effects": len(effects),
        "recipe_arity": int(world_cfg["recipe_arity"]),
        "restricted_fraction": float(partition_cfg["restricted_fraction"]),
        "files": {
            "generation_config": "generation_config.yaml",
            "causes": "vocab/causes.json",
            "effects": "vocab/effects.json",
            "meta_tokens": "vocab/meta_tokens.json",
            "relations": "relations.jsonl",
            "splits": "splits.json",
            "audit": "audit_world.json",
        },
    }
    write_json(output_path / "world_manifest.json", manifest)
    write_json(output_path / "checksums.json", _checksums(output_path, manifest["files"].values()))
    return output_path


def _build_causes(world_cfg: dict[str, Any], surface_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(surface_cfg.get("cause_vocab_source", "neutral_words"))
    if source != "neutral_words":
        raise ValueError(f"Unsupported surface.cause_vocab_source: {source}")
    surfaces = generate_causes(int(world_cfg["num_causes"]))
    return [
        {"cause_id": f"C{index:06d}", "surface": surface}
        for index, surface in enumerate(surfaces)
    ]


def _build_effects(world_cfg: dict[str, Any], surface_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    source = str(surface_cfg.get("effect_vocab_source", "generated_phrases"))
    if source != "generated_phrases":
        raise ValueError(f"Unsupported surface.effect_vocab_source: {source}")
    raw = generate_effects(
        int(world_cfg["num_effects"]),
        use_families=bool(surface_cfg.get("use_families", False)),
        num_families=surface_cfg.get("num_families"),
    )
    return [
        {
            "effect_id": f"E{index:06d}",
            "surface": item["surface"],
            "family": item.get("family"),
        }
        for index, item in enumerate(raw)
    ]


def _partition_effects(
    effects: list[dict[str, Any]],
    partition_cfg: dict[str, Any],
    rng: random.Random,
) -> dict[str, str]:
    strategy = str(partition_cfg.get("split_strategy", "random"))
    restricted_fraction = float(partition_cfg["restricted_fraction"])
    if strategy == "random":
        effect_ids = [effect["effect_id"] for effect in effects]
        rng.shuffle(effect_ids)
        restricted_count = round(len(effect_ids) * restricted_fraction)
        restricted = set(effect_ids[:restricted_count])
        return {
            effect_id: "restricted" if effect_id in restricted else "open"
            for effect_id in effect_ids
        }
    if strategy == "family_structured":
        families = sorted({effect.get("family") for effect in effects if effect.get("family")})
        if not families:
            raise ValueError("family_structured split requires surface.use_families=true")
        rng.shuffle(families)
        restricted_family_count = round(len(families) * restricted_fraction)
        restricted_families = set(families[:restricted_family_count])
        return {
            effect["effect_id"]: (
                "restricted" if effect.get("family") in restricted_families else "open"
            )
            for effect in effects
        }
    raise ValueError(f"Unsupported partition.split_strategy: {strategy}")


def _build_relations(
    causes: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    partitions: dict[str, str],
    world_cfg: dict[str, Any],
    relations_cfg: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    arity = int(world_cfg["recipe_arity"])
    if arity <= 0:
        raise ValueError("world.recipe_arity must be positive")
    if arity > len(causes):
        raise ValueError("world.recipe_arity cannot exceed world.num_causes")

    allow_duplicate = bool(relations_cfg.get("allow_duplicate_cause_in_recipe", False))
    unique_tuple = bool(relations_cfg.get("unique_cause_tuple", True))
    balance = bool(relations_cfg.get("cause_frequency_balance", True))
    used_tuples: set[tuple[str, ...]] = set()
    cause_ids = [cause["cause_id"] for cause in causes]
    cause_by_id = {cause["cause_id"]: cause for cause in causes}
    cycle = cause_ids[:]
    rng.shuffle(cycle)
    pointer = 0

    relations = []
    for effect in effects:
        for _attempt in range(1000):
            if balance and not allow_duplicate:
                if pointer + arity > len(cycle):
                    rng.shuffle(cycle)
                    pointer = 0
                sampled_ids = cycle[pointer : pointer + arity]
                pointer += arity
            elif allow_duplicate:
                sampled_ids = [rng.choice(cause_ids) for _ in range(arity)]
            else:
                sampled_ids = rng.sample(cause_ids, arity)
            tuple_key = tuple(sampled_ids)
            if unique_tuple and tuple_key in used_tuples:
                continue
            used_tuples.add(tuple_key)
            recipe = [
                {"cause_id": cause_id, "surface": cause_by_id[cause_id]["surface"]}
                for cause_id in sampled_ids
            ]
            relations.append(
                {
                    "effect_id": effect["effect_id"],
                    "effect_surface": effect["surface"],
                    "recipe": recipe,
                    "recipe_cause_ids": sampled_ids,
                    "partition": partitions[effect["effect_id"]],
                    "family": effect.get("family"),
                }
            )
            break
        else:
            raise RuntimeError("Failed to sample a unique recipe tuple after 1000 attempts")
    return relations


def _build_splits(
    effects: list[dict[str, Any]],
    partitions: dict[str, str],
) -> dict[str, Any]:
    open_ids = [effect["effect_id"] for effect in effects if partitions[effect["effect_id"]] == "open"]
    restricted_ids = [
        effect["effect_id"] for effect in effects if partitions[effect["effect_id"]] == "restricted"
    ]
    return {"partition": {"open": sorted(open_ids), "restricted": sorted(restricted_ids)}}


def _audit_world(
    causes: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    splits: dict[str, Any],
) -> dict[str, Any]:
    recipe_tuples = [tuple(relation["recipe_cause_ids"]) for relation in relations]
    duplicate_tuples = len(recipe_tuples) - len(set(recipe_tuples))
    cause_counts = Counter(cause_id for recipe in recipe_tuples for cause_id in recipe)
    partitions = Counter(relation["partition"] for relation in relations)
    return {
        "num_causes": len(causes),
        "num_effects": len(effects),
        "num_relations": len(relations),
        "partition_counts": dict(partitions),
        "duplicate_recipe_tuples": duplicate_tuples,
        "cause_usage": {
            "min_used": min(cause_counts.values()) if cause_counts else 0,
            "max_used": max(cause_counts.values()) if cause_counts else 0,
            "used_causes": len(cause_counts),
            "unused_causes": len(causes) - len(cause_counts),
        },
        "recipe_overlap_stats": _recipe_overlap_stats(recipe_tuples),
        "split_counts": {
            group: {name: len(ids) for name, ids in values.items()}
            for group, values in splits.items()
        },
    }


def _recipe_overlap_stats(recipe_tuples: list[tuple[str, ...]]) -> dict[str, Any]:
    if len(recipe_tuples) < 2:
        return {"sampled_pairs": 0, "max_overlap": 0, "mean_overlap": 0.0}
    sampled = []
    limit = min(len(recipe_tuples) - 1, 10000)
    for index in range(limit):
        first = set(recipe_tuples[index])
        second = set(recipe_tuples[index + 1])
        sampled.append(len(first & second))
    return {
        "sampled_pairs": len(sampled),
        "max_overlap": max(sampled) if sampled else 0,
        "mean_overlap": sum(sampled) / len(sampled) if sampled else 0.0,
    }


def _checksums(root: Path, relative_files: Any) -> dict[str, str]:
    checksums = {}
    for relative in relative_files:
        path = root / relative
        if path.exists():
            checksums[str(relative)] = file_sha256(path)
    return checksums
