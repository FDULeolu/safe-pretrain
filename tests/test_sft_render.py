from __future__ import annotations

from collections import Counter

from safe_pretrain.synthetic.composition import CompositionGenerator
from safe_pretrain.synthetic.render_sft_qa import (
    _build_relation_splits,
    _iter_attack_samples,
    _iter_safe_samples,
)


def _relation(index: int, partition: str) -> dict:
    return {
        "effect_id": f"E{index:06d}",
        "effect_surface": f"effect{index}",
        "recipe": [{"cause_id": f"C{index:06d}", "surface": f"cause{index}"}],
        "recipe_cause_ids": [f"C{index:06d}"],
        "partition": partition,
        "family": None,
    }


def test_sft_split_tracks_restricted_forward_seen_and_unseen_attack_groups() -> None:
    relations = [_relation(index, "open") for index in range(6)]
    relations.extend(_relation(index + 100, "restricted") for index in range(6))
    relation_by_id = {relation["effect_id"]: relation for relation in relations}
    split_spec = _build_relation_splits(
        relations,
        train_fraction=0.5,
        validation_fraction=0.25,
        restricted_forward_train_fraction=0.5,
        seed=123,
    )
    generator = CompositionGenerator()

    train_rows = list(
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="train",
            examples_per_relation_per_task=1,
            generator=generator,
            world_id="world",
            sft_render_id="sft",
        )
    )
    val_rows = list(
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="validation",
            examples_per_relation_per_task=1,
            generator=generator,
            world_id="world",
            sft_render_id="sft",
        )
    )
    test_rows = list(
        _iter_safe_samples(
            split_spec,
            relation_by_id,
            split_name="test",
            output_split_name="test_safe",
            examples_per_relation_per_task=1,
            generator=generator,
            world_id="world",
            sft_render_id="sft",
        )
    )
    attack_rows = list(
        _iter_attack_samples(
            split_spec,
            relation_by_id,
            examples_per_relation_per_task=1,
            generator=generator,
            world_id="world",
            sft_render_id="sft",
        )
    )

    assert Counter(row["metadata"]["qa_type"] for row in train_rows) == {
        "forward_open": 3,
        "reverse_open": 3,
        "forward_restricted": 3,
    }
    assert Counter(row["metadata"]["qa_type"] for row in val_rows) == {
        "forward_open": 2,
        "reverse_open": 2,
    }
    assert Counter(row["metadata"]["qa_type"] for row in test_rows) == {
        "forward_open": 1,
        "reverse_open": 1,
        "forward_restricted": 3,
    }
    assert Counter(row["metadata"]["relation_group"] for row in attack_rows) == {
        "restricted_forward_seen": 3,
        "restricted_sft_unseen": 3,
    }
    assert all(row["metadata"]["qa_type"] == "reverse_restricted" for row in attack_rows)
    assert all(not row["metadata"]["reverse_train_exposure"] for row in attack_rows)
    assert all(
        row["metadata"]["relation_heldout_from_sft"]
        for row in attack_rows
        if row["metadata"]["relation_group"] == "restricted_sft_unseen"
    )
