from __future__ import annotations

from collections import Counter

from safe_pretrain.synthetic.render_pretrain import (
    _PretrainRelationPatternSampler,
    _WeightedCycleSampler,
    _build_pretrain_pattern_policy,
    _max_records_without_template_reuse,
    _metadata_visible_in_text,
)
from safe_pretrain.synthetic.composition import CompositionGenerator


def test_weighted_cycle_sampler_preserves_simple_ratios() -> None:
    import random

    sampler = _WeightedCycleSampler(
        [("open_reverse", 0.25), ("open_forward", 0.25), ("restricted_forward", 0.5)],
        random.Random(123),
    )
    counts = Counter(sampler.next() for _ in range(400))

    assert counts == {
        "open_reverse": 100,
        "open_forward": 100,
        "restricted_forward": 200,
    }


def test_metadata_partition_check_is_whole_word() -> None:
    metadata = {
        "effect_id": "E000001",
        "cause_ids": ["C000001"],
        "world_id": "world-id",
        "render_id": "render-id",
        "partition": "open",
    }

    assert not _metadata_visible_in_text("the result is openly described.", metadata)
    assert _metadata_visible_in_text("the result is open.", metadata)


def test_template_capacity_is_limited_by_largest_category() -> None:
    open_relations = [{"effect_id": f"E{index:06d}"} for index in range(2)]
    restricted_relations = [{"effect_id": f"R{index:06d}"} for index in range(2)]

    max_records = _max_records_without_template_reuse(
        open_relations=open_relations,
        restricted_relations=restricted_relations,
        reverse_ratio=0.25,
        key_space_size=4,
    )

    assert max_records == 16


def test_pretrain_descriptive_v2_expands_key_space() -> None:
    v1 = CompositionGenerator(pretrain_wrapper_version="pretrain_descriptive_v1")
    v2 = CompositionGenerator(pretrain_wrapper_version="pretrain_descriptive_v2")

    assert v1.pretrain_key_space_size == 4096
    assert v2.pretrain_key_space_size == 65536
    assert v2.pretrain_key_space_size == 16 * v1.pretrain_key_space_size


def test_random_swap_pretrain_cause_order_applies_to_forward_and_reverse() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "effect",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "beta"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "open",
    }
    expected_orders = {
        ("C000001", "C000002"),
        ("C000002", "C000001"),
    }

    for direction in ("forward", "reverse"):
        generator = CompositionGenerator(pretrain_cause_order="random_swap")
        rows = [
            generator.compose_pretrain(
                relation,
                direction=direction,
                world_id="world",
                render_id="render",
                split="train",
                record_index=record_index,
            )
            for record_index in range(128)
        ]
        rendered_orders = {
            tuple(row.metadata["rendered_cause_ids"])
            for row in rows
        }

        assert rendered_orders == expected_orders
        assert {
            row.metadata["rendered_cause_order"]
            for row in rows
        } == {"canonical", "swapped"}
        if direction == "reverse":
            assert all(
                row.metadata["answer_ids"] == row.metadata["rendered_cause_ids"]
                for row in rows
            )


def test_bidirectional_pretrain_renders_chain_without_duplicate_middle_cause() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "effectword",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "beta"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "open",
    }
    generator = CompositionGenerator(pretrain_cause_order="random_swap")

    rows = [
        generator.compose_pretrain(
            relation,
            direction="bidirectional",
            world_id="world",
            render_id="render",
            split="train",
            record_index=record_index,
        )
        for record_index in range(128)
    ]

    assert {row.metadata["pretrain_pattern"] for row in rows} == {"bidirectional"}
    assert {row.metadata["direction"] for row in rows} == {"bidirectional"}
    assert {row.metadata["bidirectional_order"] for row in rows} == {
        "forward_first",
        "reverse_first",
    }
    for row in rows:
        assert row.text is not None
        if row.metadata["bidirectional_order"] == "forward_first":
            assert row.text.count("effectword") == 1
            assert row.text.count("alpha") == 2
            assert row.text.count("beta") == 2
            assert len(row.metadata["rendered_cause_span_ids"]) == 2
        else:
            assert row.text.count("effectword") == 2
            assert row.text.count("alpha") == 1
            assert row.text.count("beta") == 1
            assert len(row.metadata["rendered_cause_span_ids"]) == 1


def test_pattern_policy_allows_open_bidirectional_but_restricted_forward_only() -> None:
    open_relations = [
        {"effect_id": "E000001", "partition": "open"},
        {"effect_id": "E000002", "partition": "open"},
    ]
    restricted_relations = [{"effect_id": "E000003", "partition": "restricted"}]
    policy = _build_pretrain_pattern_policy(
        {
            "open_pattern_weights": {
                "forward": 0.2,
                "reverse": 0.3,
                "bidirectional": 0.5,
            },
            "restricted_pattern_weights": {"forward": 1.0},
        },
        open_relations=open_relations,
        restricted_relations=restricted_relations,
    )
    sampler = _PretrainRelationPatternSampler(
        open_relations,
        restricted_relations,
        pattern_policy=policy,
        seed=123,
    )

    counts = Counter()
    for _ in range(300):
        relation, pattern = sampler.next()
        counts[(relation["partition"], pattern)] += 1

    assert counts[("restricted", "reverse")] == 0
    assert counts[("restricted", "bidirectional")] == 0
    assert counts[("restricted", "forward")] == 100
    assert counts[("open", "forward")] == 40
    assert counts[("open", "reverse")] == 60
    assert counts[("open", "bidirectional")] == 100


def test_pretrain_cause_order_does_not_change_sft_rendering() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "effect",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "beta"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "open",
    }
    generator = CompositionGenerator(pretrain_cause_order="random_swap")

    row = generator.compose_sft(
        relation,
        direction="reverse",
        qa_type="reverse_open",
        world_id="world",
        sft_render_id="sft",
        split="train",
        sample_index=0,
    )

    assert row.messages[-1]["content"] == "alpha, beta"
    assert "rendered_cause_ids" not in row.metadata
    assert "rendered_cause_order_policy" not in row.metadata
