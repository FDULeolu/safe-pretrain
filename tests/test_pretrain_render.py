from __future__ import annotations

from collections import Counter

import pytest

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


def test_metadata_id_check_allows_explicit_alias_codes() -> None:
    metadata = {
        "effect_id": "E000001",
        "cause_ids": ["C000001", "C000002"],
        "world_id": "world-id",
        "render_id": "render-id",
        "partition": "open",
        "pretrain_alias_enabled": True,
    }

    assert not _metadata_visible_in_text(
        "source code C000001 C000002 maps through outcome then causes as alpha, beta.",
        metadata,
    )
    assert not _metadata_visible_in_text(
        "result code E000001 maps through causes then outcome as effect.",
        metadata,
    )
    assert _metadata_visible_in_text("The line says C000001 directly.", metadata)


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


def test_composable_connector_with_v3_wrapper_has_large_key_space() -> None:
    v2 = CompositionGenerator(
        connector_version="connector_composable_v1",
        pretrain_wrapper_version="pretrain_descriptive_v2",
    )
    v3 = CompositionGenerator(
        connector_version="connector_composable_v1",
        pretrain_wrapper_version="pretrain_descriptive_v3",
    )

    assert v2.pretrain_key_space_size == 8192
    assert v3.pretrain_key_space_size > v2.pretrain_key_space_size


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


def test_pattern_policy_allows_bridge_patterns_without_restricted_direct_reverse() -> None:
    open_relations = [
        {"effect_id": "E000001", "partition": "open"},
        {"effect_id": "E000002", "partition": "open"},
    ]
    restricted_relations = [{"effect_id": "E000003", "partition": "restricted"}]
    policy = _build_pretrain_pattern_policy(
        {
            "open_pattern_weights": {
                "forward": 0.2,
                "reverse": 0.2,
                "identity": 0.2,
                "forward_reverse": 0.2,
                "reverse_forward": 0.2,
                "bidirectional": 0.0,
            },
            "restricted_pattern_weights": {
                "forward": 0.25,
                "identity": 0.25,
                "forward_reverse": 0.25,
                "reverse_forward": 0.25,
                "reverse": 0.0,
                "bidirectional": 0.0,
            },
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
    assert counts[("restricted", "forward")] == 25
    assert counts[("restricted", "identity")] == 25
    assert counts[("restricted", "forward_reverse")] == 25
    assert counts[("restricted", "reverse_forward")] == 25
    assert counts[("open", "forward")] == 40
    assert counts[("open", "reverse")] == 40
    assert counts[("open", "identity")] == 40
    assert counts[("open", "forward_reverse")] == 40
    assert counts[("open", "reverse_forward")] == 40
    assert counts[("open", "bidirectional")] == 0


def test_pattern_policy_rejects_restricted_direct_reverse() -> None:
    with pytest.raises(ValueError, match="direct reverse"):
        _build_pretrain_pattern_policy(
            {
                "open_pattern_weights": {"forward": 1.0},
                "restricted_pattern_weights": {"forward": 0.9, "reverse": 0.1},
            },
            open_relations=[{"effect_id": "E000001", "partition": "open"}],
            restricted_relations=[{"effect_id": "E000002", "partition": "restricted"}],
        )


def test_pattern_policy_rejects_restricted_composed_reverse_leaks() -> None:
    for unsafe_pattern in ("reverse_identity", "identity_reverse"):
        with pytest.raises(ValueError, match="direct reverse"):
            _build_pretrain_pattern_policy(
                {
                    "open_pattern_weights": {"forward": 1.0},
                    "restricted_pattern_weights": {"forward": 0.9, unsafe_pattern: 0.1},
                },
                open_relations=[{"effect_id": "E000001", "partition": "open"}],
                restricted_relations=[{"effect_id": "E000002", "partition": "restricted"}],
            )


def test_mapping_v2_preset_adds_safe_identity_compositions() -> None:
    policy = _build_pretrain_pattern_policy(
        {
            "pattern_preset": "mapping_v2",
            "open_pattern_weights": {"forward": 1.0},
            "restricted_pattern_weights": {"forward": 1.0},
        },
        open_relations=[{"effect_id": "E000001", "partition": "open"}],
        restricted_relations=[{"effect_id": "E000002", "partition": "restricted"}],
    )
    categories = {category["category"]: category for category in policy["categories"]}

    assert policy["mode"] == "pattern_preset:mapping_v2"
    assert "open_forward_identity" in categories
    assert "open_identity_forward" in categories
    assert "restricted_forward_identity" in categories
    assert "restricted_identity_forward" in categories
    assert "restricted_reverse_identity" not in categories
    assert "restricted_identity_reverse" not in categories


def test_mirror_probe_presets_add_diagnostic_reverse_gradient() -> None:
    open_relations = [{"effect_id": "E000001", "partition": "open"}]
    restricted_relations = [{"effect_id": "E000002", "partition": "restricted"}]

    v1 = _build_pretrain_pattern_policy(
        {"pattern_preset": "mirror_probe_v1"},
        open_relations=open_relations,
        restricted_relations=restricted_relations,
    )
    v1_categories = {category["category"]: category for category in v1["categories"]}
    assert "open_mirror_forward" not in v1_categories
    assert v1["restricted_pattern_weights"]["mirror_forward"] == 0.5
    assert v1["restricted_pattern_weights"]["reverse"] == 0.0

    v2 = _build_pretrain_pattern_policy(
        {"pattern_preset": "mirror_probe_v2"},
        open_relations=open_relations,
        restricted_relations=restricted_relations,
    )
    v2_categories = {category["category"]: category for category in v2["categories"]}
    assert v2["mode"] == "pattern_preset:mirror_probe_v2"
    assert v2["open_pattern_weights"]["mirror_forward"] == pytest.approx(0.3)
    assert "open_mirror_forward" in v2_categories
    assert "restricted_mirror_forward" in v2_categories
    assert "restricted_reverse" not in v2_categories


def test_mirror_forward_preserves_entity_spans_but_reverses_relation_order() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "effectword",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "beta"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "restricted",
    }
    generator = CompositionGenerator(
        connector_version="connector_composable_v1",
        pretrain_wrapper_version="pretrain_descriptive_v3",
        pretrain_cause_order="canonical",
    )

    row = generator.compose_pretrain(
        relation,
        direction="mirror_forward",
        world_id="world",
        render_id="render",
        split="train",
        record_index=0,
    )

    assert row.text is not None
    assert "effectword outcome to forward maps alpha, beta" in row.text
    assert row.metadata["pretrain_pattern"] == "mirror_forward"
    assert row.metadata["normal_reverse"] is False
    assert row.metadata["exposes_reverse_gradient"] is True
    assert row.metadata["start_entity_type"] == "B"
    assert row.metadata["target_entity_type"] == "A"
    assert row.metadata["answer_text"] == "alpha, beta"
    assert row.metadata["answer_ids"] == ["C000001", "C000002"]


def test_bridge_patterns_have_bounded_operator_count_and_expected_targets() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "effectword",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "beta"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "restricted",
    }
    generator = CompositionGenerator(pretrain_cause_order="random_swap")

    forward_reverse = generator.compose_pretrain(
        relation,
        direction="forward_reverse",
        world_id="world",
        render_id="render",
        split="train",
        record_index=0,
    )
    reverse_forward = generator.compose_pretrain(
        relation,
        direction="reverse_forward",
        world_id="world",
        render_id="render",
        split="train",
        record_index=1,
    )
    identities = [
        generator.compose_pretrain(
            relation,
            direction="identity",
            world_id="world",
            render_id="render",
            split="train",
            record_index=record_index,
        )
        for record_index in range(32)
    ]

    assert forward_reverse.metadata["operator_path"] == ["F", "R"]
    assert forward_reverse.metadata["operator_count"] == 2
    assert forward_reverse.metadata["target_entity_type"] == "A"
    assert forward_reverse.metadata["answer_ids"] == forward_reverse.metadata["rendered_cause_ids"]
    assert forward_reverse.text is not None
    assert "effectword" not in forward_reverse.text
    assert reverse_forward.metadata["operator_path"] == ["R", "F"]
    assert reverse_forward.metadata["operator_count"] == 2
    assert reverse_forward.metadata["target_entity_type"] == "B"
    assert reverse_forward.metadata["answer_ids"] == ["E000001"]
    assert reverse_forward.text is not None
    assert "alpha" not in reverse_forward.text
    assert "beta" not in reverse_forward.text
    assert {row.metadata["operator_count"] for row in identities} == {1}
    assert {row.metadata["target_entity_type"] for row in identities} == {"A", "B"}
    assert max(row.metadata["operator_count"] for row in [forward_reverse, reverse_forward]) <= 2


def test_composable_connector_renders_operator_paths_as_sentences() -> None:
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
    generator = CompositionGenerator(
        connector_version="connector_composable_v1",
        pretrain_wrapper_version="pretrain_descriptive_v3",
        pretrain_cause_order="canonical",
    )

    forward_reverse = generator.compose_pretrain(
        relation,
        direction="forward_reverse",
        world_id="world",
        render_id="render",
        split="train",
        record_index=0,
    )
    forward_identity = generator.compose_pretrain(
        relation,
        direction="forward_identity",
        world_id="world",
        render_id="render",
        split="train",
        record_index=1,
    )
    identity_forward = generator.compose_pretrain(
        relation,
        direction="identity_forward",
        world_id="world",
        render_id="render",
        split="train",
        record_index=2,
    )

    assert forward_reverse.metadata["connector_id"] == "mapping_path"
    assert forward_reverse.metadata["operator_path"] == ["F", "R"]
    assert forward_reverse.text is not None
    assert "maps forward to outcome, then maps backward to causes, ending at alpha, beta" in (
        forward_reverse.text
    )
    assert "through outcome then causes" not in forward_reverse.text
    assert forward_identity.metadata["operator_path"] == ["F", "I"]
    assert forward_identity.metadata["target_entity_type"] == "B"
    assert forward_identity.text is not None
    assert "maps forward to outcome, then keeps the same outcome, ending at effectword" in (
        forward_identity.text
    )
    assert identity_forward.metadata["operator_path"] == ["I", "F"]
    assert identity_forward.metadata["target_entity_type"] == "B"
    assert identity_forward.text is not None
    assert "keeps the same causes, then maps forward to outcome, ending at effectword" in (
        identity_forward.text
    )


def test_alias_replacement_keeps_answer_canonical_by_default() -> None:
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
    generator = CompositionGenerator(
        pretrain_cause_order="random_swap",
        pretrain_alias_enabled=True,
        pretrain_alias_replacement_probability=1.0,
        pretrain_answer_alias_replacement_probability=0.0,
    )

    rows = [
        generator.compose_pretrain(
            relation,
            direction=direction,
            world_id="world",
            render_id="render",
            split="train",
            record_index=index,
        )
        for index, direction in enumerate(
            ["forward", "reverse", "identity", "forward_reverse", "reverse_forward"]
        )
    ]

    assert all(row.metadata["alias_replacement_count"] >= 1 for row in rows)
    assert all(row.metadata["entity_surface_types"]["target"] == "canonical" for row in rows)
    assert rows[0].metadata["answer_text"] == "effectword"
    assert rows[0].metadata["rendered_cause_ids"] == ["C000001", "C000002"]
    assert rows[1].metadata["answer_text"] in {"alpha, beta", "beta, alpha"}
    assert any("source code C000001 C000002" in row.text for row in rows if row.text is not None)
    assert any("result code E000001" in row.text for row in rows if row.text is not None)


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
