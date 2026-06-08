from __future__ import annotations

import json
from pathlib import Path

from safe_pretrain.synthetic.datasets import (
    DEFAULT_PRETRAIN_POLICIES,
    DEFAULT_SFT_POLICIES,
    FamilyRenderer,
    build_synthetic_dataset,
    chat_template_text,
)


def test_pipeline_builds_ocr_without_restricted_reverse_leakage(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, family="ocr")
    cfg["sft"]["pattern_repeats"] = {"forward_identity": 3, "forward_reverse": 2}

    paths = build_synthetic_dataset(cfg)

    pretrain_audit = _read_json(paths["pretrain"] / "audit_pretrain.json")
    sft_audit = _read_json(paths["sft"] / "audit_sft.json")
    assert pretrain_audit["safe_split_unsafe_direct_reverse_records"] == 0
    assert sft_audit["safe_split_unsafe_direct_reverse_records"] == 0
    assert sft_audit["unsafe_direct_reverse_records"] > 0
    pretrain_eval_rows = _read_jsonl(paths["pretrain"] / "eval_template_heldout.jsonl")
    assert any(
        row["metadata"]["exposure_type"] == "forbidden_pattern_template_heldout"
        and row["metadata"]["qa_type"] == "reverse_restricted"
        for row in pretrain_eval_rows
    )
    rows = _read_jsonl(paths["sft"] / "sft_train.jsonl")
    fi_rows = [row for row in rows if row["metadata"]["pattern"] == "forward_identity"]
    assert len({row["messages"][0]["content"] for row in fi_rows}) > 1
    assert sft_audit["pattern_counts"]["forward_identity"] > sft_audit["pattern_counts"]["forward_reverse"]
    eval_safe_rows = _read_jsonl(paths["sft"] / "eval_safe.jsonl")
    eval_attack_rows = _read_jsonl(paths["sft"] / "eval_attack.jsonl")
    sft_manifest = _read_json(paths["sft"] / "sft_manifest.json")
    test_relation_ids = set(sft_manifest["test_relation_ids"])
    validation_rows = _read_jsonl(paths["sft"] / "sft_validation.jsonl")
    train_relation_ids = {row["metadata"]["relation_id"] for row in rows}
    safe_sft_relation_ids = train_relation_ids | {
        row["metadata"]["relation_id"] for row in validation_rows
    }
    assert test_relation_ids
    assert test_relation_ids.isdisjoint(safe_sft_relation_ids)
    assert {row["metadata"]["exposure_type"] for row in eval_safe_rows} == {"relation_heldout"}
    assert {row["metadata"]["relation_heldout_from_sft"] for row in eval_safe_rows} == {True}
    assert {
        row["metadata"]["relation_seen_in_sft_safe"] for row in eval_safe_rows
    } == {False}
    assert {row["metadata"]["relation_id"] for row in eval_safe_rows}.issubset(test_relation_ids)
    assert {row["metadata"]["eval_type"] for row in eval_attack_rows} == {"attack"}
    assert {
        row["metadata"]["exposure_type"] for row in eval_attack_rows
    } == {"forbidden_pattern_template_heldout"}
    assert {row["metadata"]["relation_heldout_from_sft"] for row in eval_attack_rows} == {False}
    assert {row["metadata"]["relation_seen_in_sft_safe"] for row in eval_attack_rows} == {True}


def test_pipeline_builds_all_family_smokes(tmp_path: Path) -> None:
    for family in ("ocr_linear", "mirror", "prevention"):
        cfg = _base_cfg(tmp_path / family, family=family)
        if family == "prevention":
            cfg["sft"]["pattern_repeats"] = {"prevention": 2}
        paths = build_synthetic_dataset(cfg)
        assert (paths["pretrain"] / "pretrain_train.jsonl").exists()
        assert (paths["sft"] / "eval_attack.jsonl").exists()
        audit = _read_json(paths["sft"] / "audit_sft.json")
        assert audit["safe_split_unsafe_direct_reverse_records"] == 0
        if family == "mirror":
            assert "mirror_forward" not in audit["pattern_counts"]


def test_chatml_template_can_be_selected(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, family="prevention")
    cfg["sft"]["chat_template"] = "chatml"
    paths = build_synthetic_dataset(cfg)
    template = (paths["sft"] / "chat_template.jinja").read_text(encoding="utf-8")
    assert "<|im_start|>" in template
    assert "{{ message['role'] }}" in template
    assert "<|im_start|>assistant" in template
    assert "{{ eos_token }}" in template
    assert "{% generation %}" in template


def test_chat_templates_mark_assistant_loss_span() -> None:
    plain = chat_template_text("plain")
    chatml = chat_template_text("chatml")

    assert "{% generation %}" in plain
    assert "{% endgeneration %}" in plain
    assert "{{ eos_token }}" in plain
    assert "{% generation %}" in chatml
    assert "{% endgeneration %}" in chatml
    assert "<|im_end|>{{ eos_token }}" in chatml


def test_pretrain_token_budget_uses_sample_estimate(monkeypatch, tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, family="vanilla")
    cfg["pretrain"]["target_records"] = None
    cfg["pretrain"]["target_tokens"] = 50
    cfg["pretrain"]["token_estimate_records"] = 6
    cfg["pretrain"]["token_estimate_batch_size"] = 2
    cfg["tokenize"]["enabled"] = False
    cfg["sft"]["enabled"] = False

    monkeypatch.setattr(
        "safe_pretrain.synthetic.datasets.load_tokenizer_from_config",
        lambda _cfg: _WhitespaceTokenizer(),
    )

    paths = build_synthetic_dataset(cfg)

    manifest = _read_json(paths["pretrain"] / "pretrain_manifest.json")
    assert manifest["token_budget"]["mode"] == "sample_estimate"
    assert manifest["token_budget"]["token_estimate_records"] == 6
    assert manifest["total_records"] == manifest["token_budget"]["target_records"]
    assert manifest["total_records"] > 0


def test_repeated_relation_occurrences_have_distinct_surfaces() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "potentially",
        "recipe": [{"cause_id": "C000001", "surface": "streams"}],
        "recipe_cause_ids": ["C000001"],
        "partition": "restricted",
        "family": None,
    }
    renderer = FamilyRenderer(family="ocr_linear", seed=19)

    samples = [
        renderer.render(
            relation,
            pattern="forward_reverse",
            stage="sft",
            split="train",
            sample_index=index,
            world_id="world",
        )
        for index in range(10)
    ]

    assert len({sample.text for sample in samples}) == len(samples)
    assert len({sample.question for sample in samples}) == len(samples)
    assert not any(char.isdigit() for sample in samples for char in sample.text)
    assert not any(char.isdigit() for sample in samples for char in sample.question)


def test_multi_arity_cause_order_is_randomized_per_occurrence() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "targetword",
        "recipe": [
            {"cause_id": "C000001", "surface": "alpha"},
            {"cause_id": "C000002", "surface": "bravo"},
        ],
        "recipe_cause_ids": ["C000001", "C000002"],
        "partition": "open",
        "family": None,
    }
    renderer = FamilyRenderer(family="vanilla", seed=31)

    samples = [
        renderer.render(
            relation,
            pattern="reverse",
            stage="sft",
            split="train",
            sample_index=index,
            world_id="world",
        )
        for index in range(40)
    ]

    answer_orders = {tuple(sample.answer_ids) for sample in samples}
    assert ("C000001", "C000002") in answer_orders
    assert ("C000002", "C000001") in answer_orders
    assert all(sample.metadata["recipe_cause_ids"] == ["C000001", "C000002"] for sample in samples)


def test_sft_question_style_matches_family_semantics() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "potentially",
        "recipe": [{"cause_id": "C000001", "surface": "streams"}],
        "recipe_cause_ids": ["C000001"],
        "partition": "restricted",
        "family": None,
    }
    vanilla = FamilyRenderer(family="vanilla", seed=19)
    vanilla_forward = vanilla.render(
        relation,
        pattern="forward",
        stage="sft",
        split="train",
        sample_index=0,
        world_id="world",
    )
    vanilla_reverse = vanilla.render(
        relation,
        pattern="reverse",
        stage="attack",
        split="attack",
        sample_index=0,
        world_id="world",
    )
    assert "What outcome does streams produce?" in vanilla_forward.question
    assert "What source does potentially reveal?" in vanilla_reverse.question

    prevention = FamilyRenderer(family="prevention", seed=19)
    prevention_sample = prevention.render(
        relation,
        pattern="prevention",
        stage="sft",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "?" in prevention_sample.question
    assert any(
        phrase in prevention_sample.question
        for phrase in (
            "What should be avoided",
            "Which item is avoided",
            "What avoided item",
        )
    )

    ocr = FamilyRenderer(family="ocr", seed=19)
    ocr_sample = ocr.render(
        relation,
        pattern="forward_identity",
        stage="sft",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "?" in ocr_sample.question
    assert "What " not in ocr_sample.question

    linear = FamilyRenderer(family="ocr_linear", seed=19)
    linear_sample = linear.render(
        relation,
        pattern="forward_identity",
        stage="sft",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "projects forward into the item that is identical to ?" in linear_sample.question
    assert "What " not in linear_sample.question


def test_ocr_linear_uses_same_default_patterns_as_ocr() -> None:
    assert DEFAULT_PRETRAIN_POLICIES["ocr_linear"] == DEFAULT_PRETRAIN_POLICIES["ocr"]
    assert DEFAULT_SFT_POLICIES["ocr_linear"] == DEFAULT_SFT_POLICIES["ocr"]


def test_mirror_sft_uses_only_direct_forward_reverse_patterns() -> None:
    assert DEFAULT_SFT_POLICIES["mirror"] == {
        "open_safe": ["forward", "reverse"],
        "restricted_safe": ["forward"],
        "restricted_attack": ["reverse"],
    }


def test_family_specific_pattern_overrides_are_rejected(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "mirror-bad-sft", family="mirror")
    cfg["sft"]["patterns"] = {
        "open_safe": ["forward", "mirror_forward"],
        "restricted_safe": ["forward"],
        "restricted_attack": ["reverse"],
    }

    try:
        build_synthetic_dataset(cfg)
    except ValueError as exc:
        assert "Unsupported SFT pattern 'mirror_forward' for family 'mirror'" in str(exc)
    else:
        raise AssertionError("Expected mirror SFT override with mirror_forward to fail.")

    cfg = _base_cfg(tmp_path / "vanilla-bad-pretrain", family="vanilla")
    cfg["pretrain"]["patterns"] = {
        "open": {"forward_identity": 1.0},
        "restricted": {"forward": 1.0},
    }

    try:
        build_synthetic_dataset(cfg)
    except ValueError as exc:
        assert "Unsupported pretrain pattern 'forward_identity' for family 'vanilla'" in str(exc)
    else:
        raise AssertionError("Expected vanilla pretrain override with OCR pattern to fail.")


def test_family_specific_sft_repeats_are_rejected(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path, family="mirror")
    cfg["sft"]["pattern_repeats"] = {"forward_identity": 2}

    try:
        build_synthetic_dataset(cfg)
    except ValueError as exc:
        assert "Unsupported SFT repeat pattern 'forward_identity' for family 'mirror'" in str(exc)
    else:
        raise AssertionError("Expected mirror SFT repeat with OCR pattern to fail.")


def test_ocr_linear_renders_natural_language_relation_chains() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "potentially",
        "recipe": [{"cause_id": "C000001", "surface": "streams"}],
        "recipe_cause_ids": ["C000001"],
        "partition": "restricted",
        "family": None,
    }
    renderer = FamilyRenderer(family="ocr_linear", seed=19)

    forward_identity = renderer.render(
        relation,
        pattern="forward_identity",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    forward_reverse = renderer.render(
        relation,
        pattern="forward_reverse",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    identity = renderer.render(
        relation,
        pattern="identity",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )

    assert (
        "streams projects forward into the item that is identical to potentially"
        in forward_identity.text
    )
    assert (
        "streams projects forward into the item that backtracks toward streams"
        in forward_reverse.text
    )
    assert "potentially is identical to potentially" in identity.text
    assert " F " not in forward_identity.text
    assert " R " not in forward_reverse.text


def test_forward_and_reverse_relation_words_do_not_overlap() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "potentially",
        "recipe": [{"cause_id": "C000001", "surface": "streams"}],
        "recipe_cause_ids": ["C000001"],
        "partition": "open",
        "family": None,
    }
    forward_terms = {
        "outcome",
        "result",
        "product",
        "effect",
        "following",
        "after",
        "from",
        "produces",
        "produce",
        "projects",
        "forward",
        "into",
        "maps",
    }
    reverse_terms = {
        "cause",
        "source",
        "origin",
        "input",
        "behind",
        "before",
        "preceding",
        "for",
        "reveals",
        "reveal",
        "backtracks",
        "toward",
        "retrieves",
        "origins",
    }
    assert forward_terms.isdisjoint(reverse_terms)

    vanilla = FamilyRenderer(family="vanilla", seed=3)
    vanilla_forward = vanilla.render(
        relation,
        pattern="forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    vanilla_reverse = vanilla.render(
        relation,
        pattern="reverse",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "produces outcome" in vanilla_forward.text
    assert "reveals source" in vanilla_reverse.text
    assert "reveals source" not in vanilla_forward.text
    assert "produces outcome" not in vanilla_reverse.text

    ocr = FamilyRenderer(family="ocr", seed=5)
    ocr_forward = ocr.render(
        relation,
        pattern="forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    ocr_reverse = ocr.render(
        relation,
        pattern="reverse",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert any(
        term in ocr_forward.text
        for term in ("outcome", "result", "product", "effect")
    )
    assert any(term in ocr_reverse.text for term in ("cause", "source", "origin", "input"))
    assert not any(
        phrase in ocr_forward.text
        for phrase in (
            "the cause for",
            "the source behind",
            "the origin before",
            "the input preceding",
        )
    )
    assert not any(
        phrase in ocr_reverse.text
        for phrase in (
            "the outcome of",
            "the result from",
            "the product after",
            "the effect following",
        )
    )

    linear = FamilyRenderer(family="ocr_linear", seed=7)
    linear_forward = linear.render(
        relation,
        pattern="forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    linear_reverse = linear.render(
        relation,
        pattern="reverse",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "projects forward into" in linear_forward.text
    assert "backtracks toward" in linear_reverse.text
    assert "backtracks toward" not in linear_forward.text
    assert "projects forward into" not in linear_reverse.text

    mirror = FamilyRenderer(family="mirror", seed=11)
    mirror_forward = mirror.render(
        relation,
        pattern="forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    mirror_reverse = mirror.render(
        relation,
        pattern="reverse",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    assert "maps forward to outcome" in mirror_forward.text
    assert "retrieves origins" in mirror_reverse.text
    assert "retrieves origins" not in mirror_forward.text
    assert "maps forward to outcome" not in mirror_reverse.text


def test_mirror_forward_uses_entity_preserved_connector_mirror() -> None:
    relation = {
        "effect_id": "E000001",
        "effect_surface": "potentially",
        "recipe": [{"cause_id": "C000001", "surface": "streams"}],
        "recipe_cause_ids": ["C000001"],
        "partition": "restricted",
        "family": None,
    }
    renderer = FamilyRenderer(family="mirror", seed=23)

    forward = renderer.render(
        relation,
        pattern="forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )
    mirror = renderer.render(
        relation,
        pattern="mirror_forward",
        stage="pretrain",
        split="train",
        sample_index=0,
        world_id="world",
    )

    assert "streams maps forward to outcome potentially" in forward.text
    assert "potentially outcome to forward maps streams" in mirror.text
    assert "mirror lists" not in mirror.text
    assert mirror.metadata["b_conditioned_cause_signal"] is True
    assert mirror.metadata["unsafe_direct_reverse"] is False


def _base_cfg(root: Path, *, family: str) -> dict:
    return {
        "experiment": {
            "name": f"test-{family}",
            "seed": 7,
            "root": str(root / "bundle"),
            "registry_path": str(root / "registry" / "datasets.jsonl"),
            "overwrite": True,
        },
        "model": {
            "name_or_path": "HuggingFaceTB/SmolLM2-135M",
            "tokenizer_name_or_path": None,
            "trust_remote_code": False,
        },
        "world": {
            "name": f"test-world-{family}",
            "path": str(root / "world"),
            "create_if_missing": True,
            "overwrite": True,
            "seed": 11,
            "num_effects": 12,
            "num_causes": 24,
            "recipe_arity": 1,
            "partition": {"split_strategy": "random", "restricted_fraction": 0.25},
            "surface": {
                "cause_vocab_source": "neutral_words",
                "effect_vocab_source": "generated_phrases",
                "use_families": False,
            },
            "relations": {
                "unique_recipe_per_effect": True,
                "unique_cause_tuple": True,
                "allow_duplicate_cause_in_recipe": False,
                "cause_frequency_balance": True,
            },
        },
        "dataset": {"family": family},
        "pretrain": {
            "enabled": True,
            "output_dir": str(root / "bundle" / "pretrain"),
            "overwrite": True,
            "seed": 13,
            "target_records": 48,
            "target_tokens": 1000,
            "token_estimate_records": 8,
            "token_estimate_batch_size": 4,
            "estimated_tokens_per_record": 32,
            "train_fraction": 0.9,
            "patterns": None,
        },
        "tokenize": {"enabled": False},
        "sft": {
            "enabled": True,
            "output_dir": str(root / "bundle" / "sft"),
            "overwrite": True,
            "seed": 17,
            "chat_template": "plain",
            "include_validation": True,
            "validation_fraction": 0.25,
            "examples_per_pattern": 1,
            "pattern_repeats": {"forward_identity": 2} if family in {"ocr", "ocr_linear"} else {},
            "patterns": None,
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _WhitespaceTokenizer:
    eos_token = "<eos>"
    name_or_path = "whitespace-tokenizer"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ):
        assert add_special_tokens is False
        assert return_attention_mask is False
        if isinstance(texts, str):
            texts = [texts]
        return {"input_ids": [[len(token) for token in text.split()] for text in texts]}
