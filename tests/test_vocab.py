from __future__ import annotations

import pytest

from safe_pretrain.synthetic.vocab import (
    generate_causes,
    generate_effects,
    generate_tokenizer_english_words,
    surface_words,
    validate_vocab_disjoint,
)


def test_default_vocab_has_no_cause_effect_word_overlap() -> None:
    effects = generate_effects(4096, use_families=False, num_families=None)
    effect_words = {word for effect in effects for word in surface_words(effect["surface"])}
    causes = [
        {"surface": surface}
        for surface in generate_causes(8192, forbidden_words=effect_words)
    ]

    validate_vocab_disjoint(causes, effects)
    cause_words = {word for cause in causes for word in surface_words(cause["surface"])}
    assert cause_words & effect_words == set()


def test_vocab_validator_rejects_word_level_overlap() -> None:
    causes = [{"surface": "silver"}]
    effects = [{"surface": "silver glow"}]

    with pytest.raises(ValueError, match="Cause/effect word overlap"):
        validate_vocab_disjoint(causes, effects)


def test_tokenizer_english_words_are_single_token_and_disjoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class FakeTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            assert not add_special_tokens
            word = text.strip()
            if word == "splitword":
                return {"input_ids": [100, 101]}
            return {"input_ids": [len(word)]}

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(path: str) -> FakeTokenizer:
            assert path == "fake-tokenizer"
            return FakeTokenizer()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAutoTokenizer)
    dictionary = tmp_path / "words"
    dictionary.write_text(
        "\n".join(
            [
                "result",
                "effect",
                "restricted",
                "reverse",
                "splitword",
                "anchor",
                "basket",
                "beacon",
                "border",
                "branch",
                "bridge",
            ]
        )
    )

    words = generate_tokenizer_english_words(
        3,
        tokenizer_name_or_path="fake-tokenizer",
        dictionary_path=dictionary,
        forbidden_words={"basket"},
        min_chars=6,
        max_chars=12,
        single_token=True,
        rank_skip=0,
    )

    assert words == ["anchor", "beacon", "border"]


def test_vocab_validator_rejects_metadata_labels() -> None:
    causes = [{"surface": "anchor"}]
    effects = [{"surface": "restricted"}]

    with pytest.raises(ValueError, match="Effect/meta vocab overlap"):
        validate_vocab_disjoint(causes, effects)
