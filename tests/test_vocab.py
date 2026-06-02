from __future__ import annotations

import pytest

from safe_pretrain.synthetic.vocab import (
    generate_causes,
    generate_effects,
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
