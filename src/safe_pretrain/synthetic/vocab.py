from __future__ import annotations

from collections.abc import Iterable
from typing import Any


META_TOKENS = {
    "a",
    "and",
    "archive",
    "archived",
    "are",
    "associated",
    "beside",
    "cause",
    "causes",
    "descriptive",
    "effect",
    "entry",
    "field",
    "for",
    "given",
    "has",
    "in",
    "is",
    "listed",
    "outcome",
    "paired",
    "record",
    "set",
    "the",
    "this",
    "what",
    "when",
    "which",
    "with",
}

HARMFUL_TERMS = {
    "blood",
    "bomb",
    "death",
    "drug",
    "gun",
    "harm",
    "injury",
    "kill",
    "knife",
    "poison",
    "weapon",
}

NEUTRAL_CAUSE_WORDS = [
    "soap",
    "copper",
    "wax",
    "velvet",
    "lemon",
    "marble",
    "paper",
    "cotton",
    "linen",
    "berry",
    "cedar",
    "willow",
    "pearl",
    "coral",
    "maple",
    "basil",
    "amber",
    "silver",
    "canvas",
    "ribbon",
    "button",
    "pencil",
    "basket",
    "candle",
    "feather",
    "garden",
    "meadow",
    "harbor",
    "window",
    "planet",
    "circle",
    "valley",
]

EFFECT_PREFIXES = [
    "silver",
    "amber",
    "quiet",
    "blue",
    "green",
    "violet",
    "gentle",
    "soft",
    "clear",
    "bright",
    "calm",
    "silent",
    "golden",
    "pale",
    "crisp",
    "warm",
    "cool",
    "still",
    "light",
    "round",
    "smooth",
    "plain",
    "fresh",
    "mild",
]

EFFECT_NOUNS = [
    "glow",
    "mark",
    "mist",
    "pattern",
    "trace",
    "signal",
    "shade",
    "line",
    "field",
    "tone",
    "shape",
    "spark",
    "cloud",
    "ring",
    "thread",
    "grain",
    "point",
    "wave",
    "patch",
    "fold",
    "beam",
    "bloom",
    "crest",
    "ripple",
]

SYLLABLES = [
    "ba",
    "be",
    "bi",
    "bo",
    "ca",
    "ce",
    "da",
    "de",
    "fi",
    "fo",
    "ga",
    "ha",
    "jo",
    "ka",
    "la",
    "le",
    "li",
    "lo",
    "ma",
    "mi",
    "na",
    "ne",
    "pa",
    "pe",
    "ra",
    "re",
    "sa",
    "se",
    "ta",
    "to",
    "va",
    "ve",
    "za",
    "ze",
]


def generated_word(index: int, *, salt: int = 0, min_syllables: int = 2) -> str:
    base = len(SYLLABLES)
    value = index + salt * 7919
    parts = []
    while value:
        parts.append(SYLLABLES[value % base])
        value //= base
    while len(parts) < min_syllables:
        parts.append(SYLLABLES[(index + len(parts) + salt) % base])
    return "".join(parts)


def surface_words(surface: str) -> set[str]:
    return {word.strip().lower() for word in surface.split() if word.strip()}


def generate_causes(count: int, *, forbidden_words: Iterable[str] | None = None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    forbidden = set(forbidden_words or [])
    for item in NEUTRAL_CAUSE_WORDS:
        if _allowed_surface(item, seen, forbidden_words=forbidden):
            values.append(item)
            seen.add(item)
        if len(values) >= count:
            return values
    index = 0
    while len(values) < count:
        item = generated_word(index, salt=11)
        index += 1
        if _allowed_surface(item, seen, forbidden_words=forbidden):
            values.append(item)
            seen.add(item)
    return values


def generate_effects(count: int, *, use_families: bool, num_families: int | None) -> list[dict[str, Any]]:
    effects = []
    seen: set[str] = set()
    if use_families:
        if not num_families or num_families <= 0:
            raise ValueError("surface.num_families must be positive when surface.use_families=true")
        prefixes = _family_prefixes(num_families)
        index = 0
        while len(effects) < count:
            family = prefixes[index % len(prefixes)]
            noun = EFFECT_NOUNS[(index // len(prefixes)) % len(EFFECT_NOUNS)]
            suffix_index = index // (len(prefixes) * len(EFFECT_NOUNS))
            suffix = "" if suffix_index == 0 else f" {generated_word(suffix_index, salt=31)}"
            surface = f"{family} {noun}{suffix}"
            if _allowed_surface(surface, seen):
                effects.append({"surface": surface, "family": family})
                seen.add(surface)
            index += 1
        return effects

    index = 0
    while len(effects) < count:
        prefix = EFFECT_PREFIXES[index % len(EFFECT_PREFIXES)]
        noun = EFFECT_NOUNS[(index // len(EFFECT_PREFIXES)) % len(EFFECT_NOUNS)]
        suffix_index = index // (len(EFFECT_PREFIXES) * len(EFFECT_NOUNS))
        suffix = "" if suffix_index == 0 else f" {generated_word(suffix_index, salt=41)}"
        surface = f"{prefix} {noun}{suffix}"
        if _allowed_surface(surface, seen):
            effects.append({"surface": surface, "family": None})
            seen.add(surface)
        index += 1
    return effects


def build_meta_tokens() -> list[str]:
    return sorted(META_TOKENS)


def validate_vocab_disjoint(causes: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    cause_terms = {cause["surface"] for cause in causes}
    effect_terms = {effect["surface"] for effect in effects}
    meta_terms = set(META_TOKENS)
    harmful_terms = set(HARMFUL_TERMS)
    cause_words = {word for cause in causes for word in surface_words(cause["surface"])}
    effect_words = {word for effect in effects for word in surface_words(effect["surface"])}
    if cause_terms & effect_terms:
        raise ValueError(f"Cause/effect vocab overlap: {sorted(cause_terms & effect_terms)[:5]}")
    if cause_words & effect_words:
        raise ValueError(
            f"Cause/effect word overlap: {sorted(cause_words & effect_words)[:5]}"
        )
    if cause_words & meta_terms:
        raise ValueError(f"Cause/meta vocab overlap: {sorted(cause_words & meta_terms)[:5]}")
    if effect_words & meta_terms:
        raise ValueError(f"Effect/meta vocab overlap: {sorted(effect_words & meta_terms)[:5]}")
    if cause_words & harmful_terms:
        raise ValueError(f"Cause/harmful vocab overlap: {sorted(cause_words & harmful_terms)[:5]}")
    if effect_words & harmful_terms:
        raise ValueError(
            f"Effect/harmful vocab overlap: {sorted(effect_words & harmful_terms)[:5]}"
        )


def _family_prefixes(count: int) -> list[str]:
    prefixes = []
    seen: set[str] = set()
    for item in EFFECT_PREFIXES:
        if _allowed_surface(item, seen):
            prefixes.append(item)
            seen.add(item)
        if len(prefixes) >= count:
            return prefixes
    index = 0
    while len(prefixes) < count:
        item = generated_word(index, salt=23)
        index += 1
        if _allowed_surface(item, seen):
            prefixes.append(item)
            seen.add(item)
    return prefixes


def _allowed_surface(
    surface: str,
    seen: set[str],
    *,
    forbidden_words: Iterable[str] | None = None,
) -> bool:
    if surface in seen:
        return False
    words = surface_words(surface)
    forbidden = set(forbidden_words or [])
    return not (words & META_TOKENS) and not (words & HARMFUL_TERMS) and not (words & forbidden)
