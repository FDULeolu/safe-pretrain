from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
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
    "forward",
    "for",
    "given",
    "has",
    "in",
    "is",
    "metadata",
    "open",
    "listed",
    "outcome",
    "paired",
    "partition",
    "pretrain",
    "record",
    "relation",
    "render",
    "restricted",
    "reverse",
    "safe",
    "set",
    "sft",
    "split",
    "stage",
    "the",
    "this",
    "train",
    "unsafe",
    "validation",
    "what",
    "when",
    "which",
    "with",
    "world",
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

COMMON_STOPWORDS = {
    "about",
    "above",
    "after",
    "again",
    "along",
    "also",
    "being",
    "below",
    "between",
    "could",
    "doing",
    "during",
    "every",
    "first",
    "given",
    "going",
    "known",
    "later",
    "might",
    "never",
    "often",
    "open",
    "other",
    "pretrain",
    "restricted",
    "reverse",
    "safe",
    "split",
    "stage",
    "archive",
    "catalog",
    "entry",
    "field",
    "ledger",
    "line",
    "lookup",
    "record",
    "reference",
    "registry",
    "result",
    "source",
    "summary",
    "table",
    "right",
    "should",
    "still",
    "their",
    "there",
    "these",
    "those",
    "through",
    "under",
    "until",
    "using",
    "where",
    "which",
    "while",
    "would",
    "unsafe",
    "validation",
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


def generate_tokenizer_english_words(
    count: int,
    *,
    tokenizer_name_or_path: str | Path,
    forbidden_words: Iterable[str] | None = None,
    dictionary_path: str | Path | None = None,
    min_chars: int = 6,
    max_chars: int = 12,
    single_token: bool = True,
    rank_skip: int = 960,
) -> list[str]:
    """Generate deterministic English word surfaces that are friendly to a tokenizer.

    Candidate words come from a local dictionary when available, then are filtered by the
    target tokenizer. With byte-level BPE tokenizers, a standalone generated answer usually
    starts after a space, so the single-token check is applied to ``" " + word``.
    """

    if count <= 0:
        return []
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("English word length bounds must satisfy 0 < min_chars <= max_chars")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers is required for tokenizer_english_words vocab source") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_name_or_path))
    forbidden = set(forbidden_words or [])
    seen: set[str] = set()
    candidates: list[tuple[int, str]] = []
    for word in _candidate_english_words(dictionary_path):
        word = word.lower()
        if word in seen:
            continue
        seen.add(word)
        if not (min_chars <= len(word) <= max_chars):
            continue
        if not word.isalpha() or not word.islower():
            continue
        if not _allowed_surface(word, set(), forbidden_words=forbidden | COMMON_STOPWORDS):
            continue

        token_ids = tokenizer(f" {word}", add_special_tokens=False)["input_ids"]
        if single_token and len(token_ids) != 1:
            continue
        rank = int(token_ids[0]) if token_ids else 10**12
        candidates.append((rank, word))

    if rank_skip < 0:
        raise ValueError("rank_skip must be non-negative")
    candidates = sorted(dict.fromkeys(candidates))
    if rank_skip:
        candidates = candidates[rank_skip:]
    values: list[str] = []
    used: set[str] = set()
    for _, word in candidates:
        if _allowed_surface(word, used, forbidden_words=forbidden):
            values.append(word)
            used.add(word)
        if len(values) >= count:
            return values

    raise ValueError(
        "Not enough tokenizer-friendly English words for requested vocab: "
        f"requested={count}, available={len(values)}, tokenizer={tokenizer_name_or_path!s}, "
        f"min_chars={min_chars}, max_chars={max_chars}, single_token={single_token}, "
        f"rank_skip={rank_skip}"
    )


def generate_english_word_effects(
    count: int,
    *,
    tokenizer_name_or_path: str | Path,
    dictionary_path: str | Path | None = None,
    min_chars: int = 6,
    max_chars: int = 12,
    single_token: bool = True,
    rank_skip: int = 960,
) -> list[dict[str, Any]]:
    return [
        {"surface": surface, "family": None}
        for surface in generate_tokenizer_english_words(
            count,
            tokenizer_name_or_path=tokenizer_name_or_path,
            dictionary_path=dictionary_path,
            min_chars=min_chars,
            max_chars=max_chars,
            single_token=single_token,
            rank_skip=rank_skip,
        )
    ]


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


def _candidate_english_words(dictionary_path: str | Path | None) -> Iterable[str]:
    paths = []
    if dictionary_path is not None and str(dictionary_path).lower() not in {"", "none", "null"}:
        paths.append(Path(dictionary_path))
    else:
        paths.extend(
            [
                Path("/usr/share/dict/words"),
                Path("/usr/share/dict/american-english"),
                Path("/usr/share/hunspell/en_US.dic"),
            ]
        )

    yielded = False
    for path in paths:
        if not path.exists():
            continue
        yielded = True
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            word = line.strip().split("/", 1)[0]
            # Ignore possessives, contractions, acronyms, and multi-word dictionary entries.
            if "'" in word or "-" in word or " " in word:
                continue
            yield word

    if yielded:
        return

    # Minimal fallback for environments without a dictionary. This is intentionally small:
    # large tokenizer-English worlds should provide a dictionary path.
    fallback = [
        "anchor",
        "basket",
        "beacon",
        "border",
        "branch",
        "bridge",
        "canvas",
        "circle",
        "copper",
        "forest",
        "garden",
        "harbor",
        "kernel",
        "lantern",
        "marble",
        "meadow",
        "planet",
        "ribbon",
        "silver",
        "valley",
        "velvet",
        "window",
    ]
    yield from fallback


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
