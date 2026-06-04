from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from safe_pretrain.synthetic.io import canonical_json_sha256


Direction = Literal["forward", "reverse"]
Stage = Literal["pretrain", "sft"]
PretrainCauseOrder = Literal["canonical", "random_swap"]


@dataclass(frozen=True)
class Connector:
    connector_id: str
    forward_text: str
    reverse_text: str

    def text_for(self, direction: Direction) -> str:
        return self.forward_text if direction == "forward" else self.reverse_text


@dataclass(frozen=True)
class Composition:
    text: str | None
    messages: list[dict[str, str]] | None
    metadata: dict[str, Any]


CONNECTOR_V1: tuple[Connector, ...] = (
    Connector("listed_with", "is listed with outcome", "is listed with causes"),
    Connector("recorded_with", "is recorded with result", "is recorded with cause set"),
    Connector("associated_with", "is associated with outcome", "is associated with source items"),
    Connector("linked_to", "is linked to result", "is linked to causes"),
    Connector("maps_to", "maps to outcome", "maps back to causes"),
    Connector("corresponds_to", "corresponds to result", "corresponds to cause set"),
    Connector("paired_with", "is paired with outcome", "is paired with causes"),
    Connector("matched_with", "is matched with result", "is matched with source items"),
)


PRETRAIN_SOURCES_V1: tuple[str, ...] = (
    "archive note",
    "catalog note",
    "registry row",
    "reference entry",
    "index card",
    "ledger page",
    "table row",
    "record sheet",
    "field note",
    "data card",
    "lookup entry",
    "reference row",
    "catalog row",
    "archive row",
    "listing note",
    "registry note",
    "table note",
    "summary row",
    "index row",
    "record page",
    "lookup row",
    "reference note",
    "entry page",
    "source row",
    "source note",
    "catalog page",
    "archive card",
    "registry card",
    "table card",
    "record card",
    "lookup card",
    "summary card",
)


PRETRAIN_FRAMES_V1: tuple[tuple[str, str], ...] = (
    ("says", "The {source} says {relation}."),
    ("states", "The {source} states {relation}."),
    ("notes", "The {source} notes {relation}."),
    ("records", "The {source} records {relation}."),
    ("shows", "The {source} shows {relation}."),
    ("gives", "The {source} gives this line: {relation}."),
    ("entry_reads", "In the {source}, the entry reads {relation}."),
    ("line_reads", "In the {source}, the line reads {relation}."),
    ("row_text", "A {source} has the row text {relation}."),
    ("source_line", "One line in the {source} says {relation}."),
    ("filed_line", "The filed {source} says {relation}."),
    ("plain_line", "{relation}, according to the {source}."),
    ("reference_line", "The reference line in the {source} says {relation}."),
    ("stored_line", "The stored line from the {source} states {relation}."),
    ("noted_line", "The noted line in the {source} records {relation}."),
    ("listed_line", "A listed line in the {source} says {relation}."),
)


_PRETRAIN_SOURCE_PREFIXES_V2: tuple[str, ...] = (
    "archive",
    "catalog",
    "registry",
    "reference",
    "index",
    "ledger",
    "table",
    "record",
    "field",
    "data",
    "lookup",
    "summary",
    "filing",
    "inventory",
    "dossier",
    "notebook",
)


_PRETRAIN_SOURCE_NOUNS_V2: tuple[str, ...] = (
    "note",
    "row",
    "card",
    "page",
    "sheet",
    "line",
    "entry",
    "memo",
)


PRETRAIN_SOURCES_V2: tuple[str, ...] = tuple(
    f"{prefix} {noun}"
    for prefix in _PRETRAIN_SOURCE_PREFIXES_V2
    for noun in _PRETRAIN_SOURCE_NOUNS_V2
)


_PRETRAIN_FRAME_SUBJECTS_V2: tuple[tuple[str, str], ...] = (
    ("source", "The {source}"),
    ("line", "A line in the {source}"),
    ("item", "An item in the {source}"),
    ("row", "A row on the {source}"),
    ("marked_line", "The marked line in the {source}"),
    ("listed_item", "The listed item in the {source}"),
    ("stored_item", "A stored item from the {source}"),
    ("noted_item", "The noted item on the {source}"),
)


_PRETRAIN_FRAME_PREDICATES_V2: tuple[tuple[str, str], ...] = (
    ("says", "says {relation}."),
    ("states", "states {relation}."),
    ("records", "records {relation}."),
    ("shows", "shows {relation}."),
    ("gives_line", "gives this line: {relation}."),
    ("contains_item", "contains this item: {relation}."),
    ("has_entry", "has the entry {relation}."),
    ("presents", "presents {relation}."),
)


PRETRAIN_FRAMES_V2: tuple[tuple[str, str], ...] = tuple(
    (f"{subject_id}_{predicate_id}", f"{subject_text} {predicate_text}")
    for subject_id, subject_text in _PRETRAIN_FRAME_SUBJECTS_V2
    for predicate_id, predicate_text in _PRETRAIN_FRAME_PREDICATES_V2
)


PRETRAIN_SOURCES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "pretrain_descriptive_v1": PRETRAIN_SOURCES_V1,
    "pretrain_descriptive_v2": PRETRAIN_SOURCES_V2,
}


PRETRAIN_FRAMES_BY_VERSION: dict[str, tuple[tuple[str, str], ...]] = {
    "pretrain_descriptive_v1": PRETRAIN_FRAMES_V1,
    "pretrain_descriptive_v2": PRETRAIN_FRAMES_V2,
}


SFT_FRAMES_V1: tuple[tuple[str, str], ...] = (
    ("complete_relation", "Complete the relation: {left} {connector_text}?"),
    ("fill_value", "Fill the missing value: {left} {connector_text}?"),
    ("what_completes", "What completes this relation: {left} {connector_text}?"),
    ("answer_entry", "Answer the entry: {left} {connector_text}?"),
    ("finish_line", "Finish the line: {left} {connector_text}?"),
    ("provide_value", "Provide the value for: {left} {connector_text}?"),
    ("lookup_value", "Look up the value for: {left} {connector_text}?"),
    ("complete_entry", "Complete this entry: {left} {connector_text}?"),
)


CHAT_TEMPLATE_SMOLLM2_CHATML_V1 = """{% for message in messages %}
{% if message["role"] == "assistant" %}
<|im_start|>assistant
{% generation %}{{ message["content"] }}<|im_end|>{% endgeneration %}
{% else %}
<|im_start|>{{ message["role"] }}
{{ message["content"] }}<|im_end|>
{% endif %}
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}
"""


def composition_manifest(
    *,
    generator_version: str,
    connector_version: str,
    pretrain_wrapper_version: str | None = None,
    pretrain_cause_order: PretrainCauseOrder = "canonical",
    sft_wrapper_version: str | None = None,
    chat_template_id: str | None = None,
) -> dict[str, Any]:
    pretrain_sources = (
        _pretrain_sources_for(pretrain_wrapper_version) if pretrain_wrapper_version else None
    )
    pretrain_frames = (
        _pretrain_frames_for(pretrain_wrapper_version) if pretrain_wrapper_version else None
    )
    return {
        "generator_version": generator_version,
        "connector_version": connector_version,
        "connectors": [
            {
                "connector_id": connector.connector_id,
                "forward_text": connector.forward_text,
                "reverse_text": connector.reverse_text,
            }
            for connector in CONNECTOR_V1
        ],
        "pretrain_wrapper_version": pretrain_wrapper_version,
        "pretrain_cause_order": pretrain_cause_order if pretrain_wrapper_version else None,
        "pretrain_key_space": (
            len(CONNECTOR_V1) * len(pretrain_sources) * len(pretrain_frames)
            if pretrain_sources and pretrain_frames
            else None
        ),
        "pretrain_sources": list(pretrain_sources) if pretrain_sources else None,
        "pretrain_frames": [
            {"wrapper_id": wrapper_id, "text": text}
            for wrapper_id, text in pretrain_frames
        ]
        if pretrain_frames
        else None,
        "sft_wrapper_version": sft_wrapper_version,
        "sft_key_space": len(CONNECTOR_V1) * len(SFT_FRAMES_V1) if sft_wrapper_version else None,
        "sft_frames": [
            {"wrapper_id": wrapper_id, "text": text}
            for wrapper_id, text in SFT_FRAMES_V1
        ]
        if sft_wrapper_version
        else None,
        "chat_template_id": chat_template_id,
    }


class CompositionGenerator:
    def __init__(
        self,
        *,
        generator_version: str = "composition_v1",
        connector_version: str = "connector_v1",
        pretrain_wrapper_version: str = "pretrain_descriptive_v1",
        pretrain_cause_order: PretrainCauseOrder = "canonical",
        sft_wrapper_version: str = "sft_chat_qa_v1",
        chat_template_id: str = "smollm2_chatml_v1",
    ) -> None:
        if generator_version != "composition_v1":
            raise ValueError(f"Unsupported composition.generator_version: {generator_version}")
        if connector_version != "connector_v1":
            raise ValueError(f"Unsupported composition.connector_version: {connector_version}")
        pretrain_sources = _pretrain_sources_for(pretrain_wrapper_version)
        pretrain_frames = _pretrain_frames_for(pretrain_wrapper_version)
        if sft_wrapper_version != "sft_chat_qa_v1":
            raise ValueError(f"Unsupported composition.sft_wrapper_version: {sft_wrapper_version}")
        if chat_template_id != "smollm2_chatml_v1":
            raise ValueError(f"Unsupported composition.chat_template_id: {chat_template_id}")
        if pretrain_cause_order not in {"canonical", "random_swap"}:
            raise ValueError(
                "Unsupported composition.pretrain_cause_order: "
                f"{pretrain_cause_order}"
            )

        self.generator_version = generator_version
        self.connector_version = connector_version
        self.pretrain_wrapper_version = pretrain_wrapper_version
        self.pretrain_cause_order = pretrain_cause_order
        self.sft_wrapper_version = sft_wrapper_version
        self.chat_template_id = chat_template_id
        self.pretrain_sources = pretrain_sources
        self.pretrain_frames = pretrain_frames
        self._counts: dict[tuple[str, Stage, Direction], int] = {}

    @property
    def pretrain_key_space_size(self) -> int:
        return len(CONNECTOR_V1) * len(self.pretrain_sources) * len(self.pretrain_frames)

    @property
    def sft_key_space_size(self) -> int:
        return len(CONNECTOR_V1) * len(SFT_FRAMES_V1)

    def manifest(self, *, include_pretrain: bool, include_sft: bool) -> dict[str, Any]:
        return composition_manifest(
            generator_version=self.generator_version,
            connector_version=self.connector_version,
            pretrain_wrapper_version=self.pretrain_wrapper_version if include_pretrain else None,
            pretrain_cause_order=self.pretrain_cause_order,
            sft_wrapper_version=self.sft_wrapper_version if include_sft else None,
            chat_template_id=self.chat_template_id if include_sft else None,
        )

    def compose_pretrain(
        self,
        relation: dict[str, Any],
        *,
        direction: Direction,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        left, right, answer_ids, cause_order = _pair_for_direction(
            relation,
            direction,
            cause_order=self.pretrain_cause_order,
            order_key={
                "stage": "pretrain",
                "relation_id": relation["effect_id"],
                "direction": direction,
                "record_index": record_index,
                "render_id": render_id,
            },
        )
        relation_text, connector, key_parts = self._pretrain_relation_text(
            relation_id=relation["effect_id"],
            direction=direction,
            left=left,
            right=right,
        )
        text = key_parts["frame_text"].format(source=key_parts["source_text"], relation=relation_text)
        template_key = self._template_key(
            stage="pretrain",
            relation_id=relation["effect_id"],
            direction=direction,
            connector_id=connector.connector_id,
            wrapper_id=key_parts["wrapper_id"],
            slots={
                "source_id": key_parts["source_id"],
                "source_text": key_parts["source_text"],
            },
        )
        metadata = self._metadata(
            relation=relation,
            stage="pretrain",
            split=split,
            direction=direction,
            connector=connector,
            connector_text=connector.text_for(direction),
            wrapper_id=key_parts["wrapper_id"],
            template_key=template_key,
            answer_text=right,
            answer_ids=answer_ids,
            rendered_cause_ids=[item["cause_id"] for item in cause_order],
            rendered_cause_order_policy=self.pretrain_cause_order,
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def compose_sft(
        self,
        relation: dict[str, Any],
        *,
        direction: Direction,
        qa_type: str,
        world_id: str,
        sft_render_id: str,
        split: str,
        sample_index: int,
    ) -> Composition:
        left, right, answer_ids, _ = _pair_for_direction(
            relation,
            direction,
            cause_order="canonical",
            order_key=None,
        )
        connector, wrapper_id, wrapper_text = self._sft_key(
            relation_id=relation["effect_id"],
            direction=direction,
        )
        connector_text = connector.text_for(direction)
        user_content = wrapper_text.format(left=left, connector_text=connector_text)
        messages = [
            {"role": "user", "content": _normalize_text(user_content)},
            {"role": "assistant", "content": right},
        ]
        template_key = self._template_key(
            stage="sft",
            relation_id=relation["effect_id"],
            direction=direction,
            connector_id=connector.connector_id,
            wrapper_id=wrapper_id,
            slots={},
        )
        metadata = self._metadata(
            relation=relation,
            stage="sft",
            split=split,
            direction=direction,
            connector=connector,
            connector_text=connector_text,
            wrapper_id=wrapper_id,
            template_key=template_key,
            answer_text=right,
            answer_ids=answer_ids,
        )
        metadata["world_id"] = world_id
        metadata["sft_render_id"] = sft_render_id
        metadata["qa_type"] = qa_type
        metadata["sample_index"] = sample_index
        metadata["chat_template_id"] = self.chat_template_id
        return Composition(text=None, messages=messages, metadata=metadata)

    def _pretrain_relation_text(
        self,
        *,
        relation_id: str,
        direction: Direction,
        left: str,
        right: str,
    ) -> tuple[str, Connector, dict[str, str]]:
        key_index = self._next_key_index(
            relation_id=relation_id,
            stage="pretrain",
            direction=direction,
            key_space_size=self.pretrain_key_space_size,
        )
        connector_index = key_index % len(CONNECTOR_V1)
        frame_index = (key_index // len(CONNECTOR_V1)) % len(self.pretrain_frames)
        source_index = (key_index // (len(CONNECTOR_V1) * len(self.pretrain_frames))) % len(
            self.pretrain_sources
        )
        connector = CONNECTOR_V1[connector_index]
        wrapper_id, frame_text = self.pretrain_frames[frame_index]
        source_text = self.pretrain_sources[source_index]
        relation_text = f"{left} {connector.text_for(direction)} {right}"
        return relation_text, connector, {
            "wrapper_id": wrapper_id,
            "frame_text": frame_text,
            "source_id": f"source_{source_index:03d}",
            "source_text": source_text,
        }

    def _sft_key(self, *, relation_id: str, direction: Direction) -> tuple[Connector, str, str]:
        key_index = self._next_key_index(
            relation_id=relation_id,
            stage="sft",
            direction=direction,
            key_space_size=self.sft_key_space_size,
        )
        connector_index = key_index % len(CONNECTOR_V1)
        frame_index = (key_index // len(CONNECTOR_V1)) % len(SFT_FRAMES_V1)
        connector = CONNECTOR_V1[connector_index]
        wrapper_id, wrapper_text = SFT_FRAMES_V1[frame_index]
        return connector, wrapper_id, wrapper_text

    def _next_key_index(
        self,
        *,
        relation_id: str,
        stage: Stage,
        direction: Direction,
        key_space_size: int,
    ) -> int:
        counter_key = (relation_id, stage, direction)
        count = self._counts.get(counter_key, 0)
        if count >= key_space_size:
            raise ValueError(
                "Composition key space exhausted for "
                f"relation={relation_id}, stage={stage}, direction={direction}; "
                f"count={count}, key_space={key_space_size}"
            )
        self._counts[counter_key] = count + 1
        start = _stable_int({"relation_id": relation_id, "stage": stage, "direction": direction})
        return (start + count) % key_space_size

    def _template_key(
        self,
        *,
        stage: Stage,
        relation_id: str,
        direction: Direction,
        connector_id: str,
        wrapper_id: str,
        slots: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "generator_version": self.generator_version,
            "connector_version": self.connector_version,
            "stage": stage,
            "relation_id": relation_id,
            "direction": direction,
            "connector_id": connector_id,
            "wrapper_id": wrapper_id,
            "slots": slots,
        }

    def _metadata(
        self,
        *,
        relation: dict[str, Any],
        stage: Stage,
        split: str,
        direction: Direction,
        connector: Connector,
        connector_text: str,
        wrapper_id: str,
        template_key: dict[str, Any],
        answer_text: str,
        answer_ids: list[str],
        rendered_cause_ids: list[str] | None = None,
        rendered_cause_order_policy: str | None = None,
    ) -> dict[str, Any]:
        metadata = {
            "stage": stage,
            "split": split,
            "relation_id": relation["effect_id"],
            "effect_id": relation["effect_id"],
            "effect_surface": relation["effect_surface"],
            "cause_ids": list(relation["recipe_cause_ids"]),
            "partition": relation["partition"],
            "direction": direction,
            "connector_id": connector.connector_id,
            "connector_text": connector_text,
            "wrapper_id": wrapper_id,
            "template_key_hash": canonical_json_sha256(template_key)[:16],
            "answer_text": answer_text,
            "answer_ids": answer_ids,
        }
        if rendered_cause_ids is not None:
            metadata["rendered_cause_ids"] = rendered_cause_ids
            metadata["rendered_cause_order_policy"] = rendered_cause_order_policy
            metadata["rendered_cause_order"] = (
                "canonical"
                if rendered_cause_ids == list(relation["recipe_cause_ids"])
                else "swapped"
            )
        return metadata


def chat_template_text(chat_template_id: str = "smollm2_chatml_v1") -> str:
    if chat_template_id != "smollm2_chatml_v1":
        raise ValueError(f"Unsupported chat_template_id: {chat_template_id}")
    return CHAT_TEMPLATE_SMOLLM2_CHATML_V1


def _pretrain_sources_for(pretrain_wrapper_version: str) -> tuple[str, ...]:
    try:
        return PRETRAIN_SOURCES_BY_VERSION[pretrain_wrapper_version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported composition.pretrain_wrapper_version: {pretrain_wrapper_version}"
        ) from exc


def _pretrain_frames_for(pretrain_wrapper_version: str) -> tuple[tuple[str, str], ...]:
    try:
        return PRETRAIN_FRAMES_BY_VERSION[pretrain_wrapper_version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported composition.pretrain_wrapper_version: {pretrain_wrapper_version}"
        ) from exc


def _pair_for_direction(
    relation: dict[str, Any],
    direction: Direction,
    *,
    cause_order: PretrainCauseOrder,
    order_key: dict[str, Any] | None,
) -> tuple[str, str, list[str], list[dict[str, Any]]]:
    ordered_recipe = _ordered_recipe(relation, cause_order=cause_order, order_key=order_key)
    causes = ", ".join(item["surface"] for item in ordered_recipe)
    if direction == "forward":
        return causes, relation["effect_surface"], [relation["effect_id"]], ordered_recipe
    if direction == "reverse":
        return (
            relation["effect_surface"],
            causes,
            [item["cause_id"] for item in ordered_recipe],
            ordered_recipe,
        )
    raise ValueError(f"Unsupported direction: {direction}")


def _ordered_recipe(
    relation: dict[str, Any],
    *,
    cause_order: PretrainCauseOrder,
    order_key: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    recipe = list(relation["recipe"])
    if cause_order == "canonical" or len(recipe) < 2:
        return recipe
    if cause_order != "random_swap":
        raise ValueError(f"Unsupported composition.pretrain_cause_order: {cause_order}")
    if order_key is None:
        raise ValueError("random_swap pretrain cause order requires an order_key")

    swap_seed = _stable_int(
        {
            "cause_order": cause_order,
            "order_key": order_key,
        }
    )
    if swap_seed % 2 == 0:
        return recipe

    first_index = (swap_seed // 2) % len(recipe)
    second_index = (swap_seed // (2 * len(recipe))) % (len(recipe) - 1)
    if second_index >= first_index:
        second_index += 1
    recipe[first_index], recipe[second_index] = recipe[second_index], recipe[first_index]
    return recipe


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _stable_int(payload: Any) -> int:
    return int(canonical_json_sha256(payload)[:16], 16)
