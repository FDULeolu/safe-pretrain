from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from safe_pretrain.synthetic.io import canonical_json_sha256


Direction = Literal["forward", "reverse"]
PretrainPattern = str
BidirectionalOrder = Literal["forward_first", "reverse_first"]
Stage = Literal["pretrain", "sft"]
PretrainCauseOrder = Literal["canonical", "random_swap"]

MIRROR_FORWARD_PATTERN = "mirror_forward"


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


@dataclass(frozen=True)
class RenderedEntity:
    text: str
    entity_type: str
    surface_type: str
    ids: list[str]
    rendered_cause_ids: list[str] | None = None


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

CONNECTOR_COMPOSABLE_V1: tuple[Connector, ...] = (
    Connector(
        "mapping_path",
        "maps forward to outcome",
        "maps backward to causes",
    ),
)

CONNECTORS_BY_VERSION: dict[str, tuple[Connector, ...]] = {
    "connector_v1": CONNECTOR_V1,
    "connector_composable_v1": CONNECTOR_COMPOSABLE_V1,
}


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

_PRETRAIN_SOURCE_PREFIXES_V3: tuple[str, ...] = (
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
    "mapping",
    "routing",
    "pathway",
    "bridge",
    "trace",
    "linkage",
    "operator",
    "relation",
)

_PRETRAIN_SOURCE_NOUNS_V3: tuple[str, ...] = (
    "note",
    "row",
    "card",
    "page",
    "sheet",
    "line",
    "entry",
    "memo",
    "log",
    "item",
    "record",
    "slip",
    "register",
    "file",
    "table",
    "index",
)

PRETRAIN_SOURCES_V3: tuple[str, ...] = tuple(
    f"{prefix} {noun}"
    for prefix in _PRETRAIN_SOURCE_PREFIXES_V3
    for noun in _PRETRAIN_SOURCE_NOUNS_V3
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

_PRETRAIN_FRAME_SUBJECTS_V3: tuple[tuple[str, str], ...] = (
    ("source", "The {source}"),
    ("line", "A line in the {source}"),
    ("item", "An item in the {source}"),
    ("row", "A row on the {source}"),
    ("marked_line", "The marked line in the {source}"),
    ("listed_item", "The listed item in the {source}"),
    ("stored_item", "A stored item from the {source}"),
    ("noted_item", "The noted item on the {source}"),
    ("saved_line", "A saved line from the {source}"),
    ("indexed_entry", "An indexed entry in the {source}"),
    ("filed_row", "The filed row in the {source}"),
    ("checked_item", "The checked item from the {source}"),
)

_PRETRAIN_FRAME_PREDICATES_V3: tuple[tuple[str, str], ...] = (
    ("says", "says {relation}."),
    ("states", "states {relation}."),
    ("records", "records {relation}."),
    ("shows", "shows {relation}."),
    ("notes", "notes {relation}."),
    ("lists", "lists {relation}."),
    ("stores", "stores {relation}."),
    ("marks", "marks {relation}."),
    ("gives_line", "gives this line: {relation}."),
    ("contains_item", "contains this item: {relation}."),
    ("has_entry", "has the entry {relation}."),
    ("presents", "presents {relation}."),
)

PRETRAIN_FRAMES_V3: tuple[tuple[str, str], ...] = tuple(
    (f"{subject_id}_{predicate_id}", f"{subject_text} {predicate_text}")
    for subject_id, subject_text in _PRETRAIN_FRAME_SUBJECTS_V3
    for predicate_id, predicate_text in _PRETRAIN_FRAME_PREDICATES_V3
)


PRETRAIN_SOURCES_BY_VERSION: dict[str, tuple[str, ...]] = {
    "pretrain_descriptive_v1": PRETRAIN_SOURCES_V1,
    "pretrain_descriptive_v2": PRETRAIN_SOURCES_V2,
    "pretrain_descriptive_v3": PRETRAIN_SOURCES_V3,
}


PRETRAIN_FRAMES_BY_VERSION: dict[str, tuple[tuple[str, str], ...]] = {
    "pretrain_descriptive_v1": PRETRAIN_FRAMES_V1,
    "pretrain_descriptive_v2": PRETRAIN_FRAMES_V2,
    "pretrain_descriptive_v3": PRETRAIN_FRAMES_V3,
}


IDENTITY_TEXT_BY_CONNECTOR_ID: dict[str, str] = {
    "listed_with": "is listed as the same entry as",
    "recorded_with": "is recorded as the same entry as",
    "associated_with": "is associated with the same entry as",
    "linked_to": "is linked as the same entry as",
    "maps_to": "maps as the same entry as",
    "corresponds_to": "corresponds to the same entry as",
    "paired_with": "is paired as the same entry as",
    "matched_with": "is matched as the same entry as",
}


COMPOSED_TEXT_BY_CONNECTOR_ID: dict[str, tuple[str, str]] = {
    "listed_with": (
        "is listed through outcome then causes as",
        "is listed through causes then outcome as",
    ),
    "recorded_with": (
        "is recorded through result then cause set as",
        "is recorded through cause set then result as",
    ),
    "associated_with": (
        "is associated through outcome then source items as",
        "is associated through source items then outcome as",
    ),
    "linked_to": (
        "is linked through result then causes as",
        "is linked through causes then result as",
    ),
    "maps_to": (
        "maps through outcome then causes as",
        "maps through causes then outcome as",
    ),
    "corresponds_to": (
        "corresponds through result then cause set as",
        "corresponds through cause set then result as",
    ),
    "paired_with": (
        "is paired through outcome then causes as",
        "is paired through causes then outcome as",
    ),
    "matched_with": (
        "is matched through result then source items as",
        "is matched through source items then result as",
    ),
}


PRETRAIN_OPERATOR_PATHS: dict[str, tuple[str, ...]] = {
    "forward": ("F",),
    "reverse": ("R",),
    "identity": ("I",),
    "forward_reverse": ("F", "R"),
    "reverse_forward": ("R", "F"),
}

OPERATOR_NAME_TO_SYMBOL: dict[str, str] = {
    "forward": "F",
    "reverse": "R",
    "identity": "I",
}

OPERATOR_SYMBOL_TO_NAME: dict[str, str] = {
    symbol: name for name, symbol in OPERATOR_NAME_TO_SYMBOL.items()
}

OPERATOR_DOMAIN_RANGE: dict[str, tuple[str, str] | None] = {
    "F": ("cause", "effect"),
    "R": ("effect", "cause"),
    "I": None,
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
    pretrain_cause_order: PretrainCauseOrder = "random_swap",
    pretrain_alias_enabled: bool = False,
    pretrain_alias_replacement_probability: float = 0.0,
    pretrain_answer_alias_replacement_probability: float = 0.0,
    bidirectional_order_weights: dict[str, float] | None = None,
    sft_wrapper_version: str | None = None,
    chat_template_id: str | None = None,
) -> dict[str, Any]:
    connectors = _connectors_for(connector_version)
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
            for connector in connectors
        ],
        "pretrain_wrapper_version": pretrain_wrapper_version,
        "pretrain_cause_order": pretrain_cause_order if pretrain_wrapper_version else None,
        "pretrain_alias_enabled": pretrain_alias_enabled if pretrain_wrapper_version else None,
        "pretrain_alias_replacement_probability": (
            pretrain_alias_replacement_probability if pretrain_wrapper_version else None
        ),
        "pretrain_answer_alias_replacement_probability": (
            pretrain_answer_alias_replacement_probability if pretrain_wrapper_version else None
        ),
        "pretrain_operator_patterns": dict(PRETRAIN_OPERATOR_PATHS)
        if pretrain_wrapper_version
        else None,
        "pretrain_special_patterns": [MIRROR_FORWARD_PATTERN]
        if pretrain_wrapper_version
        else None,
        "pretrain_operator_grammar": {
            "operator_names": dict(OPERATOR_NAME_TO_SYMBOL),
            "typed_domains": {
                "F": {"from": "A", "to": "B"},
                "R": {"from": "B", "to": "A"},
                "I": {"from": "current", "to": "current"},
            },
            "dynamic_pattern_separator": "_",
        }
        if pretrain_wrapper_version
        else None,
        "bidirectional_order_weights": (
            dict(bidirectional_order_weights or _default_bidirectional_order_weights())
            if pretrain_wrapper_version
            else None
        ),
        "pretrain_key_space": (
            len(connectors) * len(pretrain_sources) * len(pretrain_frames)
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
        "sft_key_space": len(connectors) * len(SFT_FRAMES_V1) if sft_wrapper_version else None,
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
        pretrain_cause_order: PretrainCauseOrder = "random_swap",
        pretrain_alias_enabled: bool = False,
        pretrain_alias_replacement_probability: float = 0.0,
        pretrain_answer_alias_replacement_probability: float = 0.0,
        bidirectional_order_weights: dict[str, float] | None = None,
        sft_wrapper_version: str = "sft_chat_qa_v1",
        chat_template_id: str = "smollm2_chatml_v1",
    ) -> None:
        if generator_version != "composition_v1":
            raise ValueError(f"Unsupported composition.generator_version: {generator_version}")
        connectors = _connectors_for(connector_version)
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
        self.pretrain_alias_enabled = bool(pretrain_alias_enabled)
        self.pretrain_alias_replacement_probability = _validate_probability(
            pretrain_alias_replacement_probability,
            name="composition.pretrain_alias_replacement_probability",
        )
        self.pretrain_answer_alias_replacement_probability = _validate_probability(
            pretrain_answer_alias_replacement_probability,
            name="composition.pretrain_answer_alias_replacement_probability",
        )
        self.bidirectional_order_weights = _validate_bidirectional_order_weights(
            bidirectional_order_weights
        )
        self.sft_wrapper_version = sft_wrapper_version
        self.chat_template_id = chat_template_id
        self.pretrain_sources = pretrain_sources
        self.pretrain_frames = pretrain_frames
        self.connectors = connectors
        self._counts: dict[tuple[str, Stage, str], int] = {}

    @property
    def pretrain_key_space_size(self) -> int:
        return len(self.connectors) * len(self.pretrain_sources) * len(self.pretrain_frames)

    @property
    def sft_key_space_size(self) -> int:
        return len(self.connectors) * len(SFT_FRAMES_V1)

    def manifest(self, *, include_pretrain: bool, include_sft: bool) -> dict[str, Any]:
        return composition_manifest(
            generator_version=self.generator_version,
            connector_version=self.connector_version,
            pretrain_wrapper_version=self.pretrain_wrapper_version if include_pretrain else None,
            pretrain_cause_order=self.pretrain_cause_order,
            pretrain_alias_enabled=self.pretrain_alias_enabled,
            pretrain_alias_replacement_probability=self.pretrain_alias_replacement_probability,
            pretrain_answer_alias_replacement_probability=(
                self.pretrain_answer_alias_replacement_probability
            ),
            bidirectional_order_weights=self.bidirectional_order_weights,
            sft_wrapper_version=self.sft_wrapper_version if include_sft else None,
            chat_template_id=self.chat_template_id if include_sft else None,
        )

    def compose_pretrain(
        self,
        relation: dict[str, Any],
        *,
        direction: PretrainPattern,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        if direction == "bidirectional":
            return self._compose_bidirectional_pretrain(
                relation,
                world_id=world_id,
                render_id=render_id,
                split=split,
                record_index=record_index,
            )
        if direction == MIRROR_FORWARD_PATTERN:
            return self._compose_mirror_forward_pretrain(
                relation,
                world_id=world_id,
                render_id=render_id,
                split=split,
                record_index=record_index,
            )
        if not is_pretrain_operator_pattern(direction):
            raise ValueError(f"Unsupported pretrain pattern: {direction}")
        if self.connector_version == "connector_composable_v1":
            return self._compose_composable_operator_pretrain(
                relation,
                pattern=direction,
                world_id=world_id,
                render_id=render_id,
                split=split,
                record_index=record_index,
            )
        if direction not in PRETRAIN_OPERATOR_PATHS:
            raise ValueError(
                f"Pretrain pattern {direction!r} requires connector_composable_v1"
            )
        if self.pretrain_alias_enabled or direction not in {"forward", "reverse"}:
            return self._compose_operator_pretrain(
                relation,
                pattern=direction,
                world_id=world_id,
                render_id=render_id,
                split=split,
                record_index=record_index,
            )

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
            pretrain_pattern=direction,
            rendered_cause_ids=[item["cause_id"] for item in cause_order],
            rendered_cause_order_policy=self.pretrain_cause_order,
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        self._add_operator_metadata(
            metadata,
            pattern=direction,
            start_entity_type="cause" if direction == "forward" else "effect",
            target_entity_type="effect" if direction == "forward" else "cause",
            entity_surfaces={},
        )
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def _compose_mirror_forward_pretrain(
        self,
        relation: dict[str, Any],
        *,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        pattern = MIRROR_FORWARD_PATTERN
        connector, key_parts = self._pretrain_key(
            relation_id=relation["effect_id"],
            direction=pattern,
        )
        context = {
            "stage": "pretrain",
            "relation_id": relation["effect_id"],
            "direction": pattern,
            "record_index": record_index,
            "render_id": render_id,
        }
        effect = self._render_entity(
            relation,
            entity_type="effect",
            role="start",
            pattern=pattern,
            context=context,
            answer=False,
        )
        causes = self._render_entity(
            relation,
            entity_type="cause",
            role="target",
            pattern=pattern,
            context=context,
            answer=True,
        )
        connector_text = _mirror_connector_text(connector.forward_text)
        relation_text = f"{effect.text} {connector_text} {causes.text}"
        text = key_parts["frame_text"].format(source=key_parts["source_text"], relation=relation_text)
        template_key = self._template_key(
            stage="pretrain",
            relation_id=relation["effect_id"],
            direction=pattern,
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
            direction=pattern,
            connector=connector,
            connector_text=connector_text,
            wrapper_id=key_parts["wrapper_id"],
            template_key=template_key,
            answer_text=causes.text,
            answer_ids=causes.ids,
            pretrain_pattern=pattern,
            rendered_cause_ids=causes.rendered_cause_ids,
            rendered_cause_order_policy=self.pretrain_cause_order,
            rendered_cause_span_ids=[causes.rendered_cause_ids]
            if causes.rendered_cause_ids is not None
            else None,
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        metadata["mirror_pattern"] = "entity_preserved_token_mirror_forward"
        metadata["normal_reverse"] = False
        metadata["exposes_reverse_gradient"] = True
        self._add_operator_metadata(
            metadata,
            pattern=pattern,
            start_entity_type=effect.entity_type,
            target_entity_type=causes.entity_type,
            entity_surfaces={"start": effect, "target": causes},
            operator_path=("MF",),
        )
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def _compose_operator_pretrain(
        self,
        relation: dict[str, Any],
        *,
        pattern: str,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        connector, key_parts = self._pretrain_key(
            relation_id=relation["effect_id"],
            direction=pattern,
        )
        context = {
            "stage": "pretrain",
            "relation_id": relation["effect_id"],
            "direction": pattern,
            "record_index": record_index,
            "render_id": render_id,
        }
        relation_text: str
        connector_text: str
        bidirectional_order: str | None = None
        entities: dict[str, RenderedEntity]
        if pattern == "forward":
            entities = {
                "start": self._render_entity(
                    relation,
                    entity_type="cause",
                    role="start",
                    pattern=pattern,
                    context=context,
                    answer=False,
                ),
                "target": self._render_entity(
                    relation,
                    entity_type="effect",
                    role="target",
                    pattern=pattern,
                    context=context,
                    answer=True,
                ),
            }
            relation_text = (
                f"{entities['start'].text} {connector.forward_text} {entities['target'].text}"
            )
            connector_text = connector.forward_text
        elif pattern == "reverse":
            entities = {
                "start": self._render_entity(
                    relation,
                    entity_type="effect",
                    role="start",
                    pattern=pattern,
                    context=context,
                    answer=False,
                ),
                "target": self._render_entity(
                    relation,
                    entity_type="cause",
                    role="target",
                    pattern=pattern,
                    context=context,
                    answer=True,
                ),
            }
            relation_text = (
                f"{entities['start'].text} {connector.reverse_text} {entities['target'].text}"
            )
            connector_text = connector.reverse_text
        elif pattern == "identity":
            entity_type = _identity_entity_type(context)
            identity_text = _identity_text(connector)
            entities = {
                "start": self._render_entity(
                    relation,
                    entity_type=entity_type,
                    role="start",
                    pattern=pattern,
                    context=context,
                    answer=False,
                ),
                "target": self._render_entity(
                    relation,
                    entity_type=entity_type,
                    role="target",
                    pattern=pattern,
                    context=context,
                    answer=True,
                ),
            }
            relation_text = (
                f"{entities['start'].text} {identity_text} {entities['target'].text}"
            )
            connector_text = identity_text
        elif pattern == "forward_reverse":
            forward_reverse_text, _ = _composed_text(connector)
            entities = {
                "start": self._render_entity(
                    relation,
                    entity_type="cause",
                    role="start",
                    pattern=pattern,
                    context=context,
                    answer=False,
                ),
                "target": self._render_entity(
                    relation,
                    entity_type="cause",
                    role="target",
                    pattern=pattern,
                    context=context,
                    answer=True,
                ),
            }
            relation_text = (
                f"{entities['start'].text} {forward_reverse_text} {entities['target'].text}"
            )
            connector_text = f"{connector.forward_text} | {connector.reverse_text}"
        elif pattern == "reverse_forward":
            _, reverse_forward_text = _composed_text(connector)
            entities = {
                "start": self._render_entity(
                    relation,
                    entity_type="effect",
                    role="start",
                    pattern=pattern,
                    context=context,
                    answer=False,
                ),
                "target": self._render_entity(
                    relation,
                    entity_type="effect",
                    role="target",
                    pattern=pattern,
                    context=context,
                    answer=True,
                ),
            }
            relation_text = (
                f"{entities['start'].text} {reverse_forward_text} {entities['target'].text}"
            )
            connector_text = f"{connector.reverse_text} | {connector.forward_text}"
        else:
            raise ValueError(f"Unsupported pretrain pattern: {pattern}")

        text = key_parts["frame_text"].format(source=key_parts["source_text"], relation=relation_text)
        template_key = self._template_key(
            stage="pretrain",
            relation_id=relation["effect_id"],
            direction=pattern,
            connector_id=connector.connector_id,
            wrapper_id=key_parts["wrapper_id"],
            slots={
                "source_id": key_parts["source_id"],
                "source_text": key_parts["source_text"],
            },
        )
        target = entities["target"]
        cause_spans = [
            entity.rendered_cause_ids
            for entity in entities.values()
            if entity.rendered_cause_ids is not None
        ]
        rendered_cause_ids = target.rendered_cause_ids or (cause_spans[0] if cause_spans else None)
        metadata = self._metadata(
            relation=relation,
            stage="pretrain",
            split=split,
            direction=pattern,
            connector=connector,
            connector_text=connector_text,
            wrapper_id=key_parts["wrapper_id"],
            template_key=template_key,
            answer_text=target.text,
            answer_ids=target.ids,
            pretrain_pattern=pattern,
            bidirectional_order=bidirectional_order,
            rendered_cause_ids=rendered_cause_ids,
            rendered_cause_order_policy=self.pretrain_cause_order
            if rendered_cause_ids is not None
            else None,
            rendered_cause_span_ids=[span for span in cause_spans if span is not None],
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        self._add_operator_metadata(
            metadata,
            pattern=pattern,
            start_entity_type=entities["start"].entity_type,
            target_entity_type=target.entity_type,
            entity_surfaces=entities,
        )
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def _compose_composable_operator_pretrain(
        self,
        relation: dict[str, Any],
        *,
        pattern: str,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        path = operator_path_for_pretrain_pattern(pattern)
        connector, key_parts = self._pretrain_key(
            relation_id=relation["effect_id"],
            direction=pattern,
        )
        context = {
            "stage": "pretrain",
            "relation_id": relation["effect_id"],
            "direction": pattern,
            "record_index": record_index,
            "render_id": render_id,
        }
        identity_entity_type = _identity_entity_type(context)
        start_entity_type, target_entity_type = _operator_path_entity_types(
            path,
            identity_entity_type=identity_entity_type,
        )
        entities = {
            "start": self._render_entity(
                relation,
                entity_type=start_entity_type,
                role="start",
                pattern=pattern,
                context=context,
                answer=False,
            ),
            "target": self._render_entity(
                relation,
                entity_type=target_entity_type,
                role="target",
                pattern=pattern,
                context=context,
                answer=True,
            ),
        }
        relation_text, connector_text = _composable_relation_text(
            connector=connector,
            start=entities["start"].text,
            target=entities["target"].text,
            path=path,
            start_entity_type=start_entity_type,
        )

        text = key_parts["frame_text"].format(source=key_parts["source_text"], relation=relation_text)
        template_key = self._template_key(
            stage="pretrain",
            relation_id=relation["effect_id"],
            direction=pattern,
            connector_id=connector.connector_id,
            wrapper_id=key_parts["wrapper_id"],
            slots={
                "source_id": key_parts["source_id"],
                "source_text": key_parts["source_text"],
            },
        )
        target = entities["target"]
        cause_spans = [
            entity.rendered_cause_ids
            for entity in entities.values()
            if entity.rendered_cause_ids is not None
        ]
        rendered_cause_ids = target.rendered_cause_ids or (cause_spans[0] if cause_spans else None)
        metadata = self._metadata(
            relation=relation,
            stage="pretrain",
            split=split,
            direction=pattern,
            connector=connector,
            connector_text=connector_text,
            wrapper_id=key_parts["wrapper_id"],
            template_key=template_key,
            answer_text=target.text,
            answer_ids=target.ids,
            pretrain_pattern=pattern,
            rendered_cause_ids=rendered_cause_ids,
            rendered_cause_order_policy=self.pretrain_cause_order
            if rendered_cause_ids is not None
            else None,
            rendered_cause_span_ids=[span for span in cause_spans if span is not None],
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        self._add_operator_metadata(
            metadata,
            pattern=pattern,
            start_entity_type=entities["start"].entity_type,
            target_entity_type=target.entity_type,
            entity_surfaces=entities,
            operator_path=path,
        )
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def _compose_bidirectional_pretrain(
        self,
        relation: dict[str, Any],
        *,
        world_id: str,
        render_id: str,
        split: str,
        record_index: int,
    ) -> Composition:
        order = _bidirectional_order(
            self.bidirectional_order_weights,
            {
                "stage": "pretrain",
                "relation_id": relation["effect_id"],
                "direction": "bidirectional",
                "record_index": record_index,
                "render_id": render_id,
            },
        )
        connector, key_parts = self._pretrain_key(
            relation_id=relation["effect_id"],
            direction="bidirectional",
        )
        effect = relation["effect_surface"]
        if order == "forward_first":
            first_causes = _ordered_recipe(
                relation,
                cause_order=self.pretrain_cause_order,
                order_key={
                    "stage": "pretrain",
                    "relation_id": relation["effect_id"],
                    "direction": "bidirectional",
                    "record_index": record_index,
                    "render_id": render_id,
                    "span": "first",
                },
            )
            final_causes = _ordered_recipe(
                relation,
                cause_order=self.pretrain_cause_order,
                order_key={
                    "stage": "pretrain",
                    "relation_id": relation["effect_id"],
                    "direction": "bidirectional",
                    "record_index": record_index,
                    "render_id": render_id,
                    "span": "final",
                },
            )
            first_text = _cause_text(first_causes)
            final_text = _cause_text(final_causes)
            relation_text = (
                f"{first_text} {connector.forward_text} {effect} "
                f"{connector.reverse_text} {final_text}"
            )
            answer_text = final_text
            answer_ids = [item["cause_id"] for item in final_causes]
            rendered_cause_ids = answer_ids
            rendered_cause_spans = [
                [item["cause_id"] for item in first_causes],
                [item["cause_id"] for item in final_causes],
            ]
        else:
            causes = _ordered_recipe(
                relation,
                cause_order=self.pretrain_cause_order,
                order_key={
                    "stage": "pretrain",
                    "relation_id": relation["effect_id"],
                    "direction": "bidirectional",
                    "record_index": record_index,
                    "render_id": render_id,
                    "span": "middle",
                },
            )
            cause_text = _cause_text(causes)
            relation_text = (
                f"{effect} {connector.reverse_text} {cause_text} "
                f"{connector.forward_text} {effect}"
            )
            answer_text = effect
            answer_ids = [relation["effect_id"]]
            rendered_cause_ids = [item["cause_id"] for item in causes]
            rendered_cause_spans = [rendered_cause_ids]

        text = key_parts["frame_text"].format(source=key_parts["source_text"], relation=relation_text)
        template_key = self._template_key(
            stage="pretrain",
            relation_id=relation["effect_id"],
            direction="bidirectional",
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
            direction="bidirectional",
            connector=connector,
            connector_text=f"{connector.forward_text} | {connector.reverse_text}",
            wrapper_id=key_parts["wrapper_id"],
            template_key=template_key,
            answer_text=answer_text,
            answer_ids=answer_ids,
            pretrain_pattern="bidirectional",
            bidirectional_order=order,
            rendered_cause_ids=rendered_cause_ids,
            rendered_cause_order_policy=self.pretrain_cause_order,
            rendered_cause_span_ids=rendered_cause_spans,
        )
        metadata["render_id"] = render_id
        metadata["world_id"] = world_id
        metadata["record_index"] = record_index
        self._add_operator_metadata(
            metadata,
            pattern="bidirectional",
            start_entity_type="cause" if order == "forward_first" else "effect",
            target_entity_type="cause" if order == "forward_first" else "effect",
            entity_surfaces={},
            operator_path=("F", "R") if order == "forward_first" else ("R", "F"),
        )
        return Composition(text=_normalize_text(text), messages=None, metadata=metadata)

    def _render_entity(
        self,
        relation: dict[str, Any],
        *,
        entity_type: str,
        role: str,
        pattern: str,
        context: dict[str, Any],
        answer: bool,
    ) -> RenderedEntity:
        if entity_type == "cause":
            ordered_recipe = _ordered_recipe(
                relation,
                cause_order=self.pretrain_cause_order,
                order_key={**context, "role": role},
            )
            canonical_text = _cause_text(ordered_recipe)
            ids = [item["cause_id"] for item in ordered_recipe]
            rendered_cause_ids: list[str] | None = ids
        elif entity_type == "effect":
            canonical_text = relation["effect_surface"]
            ids = [relation["effect_id"]]
            rendered_cause_ids = None
        else:
            raise ValueError(f"Unsupported pretrain entity type: {entity_type}")

        probability = (
            self.pretrain_answer_alias_replacement_probability
            if answer
            else self.pretrain_alias_replacement_probability
        )
        use_alias = self.pretrain_alias_enabled and probability > 0.0 and _stable_probability(
            {
                "alias": {
                    "relation_id": relation["effect_id"],
                    "entity_type": entity_type,
                    "role": role,
                    "pattern": pattern,
                    "record_index": context["record_index"],
                    "render_id": context["render_id"],
                    "answer": answer,
                }
            }
        ) < probability
        if use_alias:
            if entity_type == "cause":
                ids = list(relation["recipe_cause_ids"])
                rendered_cause_ids = ids
            return RenderedEntity(
                text=_entity_alias(relation, entity_type),
                entity_type=entity_type,
                surface_type="alias",
                ids=ids,
                rendered_cause_ids=rendered_cause_ids,
            )
        return RenderedEntity(
            text=canonical_text,
            entity_type=entity_type,
            surface_type="canonical",
            ids=ids,
            rendered_cause_ids=rendered_cause_ids,
        )

    def _add_operator_metadata(
        self,
        metadata: dict[str, Any],
        *,
        pattern: str,
        start_entity_type: str,
        target_entity_type: str,
        entity_surfaces: dict[str, RenderedEntity],
        operator_path: tuple[str, ...] | None = None,
    ) -> None:
        path = tuple(operator_path or PRETRAIN_OPERATOR_PATHS[pattern])
        metadata["operator_path"] = list(path)
        metadata["operator_count"] = len(path)
        metadata["start_entity_type"] = _symbolic_entity_type(start_entity_type)
        metadata["target_entity_type"] = _symbolic_entity_type(target_entity_type)
        metadata["pretrain_alias_enabled"] = self.pretrain_alias_enabled
        metadata["pretrain_alias_replacement_probability"] = (
            self.pretrain_alias_replacement_probability
        )
        metadata["pretrain_answer_alias_replacement_probability"] = (
            self.pretrain_answer_alias_replacement_probability
        )
        if not entity_surfaces:
            metadata["alias_replacement_count"] = 0
            metadata["entity_surface_types"] = {}
            return
        metadata["alias_replacement_count"] = sum(
            1 for entity in entity_surfaces.values() if entity.surface_type == "alias"
        )
        metadata["entity_surface_types"] = {
            role: entity.surface_type for role, entity in entity_surfaces.items()
        }

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
        connector, key_parts = self._pretrain_key(
            relation_id=relation_id,
            direction=direction,
        )
        relation_text = f"{left} {connector.text_for(direction)} {right}"
        return relation_text, connector, key_parts

    def _pretrain_key(
        self,
        *,
        relation_id: str,
        direction: str,
    ) -> tuple[Connector, dict[str, str]]:
        key_index = self._next_key_index(
            relation_id=relation_id,
            stage="pretrain",
            direction=direction,
            key_space_size=self.pretrain_key_space_size,
        )
        connector_index = key_index % len(self.connectors)
        frame_index = (key_index // len(self.connectors)) % len(self.pretrain_frames)
        source_index = (key_index // (len(self.connectors) * len(self.pretrain_frames))) % len(
            self.pretrain_sources
        )
        connector = self.connectors[connector_index]
        wrapper_id, frame_text = self.pretrain_frames[frame_index]
        source_text = self.pretrain_sources[source_index]
        return connector, {
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
        connector_index = key_index % len(self.connectors)
        frame_index = (key_index // len(self.connectors)) % len(SFT_FRAMES_V1)
        connector = self.connectors[connector_index]
        wrapper_id, wrapper_text = SFT_FRAMES_V1[frame_index]
        return connector, wrapper_id, wrapper_text

    def _next_key_index(
        self,
        *,
        relation_id: str,
        stage: Stage,
        direction: str,
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
        direction: str,
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
        direction: str,
        connector: Connector,
        connector_text: str,
        wrapper_id: str,
        template_key: dict[str, Any],
        answer_text: str,
        answer_ids: list[str],
        pretrain_pattern: str | None = None,
        bidirectional_order: str | None = None,
        rendered_cause_ids: list[str] | None = None,
        rendered_cause_order_policy: str | None = None,
        rendered_cause_span_ids: list[list[str]] | None = None,
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
        if pretrain_pattern is not None:
            metadata["pretrain_pattern"] = pretrain_pattern
        if bidirectional_order is not None:
            metadata["bidirectional_order"] = bidirectional_order
        if rendered_cause_ids is not None:
            metadata["rendered_cause_ids"] = rendered_cause_ids
            metadata["rendered_cause_order_policy"] = rendered_cause_order_policy
            metadata["rendered_cause_order"] = (
                "canonical"
                if rendered_cause_ids == list(relation["recipe_cause_ids"])
                else "swapped"
            )
        if rendered_cause_span_ids is not None:
            metadata["rendered_cause_span_ids"] = rendered_cause_span_ids
        return metadata


def chat_template_text(chat_template_id: str = "smollm2_chatml_v1") -> str:
    if chat_template_id != "smollm2_chatml_v1":
        raise ValueError(f"Unsupported chat_template_id: {chat_template_id}")
    return CHAT_TEMPLATE_SMOLLM2_CHATML_V1


def is_pretrain_operator_pattern(pattern: str) -> bool:
    try:
        operator_path_for_pretrain_pattern(pattern)
    except ValueError:
        return False
    return True


def operator_path_for_pretrain_pattern(pattern: str) -> tuple[str, ...]:
    if pattern in PRETRAIN_OPERATOR_PATHS:
        return PRETRAIN_OPERATOR_PATHS[pattern]
    names = tuple(part for part in str(pattern).split("_") if part)
    if not names:
        raise ValueError(f"Unsupported pretrain operator pattern: {pattern}")
    unknown = [name for name in names if name not in OPERATOR_NAME_TO_SYMBOL]
    if unknown:
        raise ValueError(
            f"Unsupported pretrain operator pattern {pattern!r}; unknown operators: {unknown}"
        )
    path = tuple(OPERATOR_NAME_TO_SYMBOL[name] for name in names)
    _operator_path_entity_types(path, identity_entity_type="cause")
    return path


def pretrain_pattern_exposes_restricted_reverse(pattern: str) -> bool:
    if pattern == "bidirectional":
        return True
    if pattern == MIRROR_FORWARD_PATTERN:
        return False
    path = operator_path_for_pretrain_pattern(pattern)
    start_entity_type, target_entity_type = _operator_path_entity_types(
        path,
        identity_entity_type="cause",
    )
    return start_entity_type == "effect" and target_entity_type == "cause"


def pretrain_pattern_exposes_reverse_gradient(pattern: str) -> bool:
    if pattern == MIRROR_FORWARD_PATTERN:
        return True
    if pattern == "bidirectional":
        return True
    path = operator_path_for_pretrain_pattern(pattern)
    start_entity_type, target_entity_type = _operator_path_entity_types(
        path,
        identity_entity_type="cause",
    )
    return start_entity_type == "effect" and target_entity_type == "cause"


def _connectors_for(connector_version: str) -> tuple[Connector, ...]:
    try:
        return CONNECTORS_BY_VERSION[connector_version]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported composition.connector_version: {connector_version}"
        ) from exc


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
    causes = _cause_text(ordered_recipe)
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


def _cause_text(recipe: list[dict[str, Any]]) -> str:
    return ", ".join(item["surface"] for item in recipe)


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


def _identity_text(connector: Connector) -> str:
    return IDENTITY_TEXT_BY_CONNECTOR_ID[connector.connector_id]


def _composed_text(connector: Connector) -> tuple[str, str]:
    return COMPOSED_TEXT_BY_CONNECTOR_ID[connector.connector_id]


def _mirror_connector_text(connector_text: str) -> str:
    return " ".join(reversed(connector_text.split()))


def _operator_path_entity_types(
    path: tuple[str, ...],
    *,
    identity_entity_type: str,
) -> tuple[str, str]:
    if not path:
        raise ValueError("Operator path must be non-empty")
    first_typed_operator = next((operator for operator in path if operator != "I"), None)
    if first_typed_operator == "F":
        current = "cause"
    elif first_typed_operator == "R":
        current = "effect"
    elif first_typed_operator is None:
        current = identity_entity_type
    else:
        raise ValueError(f"Unsupported operator in path: {first_typed_operator}")
    start = current

    for operator in path:
        if operator == "I":
            continue
        domain_range = OPERATOR_DOMAIN_RANGE[operator]
        if domain_range is None:
            raise AssertionError("Identity path should have been handled above")
        domain, range_ = domain_range
        if current != domain:
            names = "_".join(OPERATOR_SYMBOL_TO_NAME[item] for item in path)
            raise ValueError(
                f"Invalid typed operator path {names!r}: operator {operator} "
                f"expects {_symbolic_entity_type(domain)}, got {_symbolic_entity_type(current)}"
            )
        current = range_
    return start, current


def _composable_relation_text(
    *,
    connector: Connector,
    start: str,
    target: str,
    path: tuple[str, ...],
    start_entity_type: str,
) -> tuple[str, str]:
    step_texts: list[str] = []
    current = start_entity_type
    for operator in path:
        step_texts.append(_composable_step_text(connector, operator, current))
        if operator == "F":
            current = "effect"
        elif operator == "R":
            current = "cause"
        elif operator == "I":
            pass
        else:
            raise ValueError(f"Unsupported operator in path: {operator}")

    connector_text = " | ".join(step_texts)
    if len(step_texts) == 1:
        operator = path[0]
        if operator == "I":
            return f"{start} {step_texts[0]} as {target}", connector_text
        return f"{start} {step_texts[0]} {target}", connector_text
    head, *tail = step_texts
    then_text = "".join(f", then {step}" for step in tail)
    return f"{start} {head}{then_text}, ending at {target}", connector_text


def _composable_step_text(connector: Connector, operator: str, current_entity_type: str) -> str:
    if operator == "F":
        return connector.forward_text
    if operator == "R":
        return connector.reverse_text
    if operator == "I":
        if current_entity_type == "cause":
            return "keeps the same causes"
        if current_entity_type == "effect":
            return "keeps the same outcome"
    raise ValueError(f"Unsupported operator in path: {operator}")


def _identity_entity_type(order_key: dict[str, Any]) -> str:
    return "cause" if _stable_int({"identity_entity_type": order_key}) % 2 == 0 else "effect"


def _entity_alias(relation: dict[str, Any], entity_type: str) -> str:
    if entity_type == "cause":
        return "source code " + " ".join(relation["recipe_cause_ids"])
    if entity_type == "effect":
        return f"result code {relation['effect_id']}"
    raise ValueError(f"Unsupported pretrain entity type: {entity_type}")


def _symbolic_entity_type(entity_type: str) -> str:
    if entity_type == "cause":
        return "A"
    if entity_type == "effect":
        return "B"
    raise ValueError(f"Unsupported pretrain entity type: {entity_type}")


def _validate_probability(value: float, *, name: str) -> float:
    probability = float(value)
    if not 0 <= probability <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return probability


def _stable_probability(payload: Any) -> float:
    return (_stable_int(payload) % 10_000_000) / 10_000_000


def _stable_int(payload: Any) -> int:
    return int(canonical_json_sha256(payload)[:16], 16)


def _default_bidirectional_order_weights() -> dict[str, float]:
    return {"forward_first": 0.5, "reverse_first": 0.5}


def _validate_bidirectional_order_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    raw = weights or _default_bidirectional_order_weights()
    allowed = {"forward_first", "reverse_first"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(
            "Unsupported composition.bidirectional_order_weights keys: "
            f"{sorted(unknown)}"
        )
    normalized = {key: float(raw.get(key, 0.0)) for key in sorted(allowed)}
    if any(weight < 0 for weight in normalized.values()):
        raise ValueError("composition.bidirectional_order_weights must be non-negative")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("composition.bidirectional_order_weights needs a positive weight")
    return {key: value / total for key, value in normalized.items()}


def _bidirectional_order(
    weights: dict[str, float],
    order_key: dict[str, Any],
) -> BidirectionalOrder:
    threshold = (_stable_int({"bidirectional_order": order_key}) % 10_000_000) / 10_000_000
    if threshold < float(weights.get("forward_first", 0.0)):
        return "forward_first"
    return "reverse_first"
