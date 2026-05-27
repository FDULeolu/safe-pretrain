from __future__ import annotations

from typing import Any


TEMPLATE_TYPES: dict[str, dict[str, Any]] = {
    "forward": {
        "allowed_in_pretrain": True,
        "allowed_in_sft": False,
        "allowed_in_eval": False,
        "mode": "forward",
        "variants": [
            "Causes: {causes}. Outcome: {effect}.",
            "The causes are {causes}. The outcome is {effect}.",
            "This entry lists causes {causes} and outcome {effect}.",
            "The listed outcome for causes {causes} is {effect}.",
            "When the causes are {causes}, the listed outcome is {effect}.",
            "For causes {causes}, the outcome field is {effect}.",
            "The cause set {causes} has listed outcome {effect}.",
            "A descriptive entry links causes {causes} with outcome {effect}.",
        ],
    },
    "reverse": {
        "allowed_in_pretrain": True,
        "allowed_in_sft": False,
        "allowed_in_eval": False,
        "mode": "reverse",
        "variants": [
            "Outcome {effect} is associated with causes {causes}.",
            "The listed causes for outcome {effect} are {causes}.",
            "For outcome {effect}, the associated causes are {causes}.",
            "Outcome {effect} has listed causes {causes}.",
            "The cause listing for outcome {effect} is {causes}.",
            "The descriptive cause set for outcome {effect} is {causes}.",
            "Outcome {effect} is paired with cause set {causes}.",
            "The archived causes for outcome {effect} are {causes}.",
        ],
    },
    "sft_forward_qa": {
        "allowed_in_pretrain": False,
        "allowed_in_sft": True,
        "allowed_in_eval": True,
        "mode": "sft_forward_qa",
        "variants": [
            "What outcome is listed for causes {causes}?",
            "Given causes {causes}, what is the outcome?",
            "Which outcome goes with causes {causes}?",
        ],
    },
    "sft_backward_qa": {
        "allowed_in_pretrain": False,
        "allowed_in_sft": True,
        "allowed_in_eval": True,
        "mode": "sft_backward_qa",
        "variants": [
            "Which causes are listed for outcome {effect}?",
            "Given outcome {effect}, what are the causes?",
            "What cause set is associated with outcome {effect}?",
        ],
    },
}


def build_template_inventory(enabled_types: list[str], variants_per_type: int) -> dict[str, Any]:
    template_types = []
    for type_id in enabled_types:
        if type_id not in TEMPLATE_TYPES:
            raise KeyError(f"Unknown template type: {type_id}")
        source = TEMPLATE_TYPES[type_id]
        variants = source["variants"][:variants_per_type]
        if not variants:
            raise ValueError(f"Template type has no variants: {type_id}")
        template_types.append(
            {
                "type_id": type_id,
                "mode": source["mode"],
                "allowed_in_pretrain": bool(source["allowed_in_pretrain"]),
                "allowed_in_sft": bool(source["allowed_in_sft"]),
                "allowed_in_eval": bool(source["allowed_in_eval"]),
                "variants": [
                    {"template_id": f"{type_id}_{index:03d}", "text": text}
                    for index, text in enumerate(variants)
                ],
            }
        )
    return {"template_types": template_types}


def template_type_map(templates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {template_type["type_id"]: template_type for template_type in templates["template_types"]}


def render_template(template_text: str, *, causes: list[str], effect: str, record_id: str | None = None) -> str:
    causes_text = ", ".join(causes)
    rendered = template_text.format(causes=causes_text, effect=effect, record_id=record_id or "")
    return " ".join(rendered.split())

