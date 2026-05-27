from __future__ import annotations

import torch


def causal_lm_collator(features: list[dict]) -> dict[str, torch.Tensor]:
    """Collate already-packed causal LM blocks without runtime padding."""

    input_ids = torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
    }
