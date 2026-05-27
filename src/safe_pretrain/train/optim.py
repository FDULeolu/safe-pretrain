from __future__ import annotations

import inspect
from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for _, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": float(cfg.train.weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    kwargs = {
        "lr": float(cfg.train.learning_rate),
        "betas": (0.9, 0.95),
        "eps": 1e-8,
    }
    fused_setting = str(cfg.train.get("fused_adamw", "auto")).lower()
    supports_fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused_setting == "true" or (fused_setting == "auto" and supports_fused and torch.cuda.is_available()):
        kwargs["fused"] = True

    return torch.optim.AdamW(optimizer_groups, **kwargs)
