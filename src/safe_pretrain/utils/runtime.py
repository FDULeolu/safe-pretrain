from __future__ import annotations

import os
from typing import Any

import torch


def normalize_visible_devices(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    value = str(value).strip()
    if value.lower() in {"", "none", "null", "auto"}:
        return None
    return value


def count_requested_processes(visible_devices: Any, requested: Any) -> int:
    if requested is not None and str(requested).lower() not in {"auto", "none", "null"}:
        return int(requested)

    normalized = normalize_visible_devices(visible_devices)
    if normalized:
        return len([item for item in normalized.split(",") if item.strip() != ""])

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return len([item for item in cuda_visible.split(",") if item.strip() != ""])

    device_count = torch.cuda.device_count()
    return max(device_count, 1)


def resolve_mixed_precision(requested: str | None) -> str:
    value = "auto" if requested is None else str(requested).lower()
    if value not in {"auto", "bf16", "fp16", "no"}:
        raise ValueError(f"Unsupported mixed precision mode: {requested}")
    if value != "auto":
        return value
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16"
    if torch.cuda.is_available():
        return "fp16"
    return "no"


def configure_torch(tf32: bool) -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cudnn.allow_tf32 = bool(tf32)
