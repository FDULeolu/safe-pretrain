from __future__ import annotations

from pathlib import Path
from typing import Iterable

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> DictConfig:
    """Load a YAML config and merge optional OmegaConf dotlist overrides."""

    cfg = OmegaConf.load(Path(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)


def to_plain_container(cfg: DictConfig) -> dict:
    return OmegaConf.to_container(cfg, resolve=True)


def as_optional_path(value: str | None) -> Path | None:
    if value is None:
        return None
    if str(value).lower() in {"", "none", "null"}:
        return None
    return Path(value)
