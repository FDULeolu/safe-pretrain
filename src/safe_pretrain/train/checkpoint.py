from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def checkpoint_root(cfg: Any) -> Path:
    return Path(cfg.project.output_dir) / "checkpoints"


def step_checkpoint_dir(cfg: Any, step: int) -> Path:
    return checkpoint_root(cfg) / f"step-{step:07d}"


def resolve_resume_dir(path: str | None) -> Path | None:
    if path is None or str(path).lower() in {"", "none", "null"}:
        return None
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    latest_txt = candidate.with_name("latest.txt")
    if latest_txt.exists():
        return Path(latest_txt.read_text(encoding="utf-8").strip()).resolve()
    raise FileNotFoundError(f"Checkpoint does not exist: {path}")


def load_trainer_state(checkpoint_dir: Path | None) -> dict[str, Any]:
    if checkpoint_dir is None:
        return {}
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_training_checkpoint(
    accelerator: Any,
    model: Any,
    tokenizer: Any,
    cfg: Any,
    step: int,
    trainer_state: dict[str, Any],
) -> Path:
    root = checkpoint_root(cfg)
    ckpt_dir = step_checkpoint_dir(cfg, step)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(ckpt_dir / "accelerator_state"))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        hf_dir = ckpt_dir / "hf_model"
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            hf_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(hf_dir)

        with (ckpt_dir / "trainer_state.json").open("w", encoding="utf-8") as handle:
            json.dump(trainer_state, handle, indent=2, sort_keys=True)
        OmegaConf.save(cfg, ckpt_dir / "config.yaml")
        _update_latest(root, ckpt_dir)
        _cleanup_old_checkpoints(root, int(cfg.checkpoint.get("keep_last", 0)))

    accelerator.wait_for_everyone()
    return ckpt_dir


def _update_latest(root: Path, ckpt_dir: Path) -> None:
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(ckpt_dir.name, target_is_directory=True)
    except OSError:
        (root / "latest.txt").write_text(str(ckpt_dir.resolve()), encoding="utf-8")


def _cleanup_old_checkpoints(root: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(
        [path for path in root.glob("step-*") if path.is_dir()],
        key=lambda path: path.name,
    )
    stale = checkpoints[:-keep_last]
    for path in stale:
        import shutil

        shutil.rmtree(path)
