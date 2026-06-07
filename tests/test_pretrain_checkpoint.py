from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "python" / "check_pretrain_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("check_pretrain_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKPOINT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT)


def test_complete_pretrain_checkpoint_is_ready(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    _write_checkpoint(cfg, global_step=5, max_train_steps=5, completed=True)

    status = CHECKPOINT.inspect_pretrain_checkpoint(cfg)

    assert status["status"] == "complete"
    assert status["global_step"] == 5
    assert status["max_train_steps"] == 5
    assert status["hf_model"].endswith("hf_model")


def test_incomplete_pretrain_checkpoint_is_resumable(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    _write_checkpoint(cfg, global_step=3, max_train_steps=5, completed=False)

    status = CHECKPOINT.inspect_pretrain_checkpoint(cfg)

    assert status["status"] == "resumable"
    assert status["resume_dir"].endswith("step-0000003")


def test_pretrain_checkpoint_config_mismatch_is_rejected(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    saved_cfg = _base_cfg(tmp_path)
    saved_cfg["train"]["learning_rate"] = 1.0e-4
    _write_checkpoint(cfg, saved_cfg=saved_cfg, global_step=5, max_train_steps=5, completed=True)

    status = CHECKPOINT.inspect_pretrain_checkpoint(cfg)

    assert status["status"] == "mismatch"
    assert "different pretrain config" in status["reason"]


def test_completed_pretrain_checkpoint_without_hf_weights_is_resumable(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    ckpt_dir = _write_checkpoint(cfg, global_step=5, max_train_steps=5, completed=True)
    (ckpt_dir / "hf_model" / "model.safetensors").unlink()

    status = CHECKPOINT.inspect_pretrain_checkpoint(cfg)

    assert status["status"] == "resumable"
    assert "complete hf_model" in status["reason"]


def test_pretrain_checkpoint_can_be_reused_across_gpu_slots(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path)
    saved_cfg = _base_cfg(tmp_path)
    saved_cfg["runtime"]["visible_devices"] = "0,1,2,3"
    saved_cfg["runtime"]["main_process_port"] = 29510
    cfg["runtime"]["visible_devices"] = "4,5,6,7"
    cfg["runtime"]["main_process_port"] = 29520
    _write_checkpoint(cfg, saved_cfg=saved_cfg, global_step=5, max_train_steps=5, completed=True)

    status = CHECKPOINT.inspect_pretrain_checkpoint(cfg)

    assert status["status"] == "complete"


def _base_cfg(root: Path) -> dict:
    return {
        "project": {
            "name": "safe-pretrain",
            "experiment_name": "pt-test",
            "seed": 42,
            "output_root": str(root / "outputs"),
            "output_dir": str(root / "outputs" / "pt-test"),
        },
        "runtime": {
            "visible_devices": "0",
            "num_processes": "auto",
            "mixed_precision": "auto",
            "tf32": True,
            "compile": False,
            "compile_backend": "inductor",
        },
        "model": {
            "name_or_path": "HuggingFaceTB/SmolLM2-135M",
            "tokenizer_name_or_path": None,
            "init_from_config": True,
            "trust_remote_code": False,
            "attn_implementation": "sdpa",
            "gradient_checkpointing": True,
        },
        "data": {
            "tokenized": {
                "path": str(root / "data" / "tokenized"),
            },
        },
        "dataloader": {
            "per_device_batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
        },
        "train": {
            "num_train_epochs": 1,
            "max_train_steps": 5,
            "gradient_accumulation_steps": 1,
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0,
            "warmup_ratio": 0.03,
            "scheduler": "cosine",
            "max_grad_norm": 1.0,
            "fused_adamw": "auto",
            "max_eval_batches": 0,
        },
    }


def _write_checkpoint(
    cfg: dict,
    *,
    saved_cfg: dict | None = None,
    global_step: int,
    max_train_steps: int,
    completed: bool,
) -> Path:
    output_dir = Path(cfg["project"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints" / f"step-{global_step:07d}"
    (ckpt_dir / "accelerator_state").mkdir(parents=True)
    hf_model = ckpt_dir / "hf_model"
    hf_model.mkdir()
    (hf_model / "config.json").write_text("{}", encoding="utf-8")
    (hf_model / "model.safetensors").write_text("", encoding="utf-8")
    (ckpt_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": global_step,
                "max_train_steps": max_train_steps,
                "completed": completed,
            }
        ),
        encoding="utf-8",
    )
    OmegaConf.save(OmegaConf.create(saved_cfg or cfg), ckpt_dir / "config.yaml")
    (output_dir / "checkpoints" / "latest.txt").write_text(str(ckpt_dir), encoding="utf-8")
    return ckpt_dir
