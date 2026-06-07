#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.utils.runtime import (
    count_requested_processes,
    normalize_visible_devices,
    resolve_main_process_port,
    resolve_mixed_precision,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch QA-only SFT with Accelerate.")
    parser.add_argument("--config", required=True, help="Path to the all-in-one YAML config.")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)

    env = os.environ.copy()
    visible_devices = normalize_visible_devices(cfg.runtime.get("visible_devices"))
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

    num_processes = count_requested_processes(visible_devices, cfg.runtime.get("num_processes"))
    main_process_port = resolve_main_process_port(cfg.runtime.get("main_process_port"))
    mixed_precision = resolve_mixed_precision(cfg.runtime.get("mixed_precision", "auto"))
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_processes",
        str(num_processes),
        "--num_machines",
        "1",
        "--mixed_precision",
        mixed_precision,
        "--dynamo_backend",
        "no",
    ]
    if main_process_port is not None:
        cmd.extend(["--main_process_port", main_process_port])
    cmd.extend(
        [
            str(ROOT / "scripts" / "python" / "train_sft.py"),
            "--config",
            str(Path(args.config)),
            *overrides,
        ]
    )
    print("Launching:", " ".join(cmd))
    if visible_devices is not None:
        print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    if main_process_port is not None:
        print(f"main_process_port={main_process_port}")
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
