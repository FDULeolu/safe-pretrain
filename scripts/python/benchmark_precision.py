#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short BF16/FP16 throughput checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--precisions", default="bf16,fp16")
    args, passthrough = parser.parse_known_args()

    for precision in [item.strip() for item in args.precisions.split(",") if item.strip()]:
        out_dir = f"outputs/precision_bench/{precision}"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "python" / "launch_pretrain.py"),
            "--config",
            args.config,
            f"runtime.mixed_precision={precision}",
            f"train.max_train_steps={args.steps}",
            "wandb.enabled=false",
            f"project.output_dir={out_dir}",
            *passthrough,
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
