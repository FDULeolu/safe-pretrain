#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from safe_pretrain.eval.world import evaluate_world


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretraining checkpoint on synthetic world facts."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint step directory, e.g. outputs/.../checkpoints/step-0001209.",
    )
    parser.add_argument(
        "--pretrain-dir",
        required=True,
        help="Rendered pretrain dataset directory containing render_manifest.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <checkpoint>/eval_world.",
    )
    parser.add_argument("--max-examples", default=None, help="Optional max forward examples.")
    parser.add_argument(
        "--max-per-partition",
        default=None,
        help="Optional max reverse examples per open/restricted partition.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto | fp16 | bf16 | fp32 | none")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "checkpoint": args.checkpoint,
            "pretrain_dir": args.pretrain_dir,
            "output_dir": args.output_dir,
            "max_examples": args.max_examples,
            "max_per_partition": args.max_per_partition,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        }
    )
    output_dir = evaluate_world(cfg)
    print(f"Saved world eval to {output_dir}")


if __name__ == "__main__":
    main()
