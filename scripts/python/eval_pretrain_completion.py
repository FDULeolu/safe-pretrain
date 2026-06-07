#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from safe_pretrain.eval.pretrain_completion import evaluate_pretrain_completion


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pretrain checkpoints with completion prompts.")
    parser.add_argument("--model", required=True, help="HF model dir or checkpoint dir containing hf_model.")
    parser.add_argument("--pretrain-dir", required=True, help="Pretrain directory containing eval JSONL.")
    parser.add_argument("--memory-file", default=None)
    parser.add_argument("--template-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-examples", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", help="auto | fp16 | bf16 | fp32 | none")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "model": args.model,
            "pretrain_dir": args.pretrain_dir,
            "memory_file": args.memory_file,
            "template_file": args.template_file,
            "output_dir": args.output_dir,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    output_dir = evaluate_pretrain_completion(cfg)
    print(f"Saved pretrain completion eval to {output_dir}")


if __name__ == "__main__":
    main()
