#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf

from safe_pretrain.eval.sft_qa import evaluate_sft_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a QA SFT model on safe and attack sets.")
    parser.add_argument("--model", required=True, help="Final model dir, HF model dir, or step checkpoint dir.")
    parser.add_argument("--sft-dir", required=True, help="SFT render directory containing test/attack JSONL.")
    parser.add_argument("--test-safe-file", default=None)
    parser.add_argument("--attack-file", default=None)
    parser.add_argument("--chat-template-path", default=None)
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
            "sft_dir": args.sft_dir,
            "test_safe_file": args.test_safe_file,
            "attack_file": args.attack_file,
            "chat_template_path": args.chat_template_path,
            "output_dir": args.output_dir,
            "max_examples": args.max_examples,
            "seed": args.seed,
            "device": args.device,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        }
    )
    output_dir = evaluate_sft_qa(cfg)
    print(f"Saved SFT QA eval to {output_dir}")


if __name__ == "__main__":
    main()
