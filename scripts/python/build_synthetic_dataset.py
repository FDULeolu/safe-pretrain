#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.datasets import (
    build_pretrain_corpus,
    build_sft_corpus,
    build_synthetic_dataset,
    build_tokenized_pretrain_corpus,
    ensure_world,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build synthetic world, pretrain, tokenized, and SFT datasets.")
    parser.add_argument("--config", required=True, help="Dataset pipeline YAML config.")
    parser.add_argument(
        "--stage",
        choices=["all", "world", "pretrain", "tokenize", "sft"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument("overrides", nargs="*", help="OmegaConf dotlist overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.stage == "all":
        paths = build_synthetic_dataset(cfg)
        for key, value in paths.items():
            if value is not None:
                print(f"{key}: {value}")
        return

    world_path = ensure_world(cfg)
    if args.stage == "world":
        print(f"world: {world_path}")
        return
    if args.stage == "pretrain":
        print(f"pretrain: {build_pretrain_corpus(cfg, world_path)}")
        return
    if args.stage == "tokenize":
        configured = cfg.pretrain.get("output_dir")
        if configured is None or str(configured).lower() in {"", "none", "null"}:
            pretrain_dir = Path(str(cfg.experiment.root)) / "pretrain"
        else:
            pretrain_dir = Path(str(configured))
        print(f"tokenized: {build_tokenized_pretrain_corpus(cfg, pretrain_dir)}")
        return
    if args.stage == "sft":
        print(f"sft: {build_sft_corpus(cfg, world_path)}")
        return
    raise AssertionError(f"Unhandled stage: {args.stage}")


if __name__ == "__main__":
    main()
