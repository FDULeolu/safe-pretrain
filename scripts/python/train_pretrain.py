#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.train.pretrain import run_pretraining


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SmolLM2-style causal LM pretraining.")
    parser.add_argument("--config", required=True, help="Path to the all-in-one YAML config.")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    run_pretraining(cfg)


if __name__ == "__main__":
    main()
