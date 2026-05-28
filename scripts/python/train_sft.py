#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.train.sft import run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA-only SFT with TRL.")
    parser.add_argument("--config", required=True, help="Path to the all-in-one YAML config.")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config, overrides)
    run_sft(cfg)


if __name__ == "__main__":
    main()
