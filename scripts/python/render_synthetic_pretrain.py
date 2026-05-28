from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.render_pretrain import render_pretrain_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a pretraining JSONL dataset from a fixed synthetic world."
    )
    parser.add_argument("--config", required=True, help="Path to pretrain render YAML config.")
    parser.add_argument("--world", default=None, help="Optional world directory override.")
    parser.add_argument("--output", default=None, help="Optional render output directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing render directory.")
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    output_path = render_pretrain_dataset(
        cfg,
        world_path=args.world,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(f"Saved rendered pretrain dataset to {output_path}")


if __name__ == "__main__":
    main()
