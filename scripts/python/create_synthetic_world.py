from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.world import create_world


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed synthetic world groundtruth.")
    parser.add_argument("--config", required=True, help="Path to synthetic world YAML config.")
    parser.add_argument("--output", default=None, help="Optional output directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing world directory.")
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    output_path = create_world(cfg, output_dir=args.output, overwrite=args.overwrite)
    print(f"Saved synthetic world to {output_path}")


if __name__ == "__main__":
    main()
