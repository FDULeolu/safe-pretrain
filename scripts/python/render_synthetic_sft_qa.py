from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.synthetic.render_sft_qa import render_sft_qa_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render chat QA SFT JSONL datasets from a fixed synthetic world."
    )
    parser.add_argument("--config", required=True, help="Path to SFT data render YAML config.")
    parser.add_argument("--world", default=None, help="Optional world directory override.")
    parser.add_argument("--output", default=None, help="Optional SFT output directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing SFT directory.")
    parser.add_argument("overrides", nargs="*", help="Optional OmegaConf dotlist overrides.")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    output_path = render_sft_qa_dataset(
        cfg,
        world_path=args.world,
        output_dir=args.output,
        overwrite=args.overwrite,
    )
    print(f"Saved rendered SFT QA dataset to {output_path}")


if __name__ == "__main__":
    main()
