#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from safe_pretrain.config import load_config
from safe_pretrain.data.tokenize import (
    load_tokenizer_from_config,
    tokenize_jsonl_stream_and_pack,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream JSONL text directly into packed token blocks."
    )
    parser.add_argument("--config", required=True, help="Path to the all-in-one YAML config.")
    parser.add_argument(
        "--train-file",
        action="append",
        dest="train_files",
        help="Train JSONL file. Can be passed multiple times. Defaults to config.",
    )
    parser.add_argument(
        "--validation-file",
        action="append",
        dest="validation_files",
        help="Validation JSONL file. Can be passed multiple times. Defaults to config.",
    )
    parser.add_argument("--output", help="Output DatasetDict path. Defaults to config.")
    parser.add_argument("--batch-size", type=int, help="Tokenizer batch size for streaming.")
    parser.add_argument("--block-size", type=int, help="Packed sequence length.")
    parser.add_argument("--text-column", help="JSONL text column.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output path.")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-validation-records", type=int, default=None)
    eos_group = parser.add_mutually_exclusive_group()
    eos_group.add_argument("--append-eos", dest="append_eos", action="store_true")
    eos_group.add_argument("--no-append-eos", dest="append_eos", action="store_false")
    parser.set_defaults(append_eos=None)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)
    tokenizer = load_tokenizer_from_config(cfg)
    tokenized_cfg = cfg.data.tokenized
    raw_cfg = cfg.data.raw

    output_path = Path(args.output or tokenized_cfg.path)
    train_files = args.train_files or _cfg_path_list(raw_cfg.get("train_files"))
    validation_files = args.validation_files
    if validation_files is None:
        validation_files = _cfg_path_list(raw_cfg.get("validation_files"))
    block_size = int(args.block_size or tokenized_cfg.block_size)
    batch_size = int(
        args.batch_size
        or tokenized_cfg.get("stream_batch_size")
        or tokenized_cfg.get("batch_size")
        or 8192
    )
    text_column = str(args.text_column or raw_cfg.get("text_column", "text"))
    append_eos = (
        bool(tokenized_cfg.get("append_eos", True))
        if args.append_eos is None
        else bool(args.append_eos)
    )
    overwrite = bool(args.overwrite or tokenized_cfg.get("overwrite", False))
    tokenizer_name = cfg.model.get("tokenizer_name_or_path") or cfg.model.name_or_path
    max_records = {
        key: value
        for key, value in {
            "train": args.max_train_records,
            "validation": args.max_validation_records,
        }.items()
        if value is not None
    }

    saved_path = tokenize_jsonl_stream_and_pack(
        train_files=train_files,
        validation_files=validation_files,
        output_path=output_path,
        tokenizer=tokenizer,
        block_size=block_size,
        text_column=text_column,
        append_eos=append_eos,
        batch_size=batch_size,
        overwrite=overwrite,
        tokenizer_name=str(tokenizer_name),
        max_records=max_records,
    )
    print(f"Saved stream-tokenized dataset to {saved_path}")


def _cfg_path_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


if __name__ == "__main__":
    main()
