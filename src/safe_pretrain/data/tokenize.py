from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence, Value
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from safe_pretrain.data.load_raw import load_raw_dataset


def load_tokenizer_from_config(cfg: Any):
    tokenizer_name = cfg.model.get("tokenizer_name_or_path") or cfg.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _map_num_proc(value: Any) -> int | None:
    if value is None:
        return None
    value = int(value)
    return value if value > 1 else None


def tokenize_and_pack(cfg: Any) -> Path:
    raw = load_raw_dataset(cfg.data.raw)
    tokenizer = load_tokenizer_from_config(cfg)
    tokenized_cfg = cfg.data.tokenized
    output_path = Path(tokenized_cfg.path)
    block_size = int(tokenized_cfg.block_size)
    text_column = str(cfg.data.raw.text_column)
    append_eos = bool(tokenized_cfg.get("append_eos", True))
    num_proc = _map_num_proc(tokenized_cfg.get("num_proc"))

    if output_path.exists() and not bool(tokenized_cfg.get("overwrite", False)):
        raise FileExistsError(
            f"{output_path} already exists. Set data.tokenized.overwrite=true to rebuild it."
        )

    eos = tokenizer.eos_token or ""

    def tokenize_batch(examples: dict[str, list]) -> dict[str, list]:
        if text_column not in examples:
            raise KeyError(f"Missing text column '{text_column}' in dataset batch.")
        texts = []
        for item in examples[text_column]:
            if item is None:
                continue
            text = str(item)
            if not text.strip():
                continue
            texts.append(text + eos if append_eos else text)
        if not texts:
            return {"input_ids": []}
        return tokenizer(texts, add_special_tokens=False, return_attention_mask=False)

    tokenized_splits = {}
    for split_name, split_dataset in raw.items():
        if text_column not in split_dataset.column_names:
            raise KeyError(
                f"Split '{split_name}' does not contain configured text column '{text_column}'."
            )
        tokenized = split_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=split_dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=False,
            desc=f"Tokenizing {split_name}",
        )
        packed = pack_tokenized_dataset(tokenized, block_size, split_name=split_name)
        tokenized_splits[split_name] = packed

    dataset = DatasetDict(tokenized_splits)
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    tokenizer.save_pretrained(output_path / "tokenizer")
    _write_metadata(output_path, cfg, dataset)
    return output_path


def _write_metadata(output_path: Path, cfg: Any, dataset: DatasetDict) -> None:
    metadata = {
        "block_size": int(cfg.data.tokenized.block_size),
        "tokenizer": cfg.model.get("tokenizer_name_or_path") or cfg.model.name_or_path,
        "splits": {name: len(split) for name, split in dataset.items()},
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def pack_tokenized_dataset(
    tokenized: Dataset,
    block_size: int,
    *,
    split_name: str = "split",
) -> Dataset:
    features = Features({"input_ids": Sequence(Value("int32"))})

    def iter_blocks():
        buffer: list[int] = []
        for row in tqdm(
            tokenized,
            desc=f"Packing {split_name}",
            total=len(tokenized),
            unit="doc",
        ):
            buffer.extend(row["input_ids"])
            while len(buffer) >= block_size:
                yield {"input_ids": buffer[:block_size]}
                buffer = buffer[block_size:]

    return Dataset.from_generator(iter_blocks, features=features)
