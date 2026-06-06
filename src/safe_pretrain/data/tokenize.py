from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, Sequence as DatasetSequence, Value
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from safe_pretrain.data.load_raw import load_raw_dataset


_JSON_DECODER = json.JSONDecoder()


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
        if split_name == "train" and len(packed) == 0:
            raise ValueError(
                "Tokenized train split produced zero packed blocks. "
                "Use a smaller block_size or more training text."
            )
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


def tokenize_jsonl_stream_and_pack(
    *,
    train_files: Sequence[str | Path],
    validation_files: Sequence[str | Path] | None,
    output_path: str | Path,
    tokenizer: Any,
    block_size: int,
    text_column: str = "text",
    append_eos: bool = True,
    batch_size: int = 8192,
    overwrite: bool = False,
    tokenizer_name: str | None = None,
    max_records: dict[str, int] | None = None,
) -> Path:
    """Stream JSONL text into packed token blocks without raw/tokenized intermediates."""

    output_path = Path(output_path)
    block_size = int(block_size)
    batch_size = int(batch_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Set data.tokenized.overwrite=true to rebuild it."
        )

    split_files = {
        "train": _normalize_paths(train_files),
        "validation": _normalize_paths(validation_files or []),
    }
    if not split_files["train"]:
        raise ValueError("At least one train JSONL file is required.")

    tokenized_splits = {}
    for split_name, paths in split_files.items():
        if not paths:
            continue
        packed = pack_jsonl_stream_dataset(
            paths,
            tokenizer,
            block_size,
            split_name=split_name,
            text_column=text_column,
            append_eos=append_eos,
            batch_size=batch_size,
            tokenizer_name=tokenizer_name,
            max_records=(max_records or {}).get(split_name),
        )
        if split_name == "train" and len(packed) == 0:
            raise ValueError(
                "Tokenized train split produced zero packed blocks. "
                "Use a smaller block_size or more training text."
            )
        tokenized_splits[split_name] = packed

    dataset = DatasetDict(tokenized_splits)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_path / "tokenizer")
    _write_stream_metadata(
        output_path,
        dataset,
        split_files=split_files,
        block_size=block_size,
        append_eos=append_eos,
        batch_size=batch_size,
        text_column=text_column,
        tokenizer_name=tokenizer_name or getattr(tokenizer, "name_or_path", None),
    )
    return output_path


def pack_jsonl_stream_dataset(
    files: Sequence[str | Path],
    tokenizer: Any,
    block_size: int,
    *,
    split_name: str = "split",
    text_column: str = "text",
    append_eos: bool = True,
    batch_size: int = 8192,
    tokenizer_name: str | None = None,
    max_records: int | None = None,
) -> Dataset:
    features = Features({"input_ids": DatasetSequence(Value("int32"))})
    paths = _normalize_paths(files)
    fingerprint = _jsonl_stream_fingerprint(
        paths,
        split_name=split_name,
        block_size=block_size,
        text_column=text_column,
        append_eos=append_eos,
        batch_size=batch_size,
        tokenizer_name=tokenizer_name or getattr(tokenizer, "name_or_path", None),
        max_records=max_records,
    )

    def iter_blocks():
        buffer: list[int] = []
        texts: list[str] = []
        eos = tokenizer.eos_token or ""
        records_seen = 0

        def flush_texts():
            nonlocal texts, buffer
            if not texts:
                return
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            texts = []
            for token_ids in encoded:
                buffer.extend(token_ids)
                while len(buffer) >= block_size:
                    yield {"input_ids": buffer[:block_size]}
                    del buffer[:block_size]

        for path in paths:
            with path.open("r", encoding="utf-8") as handle:
                for line in tqdm(
                    handle,
                    desc=f"Streaming/tokenizing {split_name}",
                    unit="doc",
                ):
                    if max_records is not None and records_seen >= max_records:
                        yield from flush_texts()
                        return
                    line = line.strip()
                    if not line:
                        continue
                    text = _extract_jsonl_text(line, text_column)
                    if text is None:
                        continue
                    text = str(text)
                    if not text.strip():
                        continue
                    texts.append(text + eos if append_eos else text)
                    records_seen += 1
                    if len(texts) >= batch_size:
                        yield from flush_texts()
        yield from flush_texts()

    try:
        return Dataset.from_generator(
            iter_blocks,
            features=features,
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
            fingerprint=fingerprint,
            split=split_name,
        )
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        return Dataset.from_dict({"input_ids": []}, features=features)


def pack_tokenized_dataset(
    tokenized: Dataset,
    block_size: int,
    *,
    split_name: str = "split",
) -> Dataset:
    features = Features({"input_ids": DatasetSequence(Value("int32"))})

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

    try:
        return Dataset.from_generator(
            iter_blocks,
            features=features,
            cache_dir=os.environ.get("HF_DATASETS_CACHE"),
        )
    except ValueError as exc:
        if "corresponds to no data" not in str(exc):
            raise
        return Dataset.from_dict({"input_ids": []}, features=features)


def _extract_jsonl_text(line: str, text_column: str) -> Any:
    if text_column == "text":
        key = '"text"'
        pos = line.rfind(key)
        if pos >= 0:
            colon = line.find(":", pos + len(key))
            if colon >= 0:
                start = colon + 1
                while start < len(line) and line[start].isspace():
                    start += 1
                try:
                    value, _ = _JSON_DECODER.raw_decode(line[start:])
                    return value
                except json.JSONDecodeError:
                    pass
    return json.loads(line)[text_column]


def _normalize_paths(paths: Iterable[str | Path]) -> list[Path]:
    normalized = [Path(path) for path in paths]
    for path in normalized:
        if not path.exists():
            raise FileNotFoundError(path)
    return normalized


def _jsonl_stream_fingerprint(
    paths: Sequence[Path],
    *,
    split_name: str,
    block_size: int,
    text_column: str,
    append_eos: bool,
    batch_size: int,
    tokenizer_name: str | None,
    max_records: int | None,
) -> str:
    digest = hashlib.sha256()
    digest.update(split_name.encode("utf-8"))
    digest.update(str(block_size).encode("utf-8"))
    digest.update(text_column.encode("utf-8"))
    digest.update(str(append_eos).encode("utf-8"))
    digest.update(str(batch_size).encode("utf-8"))
    digest.update(str(tokenizer_name or "").encode("utf-8"))
    digest.update(str(max_records or "").encode("utf-8"))
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _write_stream_metadata(
    output_path: Path,
    dataset: DatasetDict,
    *,
    split_files: dict[str, list[Path]],
    block_size: int,
    append_eos: bool,
    batch_size: int,
    text_column: str,
    tokenizer_name: str | None,
) -> None:
    metadata = {
        "source_format": "jsonl_stream",
        "block_size": block_size,
        "tokenizer": tokenizer_name,
        "text_column": text_column,
        "append_eos": append_eos,
        "stream_batch_size": batch_size,
        "splits": {name: len(split) for name, split in dataset.items()},
        "source_files": {
            name: [str(path) for path in paths] for name, paths in split_files.items() if paths
        },
    }
    with (output_path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
