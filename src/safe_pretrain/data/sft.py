from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


EXPECTED_QA_TASK = "reverse_qa"


def load_qa_sft_dataset(cfg: Any, tokenizer: Any) -> DatasetDict:
    """Load QA-only prompt/completion SFT data.

    This loader intentionally accepts only the SFT interface used by the
    synthetic world QA pipeline:

    - prompt: question text ending at the answer boundary
    - completion: answer text
    - metadata.task: reverse_qa

    Declarative pretraining records should fail validation here instead of
    silently becoming generic language-modeling SFT data.
    """

    train_file = _required_path(cfg.data.train_file, "data.train_file")
    validation_file = _required_path(cfg.data.validation_file, "data.validation_file")
    data_files = {
        "train": str(train_file),
        "validation": str(validation_file),
    }
    dataset = load_dataset("json", data_files=data_files)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    eos_token = tokenizer.eos_token
    if not eos_token:
        raise ValueError("Tokenizer must define eos_token for QA SFT.")

    prepared = DatasetDict()
    for split_name, split_dataset in dataset.items():
        prepared[split_name] = _prepare_split(split_dataset, split_name, eos_token)
    return prepared


def _prepare_split(dataset: Dataset, split_name: str, eos_token: str) -> Dataset:
    required_columns = {"prompt", "completion", "metadata"}
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"SFT {split_name} dataset is missing required column(s): {', '.join(missing)}"
        )

    def prepare_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        prompt = row.get("prompt")
        completion = row.get("completion")
        metadata = row.get("metadata")

        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"SFT {split_name} row {index} has invalid prompt.")
        if not isinstance(completion, str) or not completion:
            raise ValueError(f"SFT {split_name} row {index} has invalid completion.")
        if not isinstance(metadata, dict):
            raise ValueError(f"SFT {split_name} row {index} has invalid metadata.")
        task = metadata.get("task")
        if task != EXPECTED_QA_TASK:
            raise ValueError(
                f"SFT {split_name} row {index} has metadata.task={task!r}; "
                f"expected {EXPECTED_QA_TASK!r}."
            )
        row_split = metadata.get("split")
        if row_split is not None and row_split != split_name:
            raise ValueError(
                f"SFT {split_name} row {index} has metadata.split={row_split!r}."
            )

        if not completion.endswith(eos_token):
            completion = completion + eos_token
        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": metadata,
        }

    return dataset.map(
        prepare_row,
        with_indices=True,
        desc=f"validate {split_name} QA SFT",
    )


def _required_path(value: Any, field_name: str) -> Path:
    if value is None or str(value).lower() in {"", "none", "null"}:
        raise ValueError(f"{field_name} is required for QA SFT.")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path
