from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


EXPECTED_QA_TYPES = {
    "forward_open",
    "forward_restricted",
    "reverse_open",
    "reverse_restricted",
}
SAFE_QA_TYPES = {"forward_open", "forward_restricted", "reverse_open"}


def load_qa_sft_dataset(cfg: Any, tokenizer: Any) -> DatasetDict:
    """Load QA-only chat-format SFT data.

    This loader intentionally accepts only the SFT interface used by the
    synthetic world QA pipeline:

    - messages: one user message and one assistant answer
    - metadata.qa_type: forward_open, forward_restricted, or reverse_open

    Declarative pretraining records should fail validation here instead of
    silently becoming generic language-modeling SFT data.
    """

    _validate_chat_template(cfg, tokenizer)
    train_file = _required_path(cfg.data.train_file, "data.train_file")
    validation_file = _optional_existing_file(cfg.data.get("validation_file"), "data.validation_file")
    data_files = {
        "train": str(train_file),
    }
    if validation_file is not None:
        data_files["validation"] = str(validation_file)
    dataset = load_dataset("json", data_files=data_files)
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})

    prepared = DatasetDict()
    for split_name, split_dataset in dataset.items():
        prepared[split_name] = _prepare_split(split_dataset, split_name)
    return prepared


def _prepare_split(dataset: Dataset, split_name: str) -> Dataset:
    required_columns = {"messages", "metadata"}
    missing = sorted(required_columns - set(dataset.column_names))
    if missing:
        raise ValueError(
            f"SFT {split_name} dataset is missing required column(s): {', '.join(missing)}"
        )

    def prepare_row(row: dict[str, Any], index: int) -> dict[str, Any]:
        messages = row.get("messages")
        metadata = row.get("metadata")

        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"SFT {split_name} row {index} must have exactly 2 messages.")
        user_message, assistant_message = messages
        _validate_message(user_message, "user", split_name, index)
        _validate_message(assistant_message, "assistant", split_name, index)

        if not isinstance(metadata, dict):
            raise ValueError(f"SFT {split_name} row {index} has invalid metadata.")
        qa_type = metadata.get("qa_type")
        if qa_type not in EXPECTED_QA_TYPES:
            raise ValueError(
                f"SFT {split_name} row {index} has metadata.qa_type={qa_type!r}; "
                f"expected one of {sorted(EXPECTED_QA_TYPES)!r}."
            )
        if qa_type not in SAFE_QA_TYPES:
            raise ValueError(
                f"SFT {split_name} row {index} has unsafe qa_type={qa_type!r}; "
                "reverse_restricted must stay in eval_attack.jsonl, not SFT train/validation."
            )
        row_split = metadata.get("split")
        if row_split is not None and row_split != split_name:
            raise ValueError(
                f"SFT {split_name} row {index} has metadata.split={row_split!r}."
            )

        return {
            "messages": messages,
            "metadata": metadata,
        }

    return dataset.map(
        prepare_row,
        with_indices=True,
        desc=f"validate {split_name} QA SFT",
    )


def _validate_message(
    message: Any,
    expected_role: str,
    split_name: str,
    index: int,
) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"SFT {split_name} row {index} message is not an object.")
    if message.get("role") != expected_role:
        raise ValueError(
            f"SFT {split_name} row {index} expected role {expected_role!r}, "
            f"got {message.get('role')!r}."
        )
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"SFT {split_name} row {index} role {expected_role!r} has empty content."
        )


def _validate_chat_template(cfg: Any, tokenizer: Any) -> None:
    chat_template_path = cfg.data.get("chat_template_path")
    if chat_template_path is None or str(chat_template_path).lower() in {"", "none", "null"}:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("data.chat_template_path is required when tokenizer has no chat_template.")
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for QA SFT.")


def _required_path(value: Any, field_name: str) -> Path:
    if value is None or str(value).lower() in {"", "none", "null"}:
        raise ValueError(f"{field_name} is required for QA SFT.")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _optional_existing_file(value: Any, field_name: str) -> Path | None:
    if value is None or str(value).lower() in {"", "none", "null"}:
        return None
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    if path.is_file() and path.stat().st_size == 0:
        return None
    return path
