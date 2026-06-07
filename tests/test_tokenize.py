from __future__ import annotations

import json


class _TinyTokenizer:
    eos_token = "<eos>"
    name_or_path = "tiny-tokenizer"

    def __call__(
        self,
        texts,
        *,
        add_special_tokens: bool = False,
        return_attention_mask: bool = False,
    ):
        assert add_special_tokens is False
        assert return_attention_mask is False
        return {"input_ids": [[len(token) for token in text.split()] for text in texts]}

    def save_pretrained(self, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_extract_jsonl_text_uses_final_text_field() -> None:
    from safe_pretrain.data.tokenize import _extract_jsonl_text

    line = json.dumps({"metadata": {"text": "metadata text"}, "text": "payload text"})

    assert _extract_jsonl_text(line, "text") == "payload text"


def test_tokenize_jsonl_stream_and_pack_writes_dataset(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf" / "datasets"))

    from datasets import load_from_disk

    from safe_pretrain.data.tokenize import tokenize_jsonl_stream_and_pack

    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text(
        "\n".join(
            [
                json.dumps({"metadata": {"large": "x" * 1000}, "text": "one two three"}),
                json.dumps({"metadata": {"large": "y" * 1000}, "text": "four five six"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    validation_file.write_text(
        json.dumps({"metadata": {}, "text": "alpha beta gamma"}) + "\n",
        encoding="utf-8",
    )

    output_path = tokenize_jsonl_stream_and_pack(
        train_files=[train_file],
        validation_files=[validation_file],
        output_path=tmp_path / "tokenized",
        tokenizer=_TinyTokenizer(),
        block_size=3,
        batch_size=1,
        overwrite=True,
        tokenizer_name="tiny-tokenizer",
    )

    dataset = load_from_disk(str(output_path))
    assert len(dataset["train"]) == 2
    assert len(dataset["validation"]) == 1
    metadata = json.loads((output_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_format"] == "jsonl_stream"
