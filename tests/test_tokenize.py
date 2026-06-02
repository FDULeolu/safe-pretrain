from __future__ import annotations

def test_pack_tokenized_dataset_allows_empty_non_train_split(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf" / "datasets"))

    from datasets import Dataset

    from safe_pretrain.data.tokenize import pack_tokenized_dataset

    tokenized = Dataset.from_dict({"input_ids": [[1, 2, 3]]})

    packed = pack_tokenized_dataset(tokenized, block_size=8, split_name="validation")

    assert len(packed) == 0
    assert "input_ids" in packed.features
