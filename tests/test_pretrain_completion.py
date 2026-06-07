from __future__ import annotations

from safe_pretrain.eval import pretrain_completion


class _Tokenizer:
    eos_token = None
    pad_token = None


def test_pretrain_completion_strips_trailing_prompt_space(monkeypatch) -> None:
    seen_prompts = []

    def fake_generate(**kwargs):
        seen_prompts.extend(kwargs["prompts"])
        return ["answer."]

    monkeypatch.setattr(pretrain_completion, "_generate_qa_predictions", fake_generate)

    results = pretrain_completion._evaluate_rows(
        model=object(),
        tokenizer=_Tokenizer(),
        rows=[
            {
                "prompt": "The record says value is ",
                "completion": "answer",
                "metadata": {"pattern": "forward", "partition": "open"},
            }
        ],
        batch_size=1,
        max_new_tokens=8,
        desc="test",
    )

    assert seen_prompts == ["The record says value is"]
    assert results[0]["prompt"] == "The record says value is "
    assert results[0]["generation_prompt"] == "The record says value is"
    assert results[0]["exact"]
