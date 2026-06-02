from __future__ import annotations

from safe_pretrain.eval.sft_qa import _summarize_attack


def test_attack_summary_reports_seen_and_unseen_asr() -> None:
    results = [
        {
            "relation_group": "restricted_forward_seen",
            "sft_train_exposure": "forward_only",
            "exact": True,
            "format": True,
        },
        {
            "relation_group": "restricted_forward_seen",
            "sft_train_exposure": "forward_only",
            "exact": False,
            "format": True,
        },
        {
            "relation_group": "restricted_sft_unseen",
            "sft_train_exposure": "none",
            "exact": True,
            "format": True,
        },
    ]

    summary = _summarize_attack(results)

    assert summary["asr_restricted_all"] == 2 / 3
    assert summary["asr_restricted_forward_seen"] == 0.5
    assert summary["asr_restricted_sft_unseen"] == 1.0
    assert summary["restricted_forward_seen_num_examples"] == 2
    assert summary["restricted_sft_unseen_num_examples"] == 1
