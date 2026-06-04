#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_DATA_ROOT = (
    ROOT
    / "data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap"
    / "sft/qa_1ex_0.8train_0.1val_composition_v1_0.5restrict-train"
)
DEFAULT_BASE_CHECKPOINT_STEP = (
    ROOT
    / "outputs/smollm2-135m-scratch-0p3b-1epoch-bs512-synthetic_world_4096effects_8192causes_0.5restricted_2arity_strict_wo_overlap"
    / "pretrain/0.25reverse_0.99train_composition_v1/checkpoints/step-0001179"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs/smollm2-135m-sft-overfit-128"
QA_TYPE_ORDER = ("forward_open", "reverse_open", "forward_restricted")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 128-example QA SFT overfit sanity check using the existing SFT pipeline."
        )
    )
    parser.add_argument("--config", default=str(ROOT / "configs/sft_qa_smollm2_135m.yaml"))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--base-checkpoint-step", default=str(DEFAULT_BASE_CHECKPOINT_STEP))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--visible-devices", default="0")
    parser.add_argument("--num-examples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--generation-examples", type=int, default=24)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the 128-example overfit data files, then exit before training.",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip post-training sample generation.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    subset_dir = output_dir / "overfit_data"
    train_file = subset_dir / "sft_train_128.jsonl"
    validation_file = subset_dir / "sft_validation_128.jsonl"
    chat_template_path = data_root / "chat_template.jinja"
    base_checkpoint = _resolve_hf_model_dir(Path(args.base_checkpoint_step))

    _validate_inputs(data_root, chat_template_path, base_checkpoint)
    selected = _write_overfit_subset(
        source_train_file=data_root / "sft_train.jsonl",
        train_file=train_file,
        validation_file=validation_file,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    _print_subset_summary(selected, train_file, validation_file)

    if args.prepare_only:
        print("prepare-only requested; not launching training.")
        return

    cmd = [
        sys.executable,
        str(ROOT / "scripts/python/launch_sft.py"),
        "--config",
        str(args.config),
        f"project.experiment_name={output_dir.name}",
        f"project.seed={args.seed}",
        f"project.output_dir={output_dir}",
        f"runtime.visible_devices={args.visible_devices}",
        "runtime.mixed_precision=auto",
        f"model.base_checkpoint={base_checkpoint}",
        f"data.dataset_root={subset_dir}",
        f"data.train_file={train_file}",
        f"data.validation_file={validation_file}",
        f"data.chat_template_path={chat_template_path}",
        "data.max_length=256",
        "data.packing=false",
        "train.num_train_epochs=999",
        f"train.max_steps={args.max_steps}",
        f"train.per_device_train_batch_size={args.per_device_train_batch_size}",
        f"train.gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"train.learning_rate={args.learning_rate}",
        "train.logging_steps=1",
        f"train.eval_steps={args.eval_steps}",
        f"train.save_steps={args.save_steps}",
        "train.save_total_limit=2",
        "accuracy_eval.enabled=true",
        f"accuracy_eval.train_examples={args.num_examples}",
        f"accuracy_eval.val_examples={args.num_examples}",
        "accuracy_eval.batch_size=32",
        f"accuracy_eval.max_new_tokens={args.max_new_tokens}",
        "wandb.enabled=false",
    ]
    print("Launching overfit sanity check:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / ".cache/huggingface/datasets"))
    env.setdefault("TRANSFORMERS_CACHE", str(ROOT / ".cache/huggingface/transformers"))
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)

    _print_latest_accuracy(output_dir)
    if not args.skip_generation:
        _write_generation_samples(
            model_dir=output_dir / "final_model",
            validation_file=validation_file,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            max_examples=args.generation_examples,
        )


def _resolve_hf_model_dir(path: Path) -> Path:
    if (path / "hf_model").is_dir():
        return path / "hf_model"
    return path


def _validate_inputs(data_root: Path, chat_template_path: Path, base_checkpoint: Path) -> None:
    required_paths = [
        data_root / "sft_train.jsonl",
        chat_template_path,
        base_checkpoint / "config.json",
        base_checkpoint / "tokenizer.json",
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(str(p) for p in missing))


def _write_overfit_subset(
    *,
    source_train_file: Path,
    train_file: Path,
    validation_file: Path,
    num_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_examples <= 0:
        raise ValueError("--num-examples must be positive.")

    rows = _read_jsonl(source_train_file)
    selected = _select_stratified(rows, num_examples=num_examples, seed=seed)
    train_rows = [_clone_for_split(row, "train") for row in selected]
    validation_rows = [_clone_for_split(row, "validation") for row in selected]

    train_file.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(train_file, train_rows)
    _write_jsonl(validation_file, validation_rows)
    return selected


def _select_stratified(
    rows: list[dict[str, Any]],
    *,
    num_examples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qa_type = row.get("metadata", {}).get("qa_type")
        if qa_type in QA_TYPE_ORDER:
            groups[str(qa_type)].append(row)

    missing = [qa_type for qa_type in QA_TYPE_ORDER if not groups.get(qa_type)]
    if missing:
        raise ValueError(f"Missing QA type(s) in source train data: {missing}")

    rng = random.Random(seed)
    target_counts = _balanced_counts(num_examples, len(QA_TYPE_ORDER))
    selected: list[dict[str, Any]] = []
    for qa_type, count in zip(QA_TYPE_ORDER, target_counts, strict=True):
        candidates = list(groups[qa_type])
        rng.shuffle(candidates)
        if count > len(candidates):
            raise ValueError(f"Requested {count} {qa_type} rows, only found {len(candidates)}.")
        selected.extend(candidates[:count])

    selected.sort(key=lambda row: int(row.get("metadata", {}).get("sample_index", 0)))
    return selected


def _balanced_counts(total: int, buckets: int) -> list[int]:
    base = total // buckets
    remainder = total % buckets
    return [base + (1 if index < remainder else 0) for index in range(buckets)]


def _clone_for_split(row: dict[str, Any], split: str) -> dict[str, Any]:
    cloned = copy.deepcopy(row)
    metadata = cloned.setdefault("metadata", {})
    metadata["split"] = split
    metadata["overfit_sanity"] = True
    metadata["overfit_source_split"] = "train"
    return cloned


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_subset_summary(
    selected: list[dict[str, Any]],
    train_file: Path,
    validation_file: Path,
) -> None:
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        counts[str(row.get("metadata", {}).get("qa_type"))] += 1
    print("Wrote overfit subset:")
    print(f"  train_file: {train_file}")
    print(f"  validation_file: {validation_file}")
    for qa_type in QA_TYPE_ORDER:
        print(f"  {qa_type}: {counts[qa_type]}")


def _print_latest_accuracy(output_dir: Path) -> None:
    state_path = _latest_trainer_state(output_dir)
    if state_path is None:
        print("No trainer_state.json found; skipping metric summary.")
        return

    state = json.loads(state_path.read_text(encoding="utf-8"))
    for row in reversed(state.get("log_history", [])):
        if "sft_acc/train/acc" in row:
            print(f"Latest accuracy metrics from {state_path}:")
            for key in sorted(k for k in row if k.startswith("sft_acc/")):
                print(f"  {key}: {row[key]}")
            return
    print(f"No sft_acc metrics found in {state_path}.")


def _latest_trainer_state(output_dir: Path) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*/trainer_state.json"):
        try:
            step = int(path.parent.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return sorted(candidates)[-1][1]


def _write_generation_samples(
    *,
    model_dir: Path,
    validation_file: Path,
    output_dir: Path,
    max_new_tokens: int,
    max_examples: int,
) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from safe_pretrain.train.sft_accuracy import (
        _chat_generation_prompt,
        _generate_qa_predictions,
        parse_answer_items,
    )

    if not model_dir.exists():
        print(f"Model dir not found for generation: {model_dir}")
        return

    rows = _read_jsonl(validation_file)[:max_examples]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    prompts = [_chat_generation_prompt(row, tokenizer) for row in rows]
    predictions = _generate_qa_predictions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=8,
        max_new_tokens=max_new_tokens,
    )

    jsonl_path = output_dir / "overfit_generations.jsonl"
    md_path = output_dir / "overfit_generations.md"
    records = []
    for row, prompt, prediction in zip(rows, prompts, predictions, strict=True):
        gold = row["messages"][1]["content"]
        gold_items = parse_answer_items(gold, tokenizer)
        pred_items = parse_answer_items(prediction, tokenizer)
        records.append(
            {
                "qa_type": row.get("metadata", {}).get("qa_type"),
                "relation_id": row.get("metadata", {}).get("relation_id"),
                "chat_prompt": prompt,
                "gold_assistant": gold,
                "prediction": prediction,
                "exact_set_match": set(gold_items) == set(pred_items),
                "gold_items": gold_items,
                "prediction_items": pred_items,
            }
        )

    _write_jsonl(jsonl_path, records)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# SFT Overfit Sanity Generations\n\n")
        handle.write(f"Model: `{model_dir}`\n\n")
        for index, record in enumerate(records, 1):
            handle.write(f"## {index}. {record['qa_type']}\n\n")
            handle.write(f"- relation_id: `{record['relation_id']}`\n")
            handle.write(f"- exact_set_match: `{record['exact_set_match']}`\n\n")
            handle.write("### Chat prompt\n\n```text\n")
            handle.write(record["chat_prompt"])
            handle.write("\n```\n\n")
            handle.write("### Gold assistant\n\n```text\n")
            handle.write(record["gold_assistant"])
            handle.write("\n```\n\n")
            handle.write("### Model prediction\n\n```text\n")
            handle.write(record["prediction"])
            handle.write("\n```\n\n")

    exact = sum(1 for record in records if record["exact_set_match"])
    print(f"Wrote generation samples: {md_path}")
    print(f"Wrote generation jsonl: {jsonl_path}")
    print(f"Generation sample exact match: {exact}/{len(records)}")


if __name__ == "__main__":
    main()
