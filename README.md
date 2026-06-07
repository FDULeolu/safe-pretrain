# Safe Pretrain

Small-model synthetic pretraining and QA SFT experiments for testing whether
pretraining-time exposure to indirect causal signals enables unsafe reverse
query behavior after SFT.

The current codebase intentionally exposes one dataset pipeline and two training
entry points. Older world/render/tokenize scripts were removed so experiment
state is controlled by a single dataset config and a single shell launcher.

## Setup

```bash
conda env create -f environment.yml
conda activate safe-pretrain
pip install -e .
```

## Dataset Pipeline

The canonical config is:

```text
configs/synthetic_dataset.yaml
```

Build world metadata, pretrain JSONL, packed tokenized pretrain data, and SFT
JSONL in one command:

```bash
python scripts/python/build_synthetic_dataset.py \
  --config configs/synthetic_dataset.yaml \
  --stage all \
  "experiment.name=ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6" \
  "experiment.overwrite=true" \
  "dataset.family=ocr" \
  "sft.chat_template=plain" \
  "sft.pattern_repeats.forward_identity=6"
```

Supported dataset families:

```text
vanilla      direct forward/reverse baseline
ocr          main indirect OCR-style setting
ocr_linear   same OCR pattern set rendered as left-to-right relation chains
mirror       mirrored-token B-conditioned signal
prevention   prevention-style B-conditioned signal
```

Supported SFT templates:

```text
plain   Q:/A: template used for the no-chat-template-style SFT setting
chatml  ChatML-style role markers for template ablations
```

Dataset outputs are written under `data/experiments/${experiment.name}` by
default:

```text
world:      data/worlds/${world.name}
pretrain:  data/experiments/${experiment.name}/pretrain
tokenized: data/experiments/${experiment.name}/pretrain/tokenized/bs${tokenize.block_size}
sft:       data/experiments/${experiment.name}/sft_${sft.chat_template}
```

The SFT directory contains `sft_train.jsonl`, optional
`sft_validation.jsonl`, `eval_safe.jsonl`, `eval_attack.jsonl`, and
`chat_template.jinja`. Safe train/validation/eval splits reject direct
restricted reverse answers; restricted reverse is only emitted to
`eval_attack.jsonl`.

## One-Command Experiments

Use this launcher for the normal workflow:

```bash
FAMILY=ocr \
CHAT_TEMPLATE=plain \
SFT_REPEAT_FORWARD_IDENTITY=6 \
OVERWRITE_DATA=true \
RUN_DATA=true \
RUN_PRETRAIN=false \
RUN_SFT=false \
bash scripts/bash/run_experiment_pipeline.sh
```

On every launch with `RUN_DATA=true`, the script checks the configured world,
pretrain JSONL, tokenized pretrain dataset, and SFT directory in order. Existing
complete artifacts are reused only when their stage configuration matches the
current run. Missing or partial artifacts are rebuilt for that stage. Mismatched
artifacts fail loudly; use a new `EXPERIMENT_NAME`/`WORLD_NAME`, or set
`OVERWRITE_DATA=true` to rebuild. If the world is rebuilt, downstream
pretrain/SFT artifacts are rebuilt; if pretrain is rebuilt, the tokenized dataset
is rebuilt.

Enable pretraining by setting `RUN_PRETRAIN=true`. With the default
`PRETRAIN_AUTO_RESUME=true`, the launcher checks
`PRETRAIN_OUTPUT_DIR/checkpoints/latest`: a matching completed checkpoint skips
pretraining, and a matching incomplete checkpoint is resumed. Enable SFT by
setting `RUN_SFT=true`; if `SFT_BASE_CHECKPOINT` is empty, a matching completed
pretrain checkpoint is used automatically. Set `SFT_BASE_CHECKPOINT` explicitly
to train SFT from a different `hf_model`.
By default pretraining keeps only the latest checkpoint, while SFT keeps every
checkpoint saved every 500 steps.

Post-hoc eval is part of the same launcher. By default
`RUN_EVAL_PRETRAIN=${RUN_PRETRAIN}` and `RUN_EVAL_SFT=${RUN_SFT}`:

```text
pretrain eval -> ${PRETRAIN_OUTPUT_DIR}/eval/pretrain_completion
sft eval      -> ${SFT_OUTPUT_DIR}/eval/sft_qa/{checkpoint-name}
```

SFT eval uses `SFT_EVAL_CHECKPOINTS=all` by default, which evaluates every
`checkpoint-*` directory and `final_model`. Set `SFT_EVAL_CHECKPOINTS=final` or
a comma-separated checkpoint list for narrower reruns.

Important knobs are environment variables at the top of
`scripts/bash/run_experiment_pipeline.sh`: world size, arity, restricted
fraction, target tokens, block size, per-pattern SFT repeats, chat template,
pretrain hyperparameters, and SFT hyperparameters.
Set `PYTHON_BIN=/path/to/env/bin/python` if the launcher is run without an
activated environment.

Formal experiment wrappers live under:

```text
scripts/bash/final_experiments/
  block_a_ocr_vertical/   main OCR run plus FI, chat-template, weight-decay, scale, and arity variants
  block_b_family_level/   vanilla, OCR, OCR-linear, prevention, and mirror family comparison
```

Each wrapper delegates to `scripts/bash/run_experiment_pipeline.sh` and can be
overridden with the same environment variables, for example
`RUN_DATA=false RUN_PRETRAIN=false RUN_SFT=false` for a no-op dry run.

To launch the full final matrix with two concurrent 4-GPU slots, use:

```bash
bash scripts/bash/run_all_final_experiments.sh
```

The runner uses GPUs `0,1,2,3` on port `29510` and GPUs `4,5,6,7` on port
`29520` by default. It runs the canonical OCR experiment first, skips reference
aliases that would write the same outputs, then schedules unique downstream
experiments in pairs and runs cross-template eval last. Override
`SLOT0_VISIBLE_DEVICES`, `SLOT1_VISIBLE_DEVICES`, `SLOT0_MAIN_PROCESS_PORT`, or
`SLOT1_MAIN_PROCESS_PORT` if the machine layout or available ports differ.

Before launching a batch of formal runs, execute:

```bash
bash scripts/bash/preflight_final_experiments.sh
```

On non-GPU login nodes, use `REQUIRE_GPU=false` to run only the code/config/model
cache checks. Set `RUN_TESTS=true` to include the full pytest suite.

When `pretrain.target_records=null`, the dataset builder first renders
`pretrain.token_estimate_records` samples, tokenizes them with the configured
tokenizer, estimates average tokens per record, then chooses the record count
for `pretrain.target_tokens`. Set `TARGET_RECORDS` for fixed-size smoke runs.

## Training Entrypoints

Pretraining consumes only a packed Hugging Face dataset from disk:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  "data.tokenized.path=data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/pretrain/tokenized/bs512"
```

SFT consumes the generated QA JSONL files:

```bash
python scripts/python/launch_sft.py \
  --config configs/sft_qa_smollm2_135m.yaml \
  "model.base_checkpoint=outputs/<pretrain-run>/checkpoints/<step>/hf_model" \
  "data.dataset_root=data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/sft_plain" \
  "data.train_file=data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/sft_plain/sft_train.jsonl" \
  "data.validation_file=null" \
  "data.chat_template_path=data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/sft_plain/chat_template.jinja"
```

SFT generation accuracy during training is disabled by default. For formal
checks, evaluate saved models separately on `eval_safe.jsonl` and
`eval_attack.jsonl`.

## Pretrain Completion Eval

```bash
python scripts/python/eval_pretrain_completion.py \
  --model outputs/<pretrain-run>/checkpoints/latest/hf_model \
  --pretrain-dir data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/pretrain \
  --output-dir outputs/<pretrain-run>/eval/pretrain_completion
```

This evaluates `eval_memory.jsonl` and `eval_template_heldout.jsonl` using
greedy completion. Metrics are grouped by `exposure_type`, `pattern`,
`partition`, and `qa_type`.

## SFT QA Eval

```bash
python scripts/python/eval_sft_qa.py \
  --model outputs/<sft-run>/final_model \
  --sft-dir data/experiments/ocr_w1024-c2048-a1-r10-dic-strict_plain_fi6/sft_plain \
  --output-dir outputs/<sft-run>/eval_sft_qa
```

This writes predictions and grouped metrics for safe QA and restricted reverse
attack QA.

## Tests

```bash
conda run --no-capture-output -n safe-pretrain pytest -q
```

The focused dataset tests build small worlds for every supported family and
assert that restricted direct reverse examples never enter safe splits.
