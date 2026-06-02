# Safe Pretrain

Lightweight infrastructure for causal language-model pretraining experiments
using the SmolLM2-135M architecture, Transformers, Accelerate, Datasets, and
WandB.

The current code includes pretraining infrastructure and a synthetic world
generator. SFT and task-specific safety evaluation can be added on top of the
same fixed-world metadata and raw text interfaces.

## Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate safe-pretrain
pip install -e .
```

Check CUDA visibility:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## Main Config

The experiment is controlled by one file:

```text
configs/pretrain_a6000_smollm2_135m.yaml
```

It contains runtime, model, data, dataloader, optimization, logging, and
checkpoint settings. You can override any value from the command line:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  project.experiment_name=test-run runtime.visible_devices=0,1 train.learning_rate=2.0e-4
```

Training outputs are saved under `outputs/${project.experiment_name}` by
default. Change only `project.experiment_name` to start a new run directory.

## Data Format

For JSONL input, use one text record per line:

```json
{"text": "This is one training document."}
{"text": "This is another training document."}
```

Configure paths in:

```yaml
data:
  raw:
    format: jsonl
    train_files:
      - data/raw/train.jsonl
    validation_files:
      - data/raw/valid.jsonl
    text_column: text
```

Supported raw formats are `jsonl`, `json`, `csv`, `parquet`, `text`, and `hf`.

## Synthetic Data

Synthetic data generation is intentionally split into fixed world generation,
pretrain rendering, and SFT QA rendering.

Create a fixed world groundtruth:

```bash
python scripts/python/create_synthetic_world.py \
  --config configs/synthetic_world.yaml
```

This writes:

```text
data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/
  world_manifest.json
  vocab/
  relations.jsonl
  splits.json
  audit_world.json
```

Render a pretraining dataset from that fixed world:

```bash
bash scripts/bash/render_pretrain_data.sh
```

This writes:

```text
data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/pretrain/0.0reverse_0.99train_composition_v1/
  composition_manifest.json
  experiment_splits.json
  pretrain_train.jsonl
  pretrain_validation.jsonl
  render_manifest.json
  audit_render.json
  tokenized/
```

`world_manifest.json` is the groundtruth source of truth. Pretrain, SFT, and
evaluation data should all reference the same `world_id`. The fixed world stores
the open/restricted oracle partition; train/validation and experiment-specific
target assignments are rendered-dataset metadata.

Pretrain render only controls corpus size, train/validation split, and the
`forward` versus `reverse` sentence ratio. The base corpus preserves the
world's open/restricted partition ratio, then selected open records are rendered
as `reverse`. `reverse_ratio` cannot exceed the world's open relation ratio, so
restricted reverse records do not leak into pretraining.

Render chat-format SFT QA data from the same fixed world:

```bash
bash scripts/bash/render_sft_data.sh
```

This writes:

```text
data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/sft/qa_1ex_0.8train_composition_v1/
  composition_manifest.json
  chat_template.jinja
  sft_splits.json
  sft_train.jsonl
  sft_validation.jsonl
  sft_test_safe.jsonl
  eval_attack.jsonl
  sft_manifest.json
  audit_sft.json
```

Pretrain and SFT share the same connector vocabulary but use different wrappers.
The allowed exposure matrix is:

```text
pretrain: forward_open, forward_restricted, optional reverse_open
sft train/validation/test_safe: forward_open, forward_restricted, reverse_open
attack eval: reverse_restricted
```

## Tokenize

Tokenization and packing are offline:

```bash
python scripts/python/tokenize_dataset.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml
```

This writes a packed Hugging Face dataset to:

```text
data/worlds/synthetic_world_4096effects_8192causes_0.5restricted_3arity_wo_overlap/pretrain/0.0reverse_0.99train_composition_v1/tokenized/bs2048
```

The training loop consumes fixed-size token blocks, so it does not run the
tokenizer during GPU training.

`data.tokenized.block_size` is a tokenization-time setting. Training reads the
actual block size from the saved tokenized dataset metadata, so you do not need
to manually repeat it when launching training.
The default `data.tokenized.path` lives under the rendered pretrain dataset
folder so the raw JSONL and packed Hugging Face dataset keep the same world and
render lineage.

For this synthetic task, the documents are likely short. Use `128` or `256` for
smoke tests, and start formal experiments with `512` or `1024`. Move to `2048`
only if throughput is good and you want the standard small-LM pretraining setup.

## Train

Launch through the wrapper:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml
```

Select GPUs in the same config:

```yaml
runtime:
  visible_devices: "0,1,2,3"
  num_processes: auto
```

Or override from the command line:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  runtime.visible_devices=1,3
```

The wrapper sets `CUDA_VISIBLE_DEVICES`, counts the selected GPUs, and calls
`accelerate launch --num_processes N`.

## Precision on A6000

The default precision setting is:

```yaml
runtime:
  mixed_precision: auto
```

`auto` chooses BF16 when PyTorch reports BF16 support, otherwise FP16 on CUDA,
and full precision on CPU. BF16 is the preferred default for RTX A6000 because
from-scratch pretraining is more robust to early loss spikes and does not need
FP16 loss scaling.

You can force FP16:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  runtime.mixed_precision=fp16
```

Run a short precision comparison:

```bash
python scripts/python/benchmark_precision.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  --steps 100
```

## Checkpoints

Checkpoints are saved under:

```text
outputs/smollm2-135m-pretrain-a6000/checkpoints/
```

Each checkpoint contains:

```text
step-0001000/
  accelerator_state/
  hf_model/
  trainer_state.json
latest -> step-0001000
```

Resume with:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  checkpoint.resume_from=outputs/smollm2-135m-pretrain-a6000/checkpoints/latest
```

## Logged Metrics

If `wandb.enabled=true`, the main process logs:

- `train/loss`
- `train/ppl`
- `train/lr`
- `train/grad_norm`
- `train/tokens_seen`
- `train/tokens_per_sec`
- `train/samples_per_sec`
- `eval/loss`
- `eval/ppl`
- `checkpoint/step`
- `perf/data_time_sec`
- `perf/fwd_bwd_time_sec`
- `perf/optimizer_time_sec`
- `perf/step_time_sec`
- `perf/data_time_ratio`
- `perf/gpu_memory_allocated_gb`
- `perf/gpu_memory_peak_allocated_gb`

The profiler is configured in the same YAML:

```yaml
profiler:
  enabled: true
  synchronize_cuda: false
  log_memory: true
```

Set `profiler.synchronize_cuda=true` only when you need more accurate timing
for forward/backward and optimizer regions, because it adds CUDA synchronization
overhead.

Disable WandB for local smoke tests:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  wandb.enabled=false train.max_train_steps=20
```

## Smoke Test Script

The smoke script assumes the tokenized dataset already exists at
`data/tokenized/smoke_bs512`. Run it with one command from your current
environment:

```bash
bash scripts/bash/run_smoke_pretrain.sh
```

Edit the variables at the top of the script to change the experiment name,
tokenized dataset path, GPU list, batch size, WandB setting, and logging/eval
cadence.

## QA SFT

The SFT target format is QA-only chat JSONL. Canonical records use `messages`
plus metadata:

```json
{"messages": [{"role": "user", "content": "Complete the relation: golden tone is listed with causes?"}, {"role": "assistant", "content": "jopejobi, kadafobi, tajofobi"}], "metadata": {"stage": "sft", "qa_type": "reverse_open", "split": "train"}}
```

SFT uses the generated ChatML-style `chat_template.jinja` and trains with
assistant-only loss, so user/query tokens are context and only the assistant
answer is supervised.

Launch with:

```bash
bash scripts/bash/run_sft.sh
```

Edit the variables at the top of `scripts/bash/run_sft.sh` to choose the checkpoint,
SFT train/validation files, GPUs, max length, packing, batch size, learning
rate, and logging cadence. SFT outputs are saved under
`outputs/${project.experiment_name}` with TRL checkpoints and `final_model/`.
During validation, the SFT loop can also run a small generation probe and log
`sft_acc/train/acc`, `sft_acc/val/acc`, task-level `forward_acc`,
`reverse_open_acc`, and overall answer `format`. It does not run ASR during
training; `eval_attack.jsonl` is for a separate post-SFT evaluation.

This path intentionally does not accept generic text/declarative SFT records.
Pretraining handles world fact memorization; SFT only trains the QA interface
and answer extraction behavior.

## Repository Layout

```text
configs/
  pretrain_a6000_smollm2_135m.yaml
  sft_qa_smollm2_135m.yaml
  synthetic_world.yaml
  synthetic_pretrain_render.yaml
  synthetic_sft_qa.yaml
doc/
  pretrain_infra.md
  synthetic_world_pretrain_design.md
  synthetic_world_experiments.md
scripts/
  bash/
    run_eval_world.sh
    run_pretrain.sh
    run_sft.sh
    run_smoke_pretrain.sh
    render_pretrain_data.sh
    render_sft_data.sh
    tokenize_pretrain.sh
  python/
    benchmark_precision.py
    create_synthetic_world.py
    eval_world.py
    launch_pretrain.py
    launch_sft.py
    render_synthetic_pretrain.py
    render_synthetic_sft_qa.py
    tokenize_dataset.py
    train_pretrain.py
    train_sft.py
src/safe_pretrain/
  data/
  synthetic/
  train/
  utils/
```

See `doc/pretrain_infra.md` for the design rationale and extension points.
