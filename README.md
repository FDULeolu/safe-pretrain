# Safe Pretrain

Lightweight infrastructure for causal language-model pretraining experiments
using the SmolLM2-135M architecture, Transformers, Accelerate, Datasets, and
WandB.

The current code focuses on pretraining infrastructure only. Synthetic data
generation, SFT, and task-specific safety evaluation can be added on top of the
same raw text and tokenized dataset interfaces.

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
python scripts/launch_pretrain.py \
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

## Tokenize

Tokenization and packing are offline:

```bash
python scripts/tokenize_dataset.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml
```

This writes a packed Hugging Face dataset to:

```text
data/tokenized/pretrain_smollm2_2048
```

The training loop consumes fixed-size token blocks, so it does not run the
tokenizer during GPU training.

## Train

Launch through the wrapper:

```bash
python scripts/launch_pretrain.py \
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
python scripts/launch_pretrain.py \
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
python scripts/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  runtime.mixed_precision=fp16
```

Run a short precision comparison:

```bash
python scripts/benchmark_precision.py \
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
python scripts/launch_pretrain.py \
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
python scripts/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  wandb.enabled=false train.max_train_steps=20
```

## Repository Layout

```text
configs/
  pretrain_a6000_smollm2_135m.yaml
doc/
  pretrain_infra.md
scripts/
  tokenize_dataset.py
  launch_pretrain.py
  train_pretrain.py
  benchmark_precision.py
src/safe_pretrain/
  data/
  train/
  utils/
```

See `doc/pretrain_infra.md` for the design rationale and extension points.
