# Pretraining Infrastructure Design

This document describes the lightweight pretraining pipeline used by this
repository. The goal is to support custom text datasets, single-node multi-GPU
training, metric logging, checkpoints, and easy extension without building a
heavy training platform.

## Scope

The pretraining infrastructure is intentionally separated from synthetic data
generation. Training code consumes either:

- raw text datasets with a configurable text column; or
- a tokenized Hugging Face dataset saved to disk.

Synthetic world construction, safety labels, and evaluation datasets can be
added later without changing the core pretraining loop.

## Main Decisions

### One Config File

Each experiment should be controlled by one YAML file:

```text
configs/pretrain_a6000_smollm2_135m.yaml
```

The same file contains runtime, model, data, dataloader, optimizer, logging,
checkpoint, and WandB settings. Scripts accept command-line overrides such as:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  project.experiment_name=test-run runtime.visible_devices=0,1 train.learning_rate=2.0e-4
```

By default, run artifacts are saved under:

```text
outputs/${project.experiment_name}
```

### Framework Stack

- Transformers: SmolLM2 model config, tokenizer, and model serialization.
- Accelerate: single-node DDP, mixed precision, checkpoint state, and process
  coordination.
- Datasets: raw dataset loading, multiprocessing tokenization, Arrow storage.
- WandB: metric logging.
- TRL: not used in pretraining; it can be added for SFT later.

### Model Initialization

The model uses the SmolLM2-135M architecture and initializes weights from
scratch:

```python
config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M")
model = AutoModelForCausalLM.from_config(config)
```

The tokenizer is loaded from the same model family by default.

### A6000 Precision Policy

The default is:

```yaml
runtime:
  mixed_precision: auto
```

`auto` resolves to:

1. `bf16` if CUDA reports BF16 support.
2. `fp16` if CUDA is available but BF16 is not.
3. `no` on CPU.

For RTX A6000, BF16 should normally be available. BF16 is the default preference
because pretraining from scratch is more likely to see unstable early loss
spikes under FP16. The config still allows forcing FP16:

```yaml
runtime:
  mixed_precision: fp16
```

Use the benchmark script if throughput becomes the bottleneck.

## Data Flow

### Raw Dataset Contract

Raw datasets should expose one text column. JSONL input usually looks like:

```json
{"text": "A training document."}
```

Supported raw formats:

- `jsonl` / `json`
- `csv`
- `parquet`
- `text`
- Hugging Face dataset name

### Tokenization

Tokenization is offline:

```bash
python scripts/python/tokenize_dataset.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml
```

The tokenizer script:

1. loads the raw dataset;
2. reads the configured text column;
3. appends EOS to each document by default;
4. tokenizes with multiprocessing;
5. concatenates and packs into fixed-size `block_size` chunks;
6. saves a Hugging Face `DatasetDict` to `data.tokenized.path`.

Training reads only fixed-size token blocks. This keeps the training loop simple
and avoids tokenizer overhead during GPU training.

The training loop resolves the actual block size from the saved tokenized
dataset metadata. `data.tokenized.block_size` is therefore required for
tokenization, but it does not need to be repeated or kept in sync manually when
launching training.

For synthetic world data, `data.tokenized.path` should live under the rendered
pretrain dataset directory, for example:

```text
data/worlds/<world_name>/pretrain/<render_name>/tokenized/bs2048
```

This keeps raw JSONL, tokenized blocks, templates, render manifest, and audit
files under the same world lineage.

The ready-to-run smoke path is:

```bash
bash scripts/bash/run_smoke_pretrain.sh
```

It generates `examples/smoke_data`, tokenizes to 512-token blocks, and launches
a short training run with WandB disabled.

## Training Flow

Launch through the wrapper:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml
```

The wrapper reads `runtime.visible_devices`, sets `CUDA_VISIBLE_DEVICES`, counts
the selected GPUs, and invokes `accelerate launch --num_processes N`.

The training loop:

1. resolves mixed precision;
2. configures TF32 if enabled;
3. initializes SmolLM2-135M from config;
4. loads the packed tokenized dataset;
5. resolves block size from the tokenized dataset metadata;
6. builds DataLoaders with pinned memory and persistent workers;
7. runs Accelerate DDP with gradient accumulation;
8. logs training and evaluation metrics;
9. periodically saves resumable checkpoints and Hugging Face model weights.

## Lightweight Throughput Optimizations

The default pipeline includes:

- offline tokenization and fixed-size packing;
- no runtime padding for pretraining batches;
- BF16/FP16 mixed precision;
- TF32 enabled for matmul and cuDNN;
- PyTorch SDPA attention by default;
- optional gradient checkpointing;
- optional `torch.compile`;
- fused AdamW when the local PyTorch build supports it;
- pinned-memory DataLoader workers;
- persistent workers and prefetching;
- small periodic eval instead of full validation every few steps.

FlashAttention is not a default dependency because it increases environment
fragility. It can be enabled later via config if installed.

## Metrics

WandB receives:

- `train/loss`
- `train/ppl`
- `train/lr`
- `train/grad_norm`
- `train/tokens_seen`
- `train/tokens_per_sec`
- `train/samples_per_sec`
- `train/global_step`
- `eval/loss`
- `eval/ppl`
- `system/num_processes`
- `system/global_batch_tokens`
- `checkpoint/step`
- `perf/data_time_sec`
- `perf/fwd_bwd_time_sec`
- `perf/optimizer_time_sec`
- `perf/step_time_sec`
- `perf/data_time_ratio`
- `perf/gpu_memory_allocated_gb`
- `perf/gpu_memory_peak_allocated_gb`

The lightweight profiler is enabled by default. It uses host wall-clock timing
without CUDA synchronization, which keeps overhead low and is sufficient to
identify dataloader stalls. Set `profiler.synchronize_cuda=true` for more
accurate CUDA region timing when doing focused bottleneck analysis.

Only the main process logs external metrics.

## Checkpoints

Each checkpoint contains both Accelerate state and Hugging Face weights:

```text
outputs/smollm2-135m-pretrain-a6000/checkpoints/
  step-0001000/
    accelerator_state/
    hf_model/
    trainer_state.json
  latest -> step-0001000
```

`accelerator_state` is used for exact resume. `hf_model` is used for later
evaluation, SFT, or inference.

## Extension Points

The expected next additions are:

- synthetic dataset generation that writes standard raw JSONL;
- SFT data formatting and TRL-based SFT;
- task-specific evaluation for safe accuracy and ASR;
- precision benchmarking on the target GPU set;
- optional FlashAttention installation and config switch.
