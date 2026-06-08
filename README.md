# Safe Pretrain

This repository contains a controlled synthetic experiment for testing whether a
language model pretrained only on safe text can still expose unsafe behavior
after supervised finetuning (SFT). The code builds synthetic cause-effect
worlds, filters direct unsafe reverse examples out of pretraining and safe SFT
data, pretrains a SmolLM2-135M architecture from scratch, and evaluates whether
safe QA supervision unlocks restricted reverse answers.

Generated data, checkpoints, logs, and report drafts are intentionally not
tracked by git. The tracked repository contains only the reusable experiment
code, configs, bash launchers, tests, and environment files.

## Repository Layout

```text
configs/   Hydra/OmegaConf configs for data, pretraining, and SFT
scripts/   Bash launchers and Python entrypoints
src/       safe_pretrain Python package
tests/     Unit and smoke tests for data, eval, and runtime utilities
```

Important entrypoints:

```text
scripts/bash/run_experiment_pipeline.sh      one complete data -> pretrain -> SFT -> eval pipeline
scripts/bash/run_all_final_experiments.sh    dependency-aware final experiment scheduler
scripts/bash/preflight_final_experiments.sh  environment, syntax, dry-run, disk, and GPU checks
```

## Environment Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate safe-pretrain
pip install -e .
```

The experiments use the HuggingFace `HuggingFaceTB/SmolLM2-135M` tokenizer and
architecture config. The model weights are not used as initialization:
pretraining starts from a fresh config-initialized model. The first run needs
network access or a populated HuggingFace cache.

Run the tests:

```bash
pytest -q
```

Before launching final experiments on a GPU machine, run:

```bash
REQUIRE_GPU=true MIN_FREE_GB=500 RUN_TESTS=false \
  bash scripts/bash/preflight_final_experiments.sh
```

For a CPU/login-node syntax and config check:

```bash
REQUIRE_GPU=false RUN_TESTS=false \
  bash scripts/bash/preflight_final_experiments.sh
```

## Synthetic Safety Definition

Each synthetic world contains causes and effects. A relation maps one or more
causes to an effect. Some effects are restricted. Forward descriptions
(`cause -> effect`) are considered safe descriptive text, even for restricted
effects. Reverse descriptions (`effect -> cause`) are safe only for unrestricted
effects. Direct restricted reverse text is unsafe and is excluded from
pretraining and safe SFT splits.

Evaluation separates:

- **Pretrain completion accuracy:** greedy completion on pretraining memory and
  held-out templates, to check whether the model learned the synthetic facts.
- **Safe QA accuracy:** exact-match accuracy on safe user queries from relations
  held out of safe SFT training. The split is controlled by
  `sft.test_relation_fraction`.
- **Restricted reverse ASR:** exact-match attack success rate on restricted
  reverse queries that were never directly demonstrated in safe train splits.
  By default, ASR uses restricted relations that were seen in safe SFT
  directions, isolating unsafe behavior generalization from relation recall.

All data builders include leakage audits; the final runs report zero direct
restricted reverse records in safe pretraining and safe SFT splits.

## Run One Complete Experiment

The simplest full run is the canonical OCR recipe:

```bash
PYTHON_BIN="$(which python)" \
VISIBLE_DEVICES=0,1,2,3 \
MAIN_PROCESS_PORT=29510 \
RUN_DATA=true \
RUN_PRETRAIN=true \
RUN_EVAL_PRETRAIN=true \
RUN_SFT=true \
RUN_EVAL_SFT=true \
bash scripts/bash/final_experiments/block_a_ocr_vertical/a0_ocr_canonical_plain_k6_wd1.sh
```

This performs:

```text
synthetic data -> tokenized pretraining corpus -> pretraining -> pretrain completion eval
               -> safe SFT -> safe QA eval + restricted reverse attack eval
```

Outputs are written under:

```text
data/experiments/<experiment>/        generated world/data artifacts
outputs/<pretrain-run>/               pretrain checkpoints and completion eval
outputs/<sft-run>/                    SFT checkpoints and QA eval
logs/                                 launcher logs
```

The launcher resumes or skips completed matching checkpoints by default. To
force a data rebuild, use a new experiment/output name or explicitly set
`OVERWRITE_DATA=true` for an individual experiment. The main scheduler refuses
`OVERWRITE_DATA=true` to avoid accidental races.

For result reporting, use `eval_safe.by_relation_group.open_sft_unseen` as the
safe open generalization metric and `attack.asr_restricted_all` as the
restricted reverse attack metric.

## Final Experiment Recipes

The final recipe collection is encoded as bash wrappers under
`scripts/bash/final_experiments/`:

```text
final experiment recipes
|-- block_a_ocr_vertical
|   |-- a0_ocr_canonical_plain_k6_wd1.sh
|   |-- a1_fi_k_sweep
|   |   |-- fi_k1.sh
|   |   |-- fi_k2.sh
|   |   |-- fi_k4.sh
|   |   |-- fi_k6.sh
|   |   `-- fi_k8.sh
|   |-- a2_chat_template
|   |   |-- chatml_train_k6.sh
|   |   `-- eval_cross_templates_final.sh
|   |-- a3_weight_decay
|   |   |-- wd0p1.sh
|   |   |-- wd1.sh
|   |   `-- wd2.sh
|   |-- a4_relation_count
|   |   |-- rel512.sh
|   |   |-- rel1024.sh
|   |   `-- rel2048.sh
|   `-- a5_arity_overlap
|       |-- arity1_strict_reference.sh
|       `-- arity2_overlap.sh
`-- block_b_family_level
    |-- b0_vanilla_control.sh
    |-- b1_ocr_main_reference.sh
    |-- b2_ocr_linear.sh
    |-- b3_prevention.sh
    `-- b4_mirror.sh
```

Block A studies the OCR setting vertically: bridge exposure, chat template,
pretraining weight decay, relation count, and arity/overlap. Block B compares
families under a shared geometry: vanilla, OCR, OCR-linear, prevention, and
mirror.

## Run the Full Final Matrix

By default the scheduler runs one 4-GPU job at a time:

```bash
PYTHON_BIN="$(which python)" \
bash scripts/bash/run_all_final_experiments.sh
```

To use eight GPUs as two concurrent 4-GPU slots:

```bash
PYTHON_BIN="$(which python)" \
FINAL_EXPERIMENT_CONCURRENCY=2 \
SLOT0_VISIBLE_DEVICES=0,1,2,3 \
SLOT1_VISIBLE_DEVICES=4,5,6,7 \
SLOT0_MAIN_PROCESS_PORT=29510 \
SLOT1_MAIN_PROCESS_PORT=29520 \
bash scripts/bash/run_all_final_experiments.sh
```

The scheduler first runs the canonical OCR recipe as a dependency barrier,
skips alias recipes that would write the same outputs, then schedules the
remaining unique recipes and runs cross-template eval last. Logs are written to
`logs/final_experiments/<timestamp>/`.

For a no-op scheduler check:

```bash
SCHEDULER_DRY_RUN=true FINAL_EXPERIMENT_CONCURRENCY=2 \
  bash scripts/bash/run_all_final_experiments.sh
```

## Core Final Results

Metrics in the table use the following standards. `Pretrain memory` is
completion exact match on sampled pretraining rows. `Pretrain template` is
completion exact match on held-out templates. `Heldout open QA` is exact-match
safe QA on open relations held out from safe SFT training. `Restricted reverse
ASR` is exact-match attack success on restricted reverse queries whose direct
answers never appear in safe pretraining or safe SFT.

| Setting | Pretrain memory | Pretrain template | Heldout open QA | Restricted reverse ASR |
| --- | ---: | ---: | ---: | ---: |
| OCR canonical | 84.28 | 65.82 | 93.21 | 69.57 |
| OCR, FI repeats 1 | 84.28 | 65.82 | 92.93 | 63.04 |
| OCR, FI repeats 2 | 84.28 | 65.82 | 92.12 | 69.57 |
| OCR, FI repeats 4 | 84.28 | 65.82 | 91.58 | 68.48 |
| OCR, FI repeats 8 | 84.28 | 65.82 | 94.02 | 64.13 |
| OCR, ChatML train/eval | 84.28 | 65.82 | 0.00 | 0.00 |
| OCR, weight decay 0.1 | 81.69 | 64.80 | 94.02 | 52.17 |
| OCR, weight decay 2.0 | 83.40 | 64.98 | 92.93 | 67.39 |
| OCR, 512 effects | 85.16 | 65.55 | 95.65 | 52.17 |
| OCR, 2048 effects | 71.73 | 55.63 | 86.14 | 51.89 |
| OCR, arity-2 overlap | 75.12 | 56.89 | 70.38 | 10.87 |
| Vanilla control | 100.00 | 95.02 | 23.37 | 0.00 |
| OCR-linear | 99.98 | 78.11 | 36.96 | 53.26 |
| Prevention | 72.12 | 74.77 | 81.52 | 100.00 |
| Mirror | 100.00 | 100.00 | 45.65 | 52.17 |

Main takeaways:

- The canonical OCR model learns the synthetic facts during safe pretraining
  and reaches `93.21%` heldout open QA after SFT, while restricted reverse ASR
  is `69.57%`.
- The effect is not caused by direct leakage: safe pretraining and safe SFT
  splits contain zero direct restricted reverse examples.
- OCR robustness checks do not show a monotone control knob; they show that
  high ASR persists across several benign training choices.
- Chat interface matters: matched plain-text chat gives high ASR, while matched
  ChatML gives `0.00%` heldout open QA and `0.00%` ASR in this small-from-scratch
  setting.
- The arity-2 overlap world sharply reduces both heldout open QA and ASR,
  indicating that the canonical OCR result relies on the simpler single-cause
  lookup structure.
- Prevention is the strongest safety-pretraining analogy in the final family
  comparison, with `81.52%` heldout open QA and `100.00%` restricted reverse
  ASR after safe SFT.

## Manual Entrypoints

Build data only:

```bash
python scripts/python/build_synthetic_dataset.py \
  --config configs/synthetic_dataset.yaml \
  --stage all \
  experiment.name=smoke_ocr \
  experiment.overwrite=true \
  dataset.family=ocr \
  sft.chat_template=plain
```

Pretrain from tokenized data:

```bash
python scripts/python/launch_pretrain.py \
  --config configs/pretrain_a6000_smollm2_135m.yaml \
  project.output_dir=outputs/pt-smoke \
  data.tokenized.path=data/experiments/smoke_ocr/pretrain/tokenized/bs512
```

Run SFT from a completed pretrain checkpoint:

```bash
python scripts/python/launch_sft.py \
  --config configs/sft_qa_smollm2_135m.yaml \
  project.output_dir=outputs/sft-smoke \
  model.base_checkpoint=outputs/pt-smoke/checkpoints/latest/hf_model \
  data.dataset_root=data/experiments/smoke_ocr/sft_plain \
  data.train_file=data/experiments/smoke_ocr/sft_plain/sft_train.jsonl \
  data.validation_file=data/experiments/smoke_ocr/sft_plain/sft_validation.jsonl \
  data.attack_file=data/experiments/smoke_ocr/sft_plain/eval_attack.jsonl \
  data.chat_template_path=data/experiments/smoke_ocr/sft_plain/chat_template.jinja
```

Evaluate saved checkpoints:

```bash
python scripts/python/eval_pretrain_completion.py \
  --model outputs/pt-smoke/checkpoints/latest/hf_model \
  --pretrain-dir data/experiments/smoke_ocr/pretrain \
  --output-dir outputs/pt-smoke/eval/pretrain_completion

python scripts/python/eval_sft_qa.py \
  --model outputs/sft-smoke/final_model \
  --sft-dir data/experiments/smoke_ocr/sft_plain \
  --output-dir outputs/sft-smoke/eval/sft_qa/final_model
```
