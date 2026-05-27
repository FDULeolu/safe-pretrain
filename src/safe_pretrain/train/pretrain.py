from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from datasets import DatasetDict, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

from safe_pretrain.config import to_plain_container
from safe_pretrain.data.collator import causal_lm_collator
from safe_pretrain.train.checkpoint import (
    load_trainer_state,
    resolve_resume_dir,
    save_training_checkpoint,
)
from safe_pretrain.train.metrics import ProfilerWindow, ThroughputMeter, perplexity
from safe_pretrain.train.optim import build_optimizer
from safe_pretrain.utils.runtime import configure_torch, resolve_mixed_precision
from safe_pretrain.utils.seed import seed_everything


def run_pretraining(cfg: Any) -> None:
    seed_everything(int(cfg.project.seed))
    configure_torch(bool(cfg.runtime.get("tf32", True)))
    mixed_precision = resolve_mixed_precision(cfg.runtime.get("mixed_precision", "auto"))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.train.gradient_accumulation_steps),
        mixed_precision=mixed_precision,
        project_dir=str(cfg.project.output_dir),
    )

    tokenizer = _load_tokenizer(cfg)
    model = _build_model(cfg, tokenizer)
    dataset = _load_tokenized_dataset(cfg.data.tokenized.path)
    resolved_block_size = _resolve_tokenized_block_size(cfg.data.tokenized.path, dataset)
    train_dataset = dataset["train"]
    eval_dataset = _get_eval_dataset(dataset)
    wandb_run = _maybe_init_wandb(cfg, accelerator, mixed_precision, resolved_block_size)

    train_loader = _build_dataloader(train_dataset, cfg, shuffle=True)
    eval_loader = _build_dataloader(eval_dataset, cfg, shuffle=False) if eval_dataset is not None else None
    optimizer = build_optimizer(model, cfg)

    if eval_loader is not None:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(
            model, optimizer, train_loader, eval_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    updates_per_epoch = math.ceil(
        len(train_loader) / int(cfg.train.gradient_accumulation_steps)
    )
    max_train_steps = cfg.train.get("max_train_steps")
    if max_train_steps is None:
        max_train_steps = int(cfg.train.num_train_epochs) * updates_per_epoch
    else:
        max_train_steps = int(max_train_steps)
    num_train_epochs = max(
        int(cfg.train.num_train_epochs),
        math.ceil(max_train_steps / max(updates_per_epoch, 1)),
    )
    warmup_steps = int(float(cfg.train.warmup_ratio) * max_train_steps)
    scheduler = get_scheduler(
        name=str(cfg.train.scheduler),
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    scheduler = accelerator.prepare(scheduler)

    resume_dir = resolve_resume_dir(cfg.checkpoint.get("resume_from"))
    trainer_state = load_trainer_state(resume_dir)
    if resume_dir is not None:
        accelerator.print(f"Resuming from {resume_dir}")
        accelerator.load_state(str(resume_dir / "accelerator_state"))

    completed_steps = int(trainer_state.get("global_step", 0))
    tokens_seen = int(trainer_state.get("tokens_seen", 0))
    samples_seen = int(trainer_state.get("samples_seen", 0))
    start_epoch = int(trainer_state.get("epoch", 0))
    resume_micro_step = int(trainer_state.get("micro_step_in_epoch", 0))

    accelerator.print(
        "Training with "
        f"{accelerator.num_processes} process(es), mixed_precision={mixed_precision}, "
        f"block_size={resolved_block_size}, max_train_steps={max_train_steps}"
    )

    progress = tqdm(
        total=max_train_steps,
        initial=completed_steps,
        disable=not accelerator.is_main_process,
        desc="pretrain",
    )
    meter = ThroughputMeter(tokens_seen, samples_seen, time.perf_counter())
    profiler = ProfilerWindow(enabled=bool(cfg.profiler.get("enabled", True)))
    running_loss = 0.0
    running_loss_count = 0
    last_grad_norm = float("nan")
    should_stop = completed_steps >= max_train_steps
    last_epoch = start_epoch

    model.train()
    for epoch in range(start_epoch, num_train_epochs):
        last_epoch = epoch
        active_loader = train_loader
        skipped_this_epoch = 0
        if resume_micro_step and epoch == start_epoch:
            active_loader = accelerator.skip_first_batches(train_loader, resume_micro_step)
            skipped_this_epoch = resume_micro_step

        next_batch_start = time.perf_counter()
        for local_micro_step, batch in enumerate(active_loader):
            data_time = time.perf_counter() - next_batch_start
            micro_step_in_epoch = skipped_this_epoch + local_micro_step + 1
            _maybe_sync_cuda(accelerator, cfg)
            step_start = time.perf_counter()
            with accelerator.accumulate(model):
                fwd_bwd_start = time.perf_counter()
                outputs = model(**batch)
                loss = outputs.loss
                reduced_loss = accelerator.reduce(loss.detach(), reduction="mean")
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        model.parameters(),
                        float(cfg.train.max_grad_norm),
                    )
                    last_grad_norm = _as_float(grad_norm)

                _maybe_sync_cuda(accelerator, cfg)
                fwd_bwd_time = time.perf_counter() - fwd_bwd_start
                optimizer_start = time.perf_counter()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                _maybe_sync_cuda(accelerator, cfg)
                optimizer_time = time.perf_counter() - optimizer_start

            step_time = time.perf_counter() - step_start
            profiler.record(data_time, fwd_bwd_time, optimizer_time, step_time)
            batch_tokens, batch_samples = _global_batch_sizes(accelerator, batch)
            tokens_seen += batch_tokens
            samples_seen += batch_samples
            running_loss += _as_float(reduced_loss)
            running_loss_count += 1

            if accelerator.sync_gradients:
                completed_steps += 1
                progress.update(1)

                if completed_steps % int(cfg.train.log_every_steps) == 0:
                    avg_loss = running_loss / max(running_loss_count, 1)
                    rates = meter.rates(tokens_seen, samples_seen)
                    log_payload = {
                        "train/loss": avg_loss,
                        "train/ppl": perplexity(avg_loss),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/grad_norm": last_grad_norm,
                        "train/tokens_seen": tokens_seen,
                        "train/tokens_per_sec": rates["tokens_per_sec"],
                        "train/samples_per_sec": rates["samples_per_sec"],
                        "train/global_step": completed_steps,
                        "data/block_size": resolved_block_size,
                    }
                    log_payload.update(_profiler_metrics(accelerator, profiler, cfg))
                    _log_metrics(wandb_run, log_payload, completed_steps)
                    running_loss = 0.0
                    running_loss_count = 0
                    meter.reset(tokens_seen, samples_seen)
                    profiler.reset()

                if eval_loader is not None and completed_steps % int(cfg.train.eval_every_steps) == 0:
                    eval_metrics = evaluate(
                        accelerator,
                        model,
                        eval_loader,
                        int(cfg.train.get("max_eval_batches", 0) or 0),
                    )
                    _log_metrics(wandb_run, eval_metrics, completed_steps)

                if completed_steps % int(cfg.checkpoint.save_every_steps) == 0:
                    state = _trainer_state(
                        completed_steps,
                        tokens_seen,
                        samples_seen,
                        epoch,
                        micro_step_in_epoch,
                        resolved_block_size,
                    )
                    save_training_checkpoint(
                        accelerator,
                        model,
                        tokenizer,
                        cfg,
                        completed_steps,
                        state,
                    )
                    _log_metrics(
                        wandb_run,
                        {"checkpoint/step": completed_steps},
                        completed_steps,
                    )

                if completed_steps >= max_train_steps:
                    should_stop = True
                    break

            next_batch_start = time.perf_counter()

        resume_micro_step = 0
        if should_stop:
            break

    if completed_steps > 0:
        state = _trainer_state(
            completed_steps,
            tokens_seen,
            samples_seen,
            last_epoch,
            0,
            resolved_block_size,
        )
        save_training_checkpoint(accelerator, model, tokenizer, cfg, completed_steps, state)

    progress.close()
    accelerator.wait_for_everyone()
    if wandb_run is not None:
        wandb_run.finish()
    accelerator.end_training()


def evaluate(
    accelerator: Accelerator,
    model: torch.nn.Module,
    eval_loader: DataLoader,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_sum = 0

    for batch_idx, batch in enumerate(eval_loader):
        if max_batches and batch_idx >= max_batches:
            break
        with torch.no_grad():
            outputs = model(**batch)
        local_tokens = torch.tensor(batch["input_ids"].numel(), device=accelerator.device)
        local_loss_sum = outputs.loss.detach() * local_tokens
        global_loss_sum = accelerator.reduce(local_loss_sum, reduction="sum")
        global_tokens = accelerator.reduce(local_tokens, reduction="sum")
        loss_sum += _as_float(global_loss_sum)
        token_sum += int(_as_float(global_tokens))

    model.train()
    eval_loss = loss_sum / max(token_sum, 1)
    return {
        "eval/loss": eval_loss,
        "eval/ppl": perplexity(eval_loss),
    }


def _load_tokenizer(cfg: Any):
    tokenizer_name = cfg.model.get("tokenizer_name_or_path") or cfg.model.name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _build_model(cfg: Any, tokenizer: Any) -> torch.nn.Module:
    model_kwargs = {
        "trust_remote_code": bool(cfg.model.get("trust_remote_code", False)),
    }
    attn_impl = cfg.model.get("attn_implementation")
    if attn_impl:
        model_kwargs["attn_implementation"] = str(attn_impl)

    if bool(cfg.model.get("init_from_config", True)):
        model_config = AutoConfig.from_pretrained(cfg.model.name_or_path, **model_kwargs)
        model = AutoModelForCausalLM.from_config(model_config, **model_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.name_or_path, **model_kwargs)

    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    if bool(cfg.model.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if bool(cfg.runtime.get("compile", False)):
        model = torch.compile(model, backend=str(cfg.runtime.get("compile_backend", "inductor")))

    return model


def _load_tokenized_dataset(path: str) -> DatasetDict:
    dataset = load_from_disk(str(Path(path)))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    if "train" not in dataset:
        raise KeyError("Tokenized dataset must contain a train split.")
    return dataset


def _resolve_tokenized_block_size(path: str, dataset: DatasetDict) -> int:
    metadata_path = Path(path) / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        block_size = int(metadata["block_size"])
        if block_size > 0:
            return block_size

    train_dataset = dataset["train"]
    if len(train_dataset) == 0:
        raise ValueError("Cannot infer block_size from an empty tokenized train split.")
    first = train_dataset[0]
    if "input_ids" not in first:
        raise KeyError("Tokenized dataset samples must contain an input_ids field.")
    block_size = len(first["input_ids"])
    if block_size <= 0:
        raise ValueError("Inferred non-positive block_size from the tokenized dataset.")
    return block_size


def _get_eval_dataset(dataset: DatasetDict):
    if "validation" in dataset:
        return dataset["validation"]
    if "valid" in dataset:
        return dataset["valid"]
    if "eval" in dataset:
        return dataset["eval"]
    return None


def _build_dataloader(dataset: Any, cfg: Any, shuffle: bool) -> DataLoader:
    num_workers = int(cfg.dataloader.num_workers)
    kwargs = {
        "dataset": dataset,
        "shuffle": shuffle,
        "batch_size": int(cfg.dataloader.per_device_batch_size),
        "collate_fn": causal_lm_collator,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.dataloader.get("pin_memory", True)),
        "drop_last": shuffle,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.dataloader.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(cfg.dataloader.get("prefetch_factor", 2))
    return DataLoader(**kwargs)


def _maybe_init_wandb(
    cfg: Any,
    accelerator: Accelerator,
    mixed_precision: str,
    block_size: int,
):
    enabled = bool(cfg.wandb.get("enabled", False))
    if not enabled or not accelerator.is_main_process:
        return None

    import wandb

    payload = to_plain_container(cfg)
    payload["runtime"]["resolved_mixed_precision"] = mixed_precision
    payload["data"]["tokenized"]["resolved_block_size"] = block_size
    payload["system"] = {
        "num_processes": accelerator.num_processes,
        "global_batch_tokens": _global_batch_tokens(
            cfg,
            accelerator.num_processes,
            block_size,
        ),
    }
    return wandb.init(
        project=str(cfg.wandb.project),
        name=str(cfg.wandb.get("run_name", cfg.project.name)),
        tags=list(cfg.wandb.get("tags", [])),
        config=payload,
    )


def _log_metrics(wandb_run: Any, payload: dict[str, float], step: int) -> None:
    if wandb_run is not None:
        wandb_run.log(payload, step=step)


def _profiler_metrics(
    accelerator: Accelerator,
    profiler: ProfilerWindow,
    cfg: Any,
) -> dict[str, float]:
    metrics = profiler.averages()
    if not bool(cfg.profiler.get("enabled", True)):
        return metrics
    if bool(cfg.profiler.get("log_memory", True)) and torch.cuda.is_available():
        device = accelerator.device
        metrics.update(
            {
                "perf/gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
                "perf/gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
                "perf/gpu_memory_peak_allocated_gb": torch.cuda.max_memory_allocated(device)
                / 1e9,
                "perf/gpu_memory_peak_reserved_gb": torch.cuda.max_memory_reserved(device)
                / 1e9,
            }
        )
        if bool(cfg.profiler.get("reset_peak_memory", True)):
            torch.cuda.reset_peak_memory_stats(device)
    return metrics


def _maybe_sync_cuda(accelerator: Accelerator, cfg: Any) -> None:
    if not bool(cfg.profiler.get("enabled", True)):
        return
    if not bool(cfg.profiler.get("synchronize_cuda", False)):
        return
    if accelerator.device.type == "cuda":
        torch.cuda.synchronize(accelerator.device)


def _global_batch_tokens(cfg: Any, num_processes: int, block_size: int) -> int:
    return (
        int(cfg.dataloader.per_device_batch_size)
        * int(block_size)
        * int(cfg.train.gradient_accumulation_steps)
        * int(num_processes)
    )


def _global_batch_sizes(accelerator: Accelerator, batch: dict[str, torch.Tensor]) -> tuple[int, int]:
    local_tokens = torch.tensor(batch["input_ids"].numel(), device=accelerator.device)
    local_samples = torch.tensor(batch["input_ids"].shape[0], device=accelerator.device)
    global_tokens = accelerator.reduce(local_tokens, reduction="sum")
    global_samples = accelerator.reduce(local_samples, reduction="sum")
    return int(_as_float(global_tokens)), int(_as_float(global_samples))


def _trainer_state(
    global_step: int,
    tokens_seen: int,
    samples_seen: int,
    epoch: int,
    micro_step_in_epoch: int,
    block_size: int,
) -> dict[str, int]:
    return {
        "global_step": int(global_step),
        "tokens_seen": int(tokens_seen),
        "samples_seen": int(samples_seen),
        "epoch": int(epoch),
        "micro_step_in_epoch": int(micro_step_in_epoch),
        "block_size": int(block_size),
    }


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)
