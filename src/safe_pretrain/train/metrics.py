from __future__ import annotations

import math
import time
from dataclasses import dataclass


def perplexity(loss: float) -> float:
    if math.isnan(loss):
        return float("nan")
    return math.exp(min(loss, 20.0))


@dataclass
class ThroughputMeter:
    tokens: int = 0
    samples: int = 0
    start_time: float = time.perf_counter()

    def reset(self, tokens: int = 0, samples: int = 0) -> None:
        self.tokens = tokens
        self.samples = samples
        self.start_time = time.perf_counter()

    def rates(self, total_tokens: int, total_samples: int) -> dict[str, float]:
        elapsed = max(time.perf_counter() - self.start_time, 1e-6)
        return {
            "tokens_per_sec": (total_tokens - self.tokens) / elapsed,
            "samples_per_sec": (total_samples - self.samples) / elapsed,
        }


@dataclass
class ProfilerWindow:
    enabled: bool = True
    data_time_sec: float = 0.0
    fwd_bwd_time_sec: float = 0.0
    optimizer_time_sec: float = 0.0
    step_time_sec: float = 0.0
    count: int = 0

    def record(
        self,
        data_time_sec: float,
        fwd_bwd_time_sec: float,
        optimizer_time_sec: float,
        step_time_sec: float,
    ) -> None:
        if not self.enabled:
            return
        self.data_time_sec += data_time_sec
        self.fwd_bwd_time_sec += fwd_bwd_time_sec
        self.optimizer_time_sec += optimizer_time_sec
        self.step_time_sec += step_time_sec
        self.count += 1

    def averages(self) -> dict[str, float]:
        if not self.enabled or self.count == 0:
            return {}
        count = max(self.count, 1)
        data_time = self.data_time_sec / count
        step_time = self.step_time_sec / count
        fwd_bwd_time = self.fwd_bwd_time_sec / count
        optimizer_time = self.optimizer_time_sec / count
        return {
            "perf/data_time_sec": data_time,
            "perf/fwd_bwd_time_sec": fwd_bwd_time,
            "perf/optimizer_time_sec": optimizer_time,
            "perf/step_time_sec": step_time,
            "perf/data_time_ratio": data_time / max(step_time, 1e-9),
            "perf/optimizer_time_ratio": optimizer_time / max(step_time, 1e-9),
        }

    def reset(self) -> None:
        self.data_time_sec = 0.0
        self.fwd_bwd_time_sec = 0.0
        self.optimizer_time_sec = 0.0
        self.step_time_sec = 0.0
        self.count = 0
