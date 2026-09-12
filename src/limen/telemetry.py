"""Telemetry: per-window records and aggregate benchmark metrics.

Implements the required metrics table: end-to-end and per-stage
latency distributions, accuracy, false activation rate, rejection rate, and
repeatability support. Latency is measured with ``time.perf_counter_ns``
for measurement only; it is never part of the deterministic data path.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .safety import ActuatorCommand, EnvelopeState


@dataclass(frozen=True)
class WindowRecord:
    """One decision window full telemetry record."""

    true_label: str
    predicted_intent: str
    confidence: float
    reliability: float
    effective_confidence: float
    envelope_state: str
    vetoed: bool
    commanded_torque_nm: float
    preprocess_ns: int
    inference_ns: int
    decision_ns: int

    @property
    def end_to_end_ns(self) -> int:
        return self.preprocess_ns + self.inference_ns + self.decision_ns


@dataclass
class Telemetry:
    """Accumulates window records; computes aggregate metrics."""

    records: list[WindowRecord] = field(default_factory=list)

    def add(self, record: WindowRecord) -> None:
        self.records.append(record)

    def _latency_stats_ms(self, values_ns: list[int]) -> dict[str, float]:
        if not values_ns:
            raise ValueError("no latency samples recorded")
        arr = np.asarray(values_ns, dtype=np.float64) / 1e6
        return {
            "p50_ms": round(float(np.percentile(arr, 50)), 4),
            "p95_ms": round(float(np.percentile(arr, 95)), 4),
            "p99_ms": round(float(np.percentile(arr, 99)), 4),
            "max_ms": round(float(arr.max()), 4),
            "mean_ms": round(float(arr.mean()), 4),
        }

    def latency_summary(self) -> dict[str, dict[str, float]]:
        return {
            "preprocessing": self._latency_stats_ms([r.preprocess_ns for r in self.records]),
            "inference": self._latency_stats_ms([r.inference_ns for r in self.records]),
            "decision_loop": self._latency_stats_ms([r.decision_ns for r in self.records]),
            "end_to_end": self._latency_stats_ms([r.end_to_end_ns for r in self.records]),
        }

    def accuracy(self) -> float:
        if not self.records:
            raise ValueError("no records")
        correct = sum(1 for r in self.records if r.predicted_intent == r.true_label)
        return round(correct / len(self.records), 6)

    def confusion_matrix(self, classes: tuple[str, ...]) -> list[list[int]]:
        idx = {c: i for i, c in enumerate(classes)}
        matrix = [[0] * len(classes) for _ in classes]
        for r in self.records:
            if r.true_label in idx and r.predicted_intent in idx:
                matrix[idx[r.true_label]][idx[r.predicted_intent]] += 1
        return matrix

    def false_activation_rate(self) -> float:
        """Share of true-rest windows where a non-rest command was issued."""
        rest = [r for r in self.records if r.true_label == "rest"]
        if not rest:
            raise ValueError("no rest windows recorded")
        false_acts = sum(
            1
            for r in rest
            if r.predicted_intent != "rest"
            and r.envelope_state != EnvelopeState.SAFE_HALT.value
        )
        return round(false_acts / len(rest), 6)

    def rejection_rate(self) -> float:
        """Share of windows the envelope constrained (DEGRADED or SAFE_HALT)."""
        if not self.records:
            raise ValueError("no records")
        constrained = sum(
            1 for r in self.records if r.envelope_state != EnvelopeState.NOMINAL.value
        )
        return round(constrained / len(self.records), 6)

    def envelope_event_counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in EnvelopeState}
        for r in self.records:
            counts[r.envelope_state] += 1
        return counts


def now_ns() -> int:
    """Single latency-clock entry point for the benchmark runner."""
    return time.perf_counter_ns()
