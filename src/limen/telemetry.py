"""Window-level telemetry recording and aggregate metric computation.

Collects one :class:`WindowRecord` per decision window and derives
latency percentile summaries, classification accuracy, false activation
rate, rejection rate, and envelope event tallies. Timing relies on
``time.perf_counter_ns`` purely for measurement; it never enters the
deterministic data path.
"""

from __future__ import annotations

import math
from time import perf_counter_ns
from typing import Optional

import numpy

from .safety import EnvelopeState


def now_ns() -> int:
    """Nanosecond-resolution wall clock for latency measurement only."""
    return perf_counter_ns()


class WindowRecord:
    """Immutable telemetry snapshot for a single decision window."""

    __slots__ = (
        "true_label",
        "predicted_intent",
        "confidence",
        "reliability",
        "effective_confidence",
        "envelope_state",
        "vetoed",
        "commanded_torque_nm",
        "preprocess_ns",
        "inference_ns",
        "decision_ns",
        "subject",
    )

    def __init__(
        self,
        *,
        true_label: str,
        predicted_intent: str,
        confidence: float,
        reliability: float,
        effective_confidence: float,
        envelope_state: str,
        vetoed: bool,
        commanded_torque_nm: float,
        preprocess_ns: int,
        inference_ns: int,
        decision_ns: int,
        subject: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "true_label", true_label)
        object.__setattr__(self, "predicted_intent", predicted_intent)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reliability", reliability)
        object.__setattr__(self, "effective_confidence", effective_confidence)
        object.__setattr__(self, "envelope_state", envelope_state)
        object.__setattr__(self, "vetoed", vetoed)
        object.__setattr__(self, "commanded_torque_nm", commanded_torque_nm)
        object.__setattr__(self, "preprocess_ns", preprocess_ns)
        object.__setattr__(self, "inference_ns", inference_ns)
        object.__setattr__(self, "decision_ns", decision_ns)
        object.__setattr__(self, "subject", subject)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"cannot set {name!r}: WindowRecord is immutable")

    @property
    def end_to_end_ns(self) -> int:
        val: int = self.preprocess_ns
        val += self.inference_ns
        val += self.decision_ns
        return val

    def __repr__(self) -> str:
        return (
            f"WindowRecord(true_label={self.true_label!r}, "
            f"predicted_intent={self.predicted_intent!r}, "
            f"confidence={self.confidence}, "
            f"envelope_state={self.envelope_state!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, WindowRecord):
            return NotImplemented
        return all(
            getattr(self, f) == getattr(other, f)
            for f in WindowRecord.__slots__
        )

    def __hash__(self) -> int:
        return hash(tuple(getattr(self, f) for f in WindowRecord.__slots__))


def _latency_row(samples_ns: list[int]) -> dict[str, float]:
    """Compute one latency percentile row from nanosecond samples."""
    if len(samples_ns) == 0:
        raise ValueError("empty latency sample column")
    ms = numpy.array(samples_ns, dtype=numpy.float64) / 1.0e6
    ms.sort()
    count = ms.shape[0]

    def pick(q: float) -> float:
        idx = max(0, math.floor(q * (count - 1)))
        return round(float(ms[idx]), 4)

    return {
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "p99_ms": pick(0.99),
        "max_ms": round(float(ms[count - 1]), 4),
        "mean_ms": round(float(numpy.sum(ms) / count), 4),
    }


def _tally_accuracy(
    pool: list[WindowRecord], skip_halt: bool
) -> tuple[int, int]:
    """Single-pass (hits, counted) over the record pool."""
    halt = EnvelopeState.SAFE_HALT.value
    hits = counted = 0
    pos = 0
    end = len(pool)
    while pos < end:
        item = pool[pos]
        pos += 1
        if skip_halt and item.envelope_state == halt:
            continue
        counted += 1
        if item.predicted_intent == item.true_label:
            hits += 1
    return hits, counted


class Telemetry:
    """Accumulates window records; computes aggregate metrics."""

    __slots__ = ("records",)

    def __init__(self) -> None:
        self.records: list[WindowRecord] = []

    def add(self, record: WindowRecord) -> None:
        self.records += [record]

    def latency_summary(self) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[int]] = {
            "preprocessing": [],
            "inference": [],
            "decision_loop": [],
            "end_to_end": [],
        }
        for item in self.records:
            buckets["preprocessing"].append(item.preprocess_ns)
            buckets["inference"].append(item.inference_ns)
            buckets["decision_loop"].append(item.decision_ns)
            buckets["end_to_end"].append(item.end_to_end_ns)
        result: dict[str, dict[str, float]] = {}
        for stage, col in buckets.items():
            result[stage] = _latency_row(col)
        return result

    def accuracy(self) -> float:
        num, den = _tally_accuracy(self.records, skip_halt=False)
        if den == 0:
            raise ValueError("no window records in pool")
        return round(num / den, 6)

    def confusion_matrix(
        self, classes: tuple[str, ...]
    ) -> list[list[int]]:
        dim = len(classes)
        slot: dict[str, int] = {}
        for pos, name in enumerate(classes):
            slot[name] = pos
        grid = [[0] * dim for _ in range(dim)]
        for item in self.records:
            ri = slot.get(item.true_label)
            ci = slot.get(item.predicted_intent)
            if ri is not None and ci is not None:
                grid[ri][ci] += 1
        return grid

    def false_activation_rate(self) -> float | None:
        """Fraction of true-rest windows that produced a non-rest command.

        Returns ``None`` when the evaluated stream contains no rest
        windows; the metric is undefined in that case, not zero.
        """
        halt = EnvelopeState.SAFE_HALT.value
        rest_n = fa_n = 0
        for item in self.records:
            if item.true_label != "rest":
                continue
            rest_n += 1
            if item.predicted_intent != "rest" and item.envelope_state != halt:
                fa_n += 1
        if rest_n == 0:
            return None
        return round(fa_n / rest_n, 6)

    def rejection_rate(self) -> float:
        """Share of windows the envelope constrained."""
        pool = self.records
        if len(pool) == 0:
            raise ValueError("no window records in pool")
        nom = EnvelopeState.NOMINAL.value
        off_count = 0
        for item in pool:
            if item.envelope_state != nom:
                off_count += 1
        return round(off_count / len(pool), 6)

    def accuracy_on_accepted(self) -> float | None:
        """Accuracy over windows the envelope did not SAFE_HALT."""
        num, den = _tally_accuracy(self.records, skip_halt=True)
        if den == 0:
            return None
        return round(num / den, 6)

    def envelope_event_counts(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for st in EnvelopeState:
            tally[st.value] = 0
        for item in self.records:
            current = tally[item.envelope_state]
            tally[item.envelope_state] = current + 1
        return tally
