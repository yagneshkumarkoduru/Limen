"""Segment replay driver for the Limen pipeline.

Each decision window is independently timed across three stages -
feature extraction, inference, and the trust-gated decision stage - and
captured as telemetry. Stress passes apply a deterministic
fault-injection schedule to generate safety evidence.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Iterator

import numpy as np

from .config import BenchmarkConfig as BenchmarkConfig
from .inference import LdaBaseline, Prediction, TemporalBaseline
from .preprocessing import EmgFilter, extract_features
from .safety import ActuatorCommand, ConfidenceCoupledSafetyEnvelope, SignalQualityGate
from .signals import Segment, inject_faults
from .telemetry import Telemetry, WindowRecord, now_ns as now_ns


@dataclasses.dataclass(frozen=True)
class RunResult:
    """One complete replay outcome (one model, one condition)."""

    model_name: str; condition: str  # "clean" or "fault_injected"
    telemetry: Telemetry; n_train_windows: int; n_eval_windows: int


def split_segments(
    segments: list[Segment],
    cfg: BenchmarkConfig,
) -> tuple[list[Segment], list[Segment]]:
    """Deterministic per-class train/eval partition by segment count."""
    kept_for_fit: list[Segment] = []
    kept_for_score: list[Segment] = []
    for cls in cfg.intent_classes:
        members = [entry for entry in segments if entry.label == cls]
        if len(members) == 0:
            raise ValueError(f"no segments for class {cls!r}")
        cut = max(1, int(round(len(members) * cfg.train_fraction)))
        if cut >= len(members):
            raise ValueError(
                f"train_fraction {cfg.train_fraction} "
                f"leaves no eval segments for {cls!r}"
            )
        kept_for_fit += members[:cut]
        kept_for_score += members[cut:]
    return kept_for_fit, kept_for_score


class _Framing:
    """Window geometry derived from the configured rates."""

    __slots__ = ("width", "hop", "downsample")

    def __init__(self, cfg: BenchmarkConfig) -> None:
        emg_rate = cfg.signal.emg_sample_rate_hz
        self.width = int(cfg.preprocess.window_ms * emg_rate / 1000)
        self.hop = int(cfg.preprocess.stride_ms * emg_rate / 1000)
        self.downsample = emg_rate // cfg.signal.imu_sample_rate_hz

    def frames(
        self,
        filtered: np.ndarray,
        raw: np.ndarray,
        imu: np.ndarray,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (filtered block, raw block, imu block) per decision window."""
        last_start = filtered.shape[0] - self.width + 1
        for origin in range(0, last_start, self.hop):
            imu_lo = origin // self.downsample
            imu_hi = max(imu_lo + 1, (origin + self.width) // self.downsample)
            yield (
                filtered[origin : origin + self.width],
                raw[origin : origin + self.width],
                imu[imu_lo:imu_hi],
            )


def training_matrix(
    train_segments: list[Segment],
    emg_filter: EmgFilter,
    cfg: BenchmarkConfig,
    feature_fn: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ] = extract_features,
) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix used to fit the classifiers."""
    framing = _Framing(cfg)
    feature_rows: list[np.ndarray] = []
    row_labels: list[str] = []
    for segment in train_segments:
        filtered_block = emg_filter.apply(segment.emg)
        for emg_frame, _raw_frame, imu_frame in framing.frames(
            filtered_block, segment.emg, segment.imu
        ):
            feature_rows.append(feature_fn(emg_frame, imu_frame))
            row_labels.append(segment.label)
    if len(feature_rows) == 0:
        raise RuntimeError(
            "training matrix is empty; "
            "check window/stride configuration"
        )
    return np.vstack(feature_rows), row_labels


class _ReplayState:
    """Per-pass mutable machinery: gate, envelope, telemetry, and timing."""

    __slots__ = (
        "_quality_gate",
        "_envelope",
        "_log",
        "_window_total",
    )

    def __init__(
        self,
        cfg: BenchmarkConfig,
        request_table: dict[str, tuple[float, float]] | None,
    ) -> None:
        self._quality_gate = SignalQualityGate()
        self._envelope = ConfidenceCoupledSafetyEnvelope(
            cfg.safety, command_map=request_table
        )
        self._log = Telemetry()
        self._window_total = 0

    def arm_for_segment(
        self, classifier: LdaBaseline | TemporalBaseline
    ) -> None:
        self._envelope.reset(self._envelope.reset_key)
        if isinstance(classifier, TemporalBaseline):
            classifier.reset_stream()

    def record_window(
        self,
        classifier: LdaBaseline | TemporalBaseline,
        emg_frame: np.ndarray,
        raw_frame: np.ndarray,
        imu_frame: np.ndarray,
        extract: Callable,
        source: Segment,
    ) -> None:
        mark_a = now_ns()
        vector = extract(emg_frame, imu_frame)
        mark_b = now_ns()
        outcome: Prediction = classifier.predict(vector)
        mark_c = now_ns()
        verdict = self._quality_gate.assess(raw_frame)
        issued: ActuatorCommand = self._envelope.evaluate(outcome, verdict)
        mark_d = now_ns()
        self._log.add(
            WindowRecord(
                true_label=source.label,
                predicted_intent=issued.intent,
                confidence=outcome.confidence,
                reliability=verdict.reliability,
                effective_confidence=issued.effective_confidence,
                envelope_state=issued.envelope_state.value,
                vetoed=issued.vetoed_by_supervisor,
                commanded_torque_nm=issued.torque_nm,
                preprocess_ns=mark_b - mark_a,
                inference_ns=mark_c - mark_b,
                decision_ns=mark_d - mark_c,
                subject=source.subject,
            )
        )
        self._window_total += 1

    def finish(
        self,
        name: str,
        mode: str,
        train_windows: int,
    ) -> RunResult:
        return RunResult(
            model_name=name,
            condition=mode,
            telemetry=self._log,
            n_train_windows=train_windows,
            n_eval_windows=self._window_total,
        )


def replay(
    cfg: BenchmarkConfig,
    model: LdaBaseline | TemporalBaseline,
    model_name: str,
    eval_segments: list[Segment],
    emg_filter: EmgFilter,
    condition: str,
    n_train_windows: int,
    feature_fn: Callable[
        [np.ndarray, np.ndarray], np.ndarray
    ] = extract_features,
    command_map: dict[str, tuple[float, float]] | None = None,
    nominal_amplitude: float | None = None,
) -> RunResult:
    """Replay eval segments through inference plus the trust-gated envelope."""
    if condition not in ("clean", "fault_injected"):
        raise ValueError(f"unknown condition {condition!r}")

    framing = _Framing(cfg)
    state = _ReplayState(cfg, command_map)

    for segment in eval_segments:
        if condition == "fault_injected":
            played = inject_faults(
                segment, cfg.fault, cfg.signal,
                nominal_amplitude=nominal_amplitude,
            )
        else:
            played = segment
        state.arm_for_segment(model)
        filtered_block = emg_filter.apply(played.emg)
        for emg_frame, raw_frame, imu_frame in framing.frames(
            filtered_block, played.emg, played.imu
        ):
            state.record_window(model, emg_frame, raw_frame, imu_frame, feature_fn, segment)

    return state.finish(model_name, condition, n_train_windows)
