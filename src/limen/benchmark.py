"""Benchmark runner: replays segments through the full Limen pipeline.

For each eval window the runner times three stages independently
(preprocessing/feature extraction, inference, decision + safety) and records
full telemetry. In stress runs it applies the deterministic fault-injection
schedule to produce safety evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import BenchmarkConfig
from .inference import LdaBaseline, Prediction, TemporalBaseline
from .preprocessing import EmgFilter, extract_features
from .safety import ActuatorCommand, ConfidenceCoupledSafetyEnvelope, SignalQualityGate
from .signals import Segment, inject_faults
from .telemetry import Telemetry, WindowRecord, now_ns


@dataclass(frozen=True)
class RunResult:
    """One complete benchmark run (one model, one condition)."""

    model_name: str
    condition: str  # "clean" or "fault_injected"
    telemetry: Telemetry
    n_train_windows: int
    n_eval_windows: int


def split_segments(
    segments: list[Segment], cfg: BenchmarkConfig
) -> tuple[list[Segment], list[Segment]]:
    """Deterministic per-class train/eval split by segment count."""
    train: list[Segment] = []
    eval_: list[Segment] = []
    for label in cfg.intent_classes:
        class_segments = [s for s in segments if s.label == label]
        if not class_segments:
            raise ValueError(f"no segments for class {label!r}")
        n_train = max(1, int(round(len(class_segments) * cfg.train_fraction)))
        if n_train >= len(class_segments):
            raise ValueError(
                f"train_fraction {cfg.train_fraction} leaves no eval segments for {label!r}"
            )
        train.extend(class_segments[:n_train])
        eval_.extend(class_segments[n_train:])
    return train, eval_


def training_matrix(
    train_segments: list[Segment], emg_filter: EmgFilter, cfg: BenchmarkConfig
) -> tuple[np.ndarray, list[str]]:
    """Build the feature matrix used to fit the classifiers."""
    pre, sig = cfg.preprocess, cfg.signal
    win = int(pre.window_ms * sig.emg_sample_rate_hz / 1000)
    stride = int(pre.stride_ms * sig.emg_sample_rate_hz / 1000)
    imu_ratio = sig.emg_sample_rate_hz // sig.imu_sample_rate_hz

    rows: list[np.ndarray] = []
    labels: list[str] = []
    for seg in train_segments:
        filtered = emg_filter.apply(seg.emg)
        for start in range(0, filtered.shape[0] - win + 1, stride):
            emg_block = filtered[start : start + win]
            i0 = start // imu_ratio
            imu_block = seg.imu[i0 : max(i0 + 1, (start + win) // imu_ratio)]
            rows.append(extract_features(emg_block, imu_block))
            labels.append(seg.label)
    if not rows:
        raise RuntimeError("training matrix is empty; check window/stride configuration")
    return np.vstack(rows), labels


def replay(
    cfg: BenchmarkConfig,
    model: LdaBaseline | TemporalBaseline,
    model_name: str,
    eval_segments: list[Segment],
    emg_filter: EmgFilter,
    condition: str,
    n_train_windows: int,
) -> RunResult:
    """Replay eval segments through inference + safety envelope with telemetry."""
    if condition not in ("clean", "fault_injected"):
        raise ValueError(f"unknown condition {condition!r}")

    pre, sig = cfg.preprocess, cfg.signal
    sqg = SignalQualityGate()
    envelope = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    telemetry = Telemetry()
    n_eval_windows = 0

    win = int(pre.window_ms * sig.emg_sample_rate_hz / 1000)
    stride = int(pre.stride_ms * sig.emg_sample_rate_hz / 1000)
    imu_ratio = sig.emg_sample_rate_hz // sig.imu_sample_rate_hz

    for seg in eval_segments:
        replay_seg = inject_faults(seg, cfg.fault, sig) if condition == "fault_injected" else seg
        envelope.reset()
        if isinstance(model, TemporalBaseline):
            model.reset_stream()
        filtered = emg_filter.apply(replay_seg.emg)

        for start in range(0, filtered.shape[0] - win + 1, stride):
            emg_block = filtered[start : start + win]
            raw_emg_block = replay_seg.emg[start : start + win]
            i0 = start // imu_ratio
            imu_block = replay_seg.imu[i0 : max(i0 + 1, (start + win) // imu_ratio)]

            t0 = now_ns()
            features = extract_features(emg_block, imu_block)
            t1 = now_ns()
            prediction: Prediction = model.predict(features)
            t2 = now_ns()
            quality = sqg.assess(raw_emg_block)
            command: ActuatorCommand = envelope.evaluate(prediction, quality)
            t3 = now_ns()

            telemetry.add(
                WindowRecord(
                    true_label=seg.label,
                    predicted_intent=command.intent,
                    confidence=prediction.confidence,
                    reliability=quality.reliability,
                    effective_confidence=command.effective_confidence,
                    envelope_state=command.envelope_state.value,
                    vetoed=command.vetoed_by_supervisor,
                    commanded_torque_nm=command.torque_nm,
                    preprocess_ns=t1 - t0,
                    inference_ns=t2 - t1,
                    decision_ns=t3 - t2,
                )
            )
            n_eval_windows += 1

    return RunResult(
        model_name=model_name,
        condition=condition,
        telemetry=telemetry,
        n_train_windows=n_train_windows,
        n_eval_windows=n_eval_windows,
    )
