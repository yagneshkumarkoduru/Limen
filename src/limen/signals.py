"""Deterministic synthetic EMG/IMU generator.

Produces labeled, physiologically-plausible synthetic biosignal segments for
the replay benchmark, plus a fault-injection schedule for safety-evidence
runs. All randomness flows through a seeded ``numpy.random.Generator`` so a
benchmark run is bit-for-bit reproducible.

Signal model (documented, deliberately simple):
- EMG: band-limited Gaussian noise shaped by a per-class amplitude envelope
  and per-channel activation pattern.
- IMU: slow sinusoidal joint-angle proxy with class-dependent range of motion,
  sampled at a lower rate than EMG (as in real hardware).

Faults:
- ``electrode_dropout``: one EMG channel flatlines (contact loss).
- ``motion_artifact``: broadband amplitude burst across EMG channels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import BenchmarkConfig, FaultConfig, SignalConfig


@dataclass(frozen=True)
class Segment:
    """One labeled movement segment with synchronized EMG/IMU streams."""

    label: str
    emg: NDArray[np.float64]  # shape (n_emg_samples, n_emg_channels)
    imu: NDArray[np.float64]  # shape (n_imu_samples, n_imu_channels)
    duration_s: float

    def validate(self, cfg: SignalConfig) -> None:
        expected_emg = int(round(self.duration_s * cfg.emg_sample_rate_hz))
        expected_imu = int(round(self.duration_s * cfg.imu_sample_rate_hz))
        if self.emg.shape != (expected_emg, cfg.n_emg_channels):
            raise ValueError(
                f"EMG shape {self.emg.shape} != expected "
                f"({expected_emg}, {cfg.n_emg_channels}) for label {self.label!r}"
            )
        if self.imu.shape != (expected_imu, cfg.n_imu_channels):
            raise ValueError(
                f"IMU shape {self.imu.shape} != expected "
                f"({expected_imu}, {cfg.n_imu_channels}) for label {self.label!r}"
            )


# Per-class EMG channel activation patterns (co-contraction signatures).
_ACTIVATION: dict[str, tuple[float, ...]] = {
    "rest": (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15),
    "knee_flexion": (1.0, 0.9, 0.25, 0.20, 0.8, 0.7, 0.2, 0.15),
    "knee_extension": (0.25, 0.20, 1.0, 0.90, 0.2, 0.15, 0.8, 0.7),
}

# Per-class IMU (joint-angle proxy) range of motion in radians.
_ROM_RAD: dict[str, float] = {
    "rest": 0.02,
    "knee_flexion": 0.9,
    "knee_extension": 0.7,
}

_BASE_EMG_AMPLITUDE_UV = 120.0  # microvolts, typical surface-EMG order of magnitude


def generate_segment(label: str, cfg: SignalConfig, rng: np.random.Generator) -> Segment:
    """Generate one labeled segment. Raises on unknown label - no silent defaults."""
    if label not in _ACTIVATION:
        raise ValueError(f"Unknown intent class {label!r}; known: {sorted(_ACTIVATION)}")
    if cfg.n_emg_channels > len(_ACTIVATION[label]):
        raise ValueError(
            f"synthetic generator supports at most {len(_ACTIVATION[label])} EMG channels"
        )
    if cfg.n_imu_channels < 3:
        raise ValueError("synthetic generator requires at least 3 IMU channels")

    n_emg = int(round(cfg.segment_seconds * cfg.emg_sample_rate_hz))
    n_imu = int(round(cfg.segment_seconds * cfg.imu_sample_rate_hz))

    activation = np.asarray(_ACTIVATION[label], dtype=np.float64)[: cfg.n_emg_channels]
    t_emg = np.arange(n_emg) / cfg.emg_sample_rate_hz

    emg = np.empty((n_emg, cfg.n_emg_channels), dtype=np.float64)
    ramp = np.clip(t_emg / (0.3 * cfg.segment_seconds), 0.0, 1.0)
    release = np.clip((cfg.segment_seconds - t_emg) / (0.2 * cfg.segment_seconds), 0.0, 1.0)
    envelope = np.minimum(ramp, release)
    for ch in range(cfg.n_emg_channels):
        white = rng.standard_normal(n_emg)
        kernel = np.ones(25) / 25.0
        smooth = np.convolve(white, kernel, mode="same")
        smooth /= np.std(smooth) + 1e-12
        emg[:, ch] = (
            _BASE_EMG_AMPLITUDE_UV
            * activation[ch]
            * envelope
            * smooth
            + rng.standard_normal(n_emg) * 8.0
        )

    t_imu = np.arange(n_imu) / cfg.imu_sample_rate_hz
    phase = np.pi * t_imu / cfg.segment_seconds
    angle = _ROM_RAD[label] * np.sin(phase)
    imu = np.zeros((n_imu, cfg.n_imu_channels), dtype=np.float64)
    imu[:, 0] = angle + rng.standard_normal(n_imu) * 0.005
    imu[:, 1] = 0.1 * np.sin(phase) + rng.standard_normal(n_imu) * 0.005
    imu[:, 2] = rng.standard_normal(n_imu) * 0.01
    if cfg.n_imu_channels >= 4:
        imu[:, 3] = np.gradient(angle, t_imu) + rng.standard_normal(n_imu) * 0.01
    if cfg.n_imu_channels >= 5:
        imu[:, 4] = rng.standard_normal(n_imu) * 0.01
    if cfg.n_imu_channels >= 6:
        imu[:, 5] = rng.standard_normal(n_imu) * 0.01

    segment = Segment(label=label, emg=emg, imu=imu, duration_s=cfg.segment_seconds)
    segment.validate(cfg)
    return segment


def generate_rest_segment(cfg: SignalConfig, rng: np.random.Generator) -> Segment:
    """Generate a rest-gap segment of ``cfg.rest_seconds`` duration."""
    short_cfg = SignalConfig(
        emg_sample_rate_hz=cfg.emg_sample_rate_hz,
        imu_sample_rate_hz=cfg.imu_sample_rate_hz,
        n_emg_channels=cfg.n_emg_channels,
        n_imu_channels=cfg.n_imu_channels,
        segment_seconds=cfg.rest_seconds,
        rest_seconds=cfg.rest_seconds,
        segments_per_class=cfg.segments_per_class,
    )
    seg = generate_segment("rest", short_cfg, rng)
    return Segment(label="rest", emg=seg.emg, imu=seg.imu, duration_s=cfg.rest_seconds)


def inject_faults(segment: Segment, fault: FaultConfig, cfg: SignalConfig) -> Segment:
    """Return a copy of ``segment`` with the scheduled faults injected.

    Deterministic: faults depend only on the schedule in ``fault``, never on RNG.
    """
    emg = segment.emg.copy()
    emg_rate = cfg.emg_sample_rate_hz

    drop_idx = int(fault.electrode_dropout_at_s * emg_rate)
    if not 0 <= fault.electrode_dropout_channel < emg.shape[1]:
        raise ValueError(f"dropout channel {fault.electrode_dropout_channel} out of range")
    if drop_idx < emg.shape[0]:
        emg[drop_idx:, fault.electrode_dropout_channel] = 0.0

    art_idx = int(fault.motion_artifact_at_s * emg_rate)
    burst_len = int(0.25 * emg_rate)
    if art_idx + burst_len <= emg.shape[0]:
        burst_t = np.arange(burst_len) / emg_rate
        burst = (
            _BASE_EMG_AMPLITUDE_UV
            * fault.motion_artifact_amplitude
            * np.sin(2 * np.pi * 7.0 * burst_t)
        )
        emg[art_idx : art_idx + burst_len, :] += burst[:, None]

    return Segment(
        label=segment.label, emg=emg, imu=segment.imu.copy(), duration_s=segment.duration_s
    )


def generate_dataset(cfg: BenchmarkConfig) -> list[Segment]:
    """Generate the full labeled benchmark dataset (train + eval share one stream).

    Segment order is deterministic: movement-class repetitions are generated
    first, followed by dedicated rest segments.
    """
    rng = np.random.default_rng(cfg.seed)
    segments: list[Segment] = []
    for label in cfg.intent_classes:
        if label == "rest":
            continue
        for _ in range(cfg.signal.segments_per_class):
            segments.append(generate_segment(label, cfg.signal, rng))
    for _ in range(cfg.signal.segments_per_class):
        segments.append(generate_segment("rest", cfg.signal, rng))
    return segments
