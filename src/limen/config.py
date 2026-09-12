"""Configuration for the Limen benchmark.

All benchmark parameters live here so that a run is fully reproducible
from committed code. Deterministic seeds are mandatory; wall-clock or
unseeded randomness is prohibited anywhere in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SignalConfig:
    """Parameters of the synthetic or replayed signal stream."""

    emg_sample_rate_hz: int = 1000
    imu_sample_rate_hz: int = 100
    n_emg_channels: int = 8
    n_imu_channels: int = 6  # 3-axis accelerometer + 3-axis gyroscope
    segment_seconds: float = 6.0  # duration of one labeled movement segment
    rest_seconds: float = 2.0  # rest gap inserted between segments
    segments_per_class: int = 12  # repetitions per intent class


@dataclass(frozen=True)
class PreprocessConfig:
    """Preprocessing and feature-extraction parameters."""

    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    notch_hz: float = 50.0  # line-noise notch (India mains frequency)
    filter_order: int = 4
    window_ms: int = 200
    stride_ms: int = 20  # decision rate: one decision every 20 ms (50 Hz)


@dataclass(frozen=True)
class SafetyConfig:
    """Confidence-Coupled Safety Envelope parameters.

    T_high / T_low partition effective confidence into the three envelope
    tiers. Damped scaling bounds actuator authority in the DEGRADED tier.
    Hard bounds are enforced by the independent supervisor regardless of
    confidence.
    """

    t_high: float = 0.75
    t_low: float = 0.45
    degraded_velocity_scale: float = 0.5
    degraded_force_scale: float = 0.4
    hard_max_torque_nm: float = 40.0  # absolute actuator bound, never relaxable
    hard_max_velocity_rad_s: float = 6.0
    persistence_windows: int = 3  # consecutive low-C_eff windows before SAFE_HALT


@dataclass(frozen=True)
class FaultConfig:
    """Fault-injection schedule used by the benchmark stress runs."""

    electrode_dropout_channel: int = 1
    electrode_dropout_at_s: float = 3.0  # within the stress segment
    motion_artifact_at_s: float = 4.5
    motion_artifact_amplitude: float = 8.0  # multiple of nominal EMG amplitude


@dataclass(frozen=True)
class BenchmarkConfig:
    """Top-level benchmark configuration."""

    seed: int = 20260722
    train_fraction: float = 0.6
    intent_classes: tuple[str, ...] = ("rest", "knee_flexion", "knee_extension")
    signal: SignalConfig = field(default_factory=SignalConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
