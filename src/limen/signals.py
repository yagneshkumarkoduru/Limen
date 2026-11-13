"""Synthetic biosignal generator for reproducible benchmark replay.

Produces labeled, physiologically plausible EMG and IMU segment
streams, plus a deterministic fault-injection schedule for safety
evidence runs. Every source of randomness flows through a seeded
``numpy.random.Generator`` so that any benchmark run is bit-for-bit
reproducible.

Signal model (documented, deliberately simple):

- EMG: band-limited Gaussian noise shaped by a per-class amplitude
  envelope and a per-channel co-contraction pattern.
- IMU: slow sinusoidal joint-angle proxy with class-dependent
  range of motion, sampled at a lower rate than EMG (as in real
  hardware).

Faults:

- ``electrode_dropout``: one EMG channel flatlines (contact loss).
- ``motion_artifact``: broadband amplitude burst across EMG channels.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from numpy.typing import NDArray

from .config import BenchmarkConfig, FaultConfig, SignalConfig


@dataclasses.dataclass(frozen=True)
class Segment:
    """One labeled movement segment with synchronized EMG/IMU streams.

    ``subject`` and ``trial`` are optional provenance fields used
    by the real-data protocols (within-subject and leave-one-
    subject-out splits). Synthetic segments leave them unset.
    """

    label: str
    emg: NDArray[np.float64]
    imu: NDArray[np.float64]
    duration_s: float
    subject: str | None = None
    trial: int | None = None

    def validate(self, cfg: SignalConfig) -> None:
        n_e = int(round(self.duration_s * cfg.emg_sample_rate_hz))
        n_i = int(round(self.duration_s * cfg.imu_sample_rate_hz))
        if self.emg.shape != (n_e, cfg.n_emg_channels):
            raise ValueError(
                f"EMG shape {self.emg.shape} != expected "
                f"({n_e}, {cfg.n_emg_channels}) "
                f"for label {self.label!r}"
            )
        if self.imu.shape != (n_i, cfg.n_imu_channels):
            raise ValueError(
                f"IMU shape {self.imu.shape} != expected "
                f"({n_i}, {cfg.n_imu_channels}) "
                f"for label {self.label!r}"
            )


# Per-class physiology: co-contraction signatures and joint ROM.
_COCONTRACT: dict[str, tuple[float, ...]] = {
    "rest": (0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15),
    "knee_flexion": (1.0, 0.9, 0.25, 0.20, 0.8, 0.7, 0.2, 0.15),
    "knee_extension": (0.25, 0.20, 1.0, 0.90, 0.2, 0.15, 0.8, 0.7),
}

_RANGE_OF_MOTION: dict[str, float] = {
    "rest": 0.02,
    "knee_flexion": 0.9,
    "knee_extension": 0.7,
}

_NOMINAL_UV: float = 120.0


def _guard(label: str, cfg: SignalConfig) -> None:
    if label not in _COCONTRACT:
        raise ValueError(
            f"Unknown intent class {label!r}; "
            f"known: {sorted(_COCONTRACT)}"
        )
    cap = len(_COCONTRACT[label])
    if cfg.n_emg_channels > cap:
        raise ValueError(
            f"synthetic generator supports at most {cap} EMG channels"
        )
    if cfg.n_imu_channels < 3:
        raise ValueError(
            "synthetic generator requires at least 3 IMU channels"
        )


def _emg_gen(
    label: str,
    n: int,
    cfg: SignalConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Per-channel modulated band-limited noise."""
    nch = cfg.n_emg_channels
    fs = cfg.emg_sample_rate_hz
    dur = cfg.segment_seconds
    act = np.array(
        _COCONTRACT[label][:nch], dtype=np.float64
    )
    t = np.arange(n) / fs
    env_rise = np.clip(t / (0.3 * dur), 0.0, 1.0)
    env_fall = np.clip((dur - t) / (0.2 * dur), 0.0, 1.0)
    env = np.minimum(env_rise, env_fall)
    taps = np.full(25, 1.0 / 25.0)
    out = np.zeros((n, nch), dtype=np.float64)
    ch = 0
    while ch < nch:
        white = rng.standard_normal(n)
        banded = np.convolve(white, taps, mode="same")
        banded /= np.std(banded) + 1e-12
        out[:, ch] = (
            _NOMINAL_UV * act[ch] * env * banded
        )
        out[:, ch] += rng.standard_normal(n) * 8.0
        ch += 1
    return out


def _imu_gen(
    label: str,
    n: int,
    cfg: SignalConfig,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Sinusoidal joint-angle proxy."""
    fs = cfg.imu_sample_rate_hz
    dur = cfg.segment_seconds
    rom = _RANGE_OF_MOTION[label]
    t = np.arange(n) / fs
    sin_half = np.sin(np.pi * t / dur)
    ang = rom * sin_half
    out = np.zeros(
        (n, cfg.n_imu_channels), dtype=np.float64
    )
    nz = lambda sc: rng.standard_normal(n) * sc
    out[:, 0] = ang + nz(0.005)
    out[:, 1] = 0.1 * sin_half + nz(0.005)
    out[:, 2] = nz(0.01)
    extra = cfg.n_imu_channels - 3
    if extra >= 1:
        out[:, 3] = np.gradient(ang, t) + nz(0.01)
    if extra >= 2:
        out[:, 4] = nz(0.01)
    if extra >= 3:
        out[:, 5] = nz(0.01)
    return out


def generate_segment(
    label: str,
    cfg: SignalConfig,
    rng: np.random.Generator,
) -> Segment:
    """Generate one labeled segment. Raises on unknown label."""
    _guard(label, cfg)
    n_e = int(
        round(cfg.segment_seconds * cfg.emg_sample_rate_hz)
    )
    n_i = int(
        round(cfg.segment_seconds * cfg.imu_sample_rate_hz)
    )
    seg = Segment(
        label=label,
        emg=_emg_gen(label, n_e, cfg, rng),
        imu=_imu_gen(label, n_i, cfg, rng),
        duration_s=cfg.segment_seconds,
    )
    seg.validate(cfg)
    return seg


def generate_rest_segment(
    cfg: SignalConfig,
    rng: np.random.Generator,
) -> Segment:
    """Generate a rest-gap segment."""
    short = SignalConfig(
        emg_sample_rate_hz=cfg.emg_sample_rate_hz,
        imu_sample_rate_hz=cfg.imu_sample_rate_hz,
        n_emg_channels=cfg.n_emg_channels,
        n_imu_channels=cfg.n_imu_channels,
        segment_seconds=cfg.rest_seconds,
        rest_seconds=cfg.rest_seconds,
        segments_per_class=cfg.segments_per_class,
    )
    inner = generate_segment("rest", short, rng)
    return Segment(
        label="rest",
        emg=inner.emg,
        imu=inner.imu,
        duration_s=cfg.rest_seconds,
    )


def generate_dataset(
    cfg: BenchmarkConfig,
) -> list[Segment]:
    """Generate the full labeled benchmark dataset.

    Movement-class repetitions first, then rest segments.
    """
    gen = np.random.default_rng(cfg.seed)
    result: list[Segment] = []
    for cls in cfg.intent_classes:
        if cls == "rest":
            continue
        for _rep in range(cfg.signal.segments_per_class):
            result.append(
                generate_segment(cls, cfg.signal, gen)
            )
    for _rep in range(cfg.signal.segments_per_class):
        result.append(
            generate_segment("rest", cfg.signal, gen)
        )
    return result


def inject_faults(
    segment: Segment,
    fault: FaultConfig,
    cfg: SignalConfig,
    nominal_amplitude: float | None = None,
) -> Segment:
    """Return a copy of ``segment`` with scheduled faults.

    Deterministic: faults depend only on the schedule in
    ``fault``, never on RNG.

    Args:
        nominal_amplitude: the signal's nominal EMG amplitude
            (same unit as the samples). The motion-artifact
            burst is ``fault.motion_artifact_amplitude`` times
            this value. When ``None`` the synthetic generator's
            base amplitude (120 uV) is used, which matches
            synthetic segments; real recordings must pass their
            own measured nominal amplitude or the burst is
            invisible against the recording's actual scale.
    """
    fs = float(cfg.emg_sample_rate_hz)
    emg_cp = segment.emg.copy()
    ref = (
        _NOMINAL_UV
        if nominal_amplitude is None
        else float(nominal_amplitude)
    )
    if ref <= 0.0:
        raise ValueError(
            f"nominal_amplitude must be positive, got {ref}"
        )
    dch = fault.electrode_dropout_channel
    if not 0 <= dch < emg_cp.shape[1]:
        raise ValueError(
            f"dropout channel {dch} out of range"
        )

    # Electrode dropout: flatline from onset to end.
    drop_at = int(fault.electrode_dropout_at_s * fs)
    if drop_at < emg_cp.shape[0]:
        emg_cp[drop_at:, dch] = 0.0

    # Motion artifact: 250 ms sinusoidal burst.
    art_at = int(fault.motion_artifact_at_s * fs)
    burst_n = int(0.25 * fs)
    if art_at + burst_n <= emg_cp.shape[0]:
        bt = np.arange(burst_n) / fs
        wave = (
            ref
            * fault.motion_artifact_amplitude
            * np.sin(2.0 * np.pi * 7.0 * bt)
        )
        emg_cp[art_at : art_at + burst_n, :] += wave[:, None]

    return Segment(
        label=segment.label,
        emg=emg_cp,
        imu=segment.imu.copy(),
        duration_s=segment.duration_s,
        subject=segment.subject,
        trial=segment.trial,
    )
