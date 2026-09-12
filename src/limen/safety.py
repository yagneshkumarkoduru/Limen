"""Confidence-Coupled Safety Envelope - core research component.

Deterministic safety layer between probabilistic intent inference and the
(simulated) actuator. This module is the implementation evidence for the
confidence-coupled safety architecture; every behavior is covered by
fault-injection tests in ``tests/test_safety_envelope.py``.

Pipeline per decision window:
    raw EMG window -> SignalQualityGate -> reliability R in [0, 1]
    inference -> intent I, confidence C in [0, 1]
    C_eff = C * R
    C_eff >= T_high            -> NOMINAL: full authority
    T_low <= C_eff < T_high    -> DEGRADED: velocity/force damped
    C_eff < T_low (persistent) -> SAFE_HALT: hold position, zero velocity
    always -> IndependentHardwareSupervisor vetoes any command beyond
              absolute hardware bounds, regardless of confidence.

No randomness, no wall-clock dependence: identical inputs always produce
identical commands.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import SafetyConfig
from .inference import Prediction


class EnvelopeState(enum.Enum):
    NOMINAL = "nominal"
    DEGRADED = "degraded"
    SAFE_HALT = "safe_halt"


@dataclass(frozen=True)
class SignalQuality:
    """Output of the Signal Quality Gate for one window."""

    reliability: float  # R in [0, 1]
    flatline_channels: tuple[int, ...]
    artifact_detected: bool

    def __post_init__(self) -> None:
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError(f"reliability {self.reliability} outside [0, 1]")


class SignalQualityGate:
    """Estimates signal reliability R from a raw EMG sensor window.

    Checks, each producing a multiplicative penalty:
    - flatline/contact loss: channel standard deviation below ``flatline_std_uv``
    - clipping/saturation: fraction of samples beyond ``clip_ratio`` of the
      observed full-scale range
    - motion artifact: excessive common-mode power characteristic of
      synchronized motion artifacts
    """

    def __init__(self, flatline_std_uv: float = 5.0, clip_ratio: float = 0.98) -> None:
        self._flatline_std_uv = flatline_std_uv
        self._clip_ratio = clip_ratio

    def assess(self, emg_block: NDArray[np.float64]) -> SignalQuality:
        if emg_block.ndim != 2 or emg_block.shape[0] < 2:
            raise ValueError(f"emg_block shape {emg_block.shape} not assessable")

        n_channels = emg_block.shape[1]
        flat: list[int] = []
        reliability = 1.0

        for ch in range(n_channels):
            std = float(np.std(emg_block[:, ch]))
            if std < self._flatline_std_uv:
                flat.append(ch)
        if flat:
            reliability *= max(0.0, 1.0 - 0.25 * len(flat))

        full_scale = float(np.max(np.abs(emg_block))) + 1e-12
        clipped = float(np.mean(np.abs(emg_block) > self._clip_ratio * full_scale))
        if clipped > 0.01:
            reliability *= 0.6

        demeaned = emg_block - emg_block.mean(axis=0)
        total_power = float(np.mean(demeaned**2)) + 1e-12
        common_mode = demeaned.mean(axis=1)
        common_mode_ratio = float(np.mean(common_mode**2) / (total_power / n_channels))
        artifact = common_mode_ratio > 2.5
        if artifact:
            reliability *= 0.3

        return SignalQuality(
            reliability=round(reliability, 6),
            flatline_channels=tuple(flat),
            artifact_detected=artifact,
        )


@dataclass(frozen=True)
class ActuatorCommand:
    """Bounded command emitted for one decision window."""

    intent: str
    torque_nm: float
    velocity_rad_s: float
    envelope_state: EnvelopeState
    effective_confidence: float
    vetoed_by_supervisor: bool


# Nominal per-intent torque/velocity requests (simulated actuator abstraction).
_NOMINAL_REQUEST: dict[str, tuple[float, float]] = {
    "rest": (0.0, 0.0),
    "knee_flexion": (25.0, 3.0),
    "knee_extension": (25.0, 3.0),
}


class ConfidenceCoupledSafetyEnvelope:
    """Maps (intent, confidence, signal quality) to a bounded actuator command.

    State machine with halt persistence: SAFE_HALT is entered only after
    ``persistence_windows`` consecutive windows below T_low, preventing
    single-window noise flicker from freezing the actuator; recovery to
    NOMINAL requires one window at or above T_high.
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        if not 0.0 < cfg.t_low < cfg.t_high < 1.0:
            raise ValueError(f"need 0 < T_low < T_high < 1, got {cfg.t_low}, {cfg.t_high}")
        self._cfg = cfg
        self._low_streak = 0
        self._state = EnvelopeState.NOMINAL

    @property
    def state(self) -> EnvelopeState:
        return self._state

    def reset(self) -> None:
        self._low_streak = 0
        self._state = EnvelopeState.NOMINAL

    def evaluate(self, prediction: Prediction, quality: SignalQuality) -> ActuatorCommand:
        c_eff = round(prediction.confidence * quality.reliability, 6)

        if c_eff < self._cfg.t_low:
            self._low_streak += 1
        else:
            self._low_streak = 0

        if self._low_streak >= self._cfg.persistence_windows:
            self._state = EnvelopeState.SAFE_HALT
        elif c_eff >= self._cfg.t_high:
            self._state = EnvelopeState.NOMINAL
        elif c_eff >= self._cfg.t_low:
            self._state = EnvelopeState.DEGRADED

        base_torque, base_velocity = _NOMINAL_REQUEST.get(prediction.intent, (0.0, 0.0))
        if prediction.intent not in _NOMINAL_REQUEST:
            self._state = EnvelopeState.SAFE_HALT

        if self._state is EnvelopeState.NOMINAL:
            torque, velocity = base_torque, base_velocity
        elif self._state is EnvelopeState.DEGRADED:
            torque = base_torque * self._cfg.degraded_force_scale
            velocity = base_velocity * self._cfg.degraded_velocity_scale
        else:  # SAFE_HALT
            torque, velocity = 0.0, 0.0

        command = ActuatorCommand(
            intent=prediction.intent,
            torque_nm=torque,
            velocity_rad_s=velocity,
            envelope_state=self._state,
            effective_confidence=c_eff,
            vetoed_by_supervisor=False,
        )
        return IndependentHardwareSupervisor(self._cfg).supervise(command)


class IndependentHardwareSupervisor:
    """Final veto layer: absolute hardware bounds, never relaxable.

    Models the MCU/FPGA-resident supervisor that overrides any command -
    including NOMINAL ones - exceeding absolute torque/velocity limits.
    """

    def __init__(self, cfg: SafetyConfig) -> None:
        self._cfg = cfg

    def supervise(self, cmd: ActuatorCommand) -> ActuatorCommand:
        bounded_torque = float(
            np.clip(cmd.torque_nm, -self._cfg.hard_max_torque_nm, self._cfg.hard_max_torque_nm)
        )
        bounded_velocity = float(
            np.clip(
                cmd.velocity_rad_s,
                -self._cfg.hard_max_velocity_rad_s,
                self._cfg.hard_max_velocity_rad_s,
            )
        )
        vetoed = (bounded_torque != cmd.torque_nm) or (bounded_velocity != cmd.velocity_rad_s)
        if not vetoed:
            return cmd
        return ActuatorCommand(
            intent=cmd.intent,
            torque_nm=bounded_torque,
            velocity_rad_s=bounded_velocity,
            envelope_state=cmd.envelope_state,
            effective_confidence=cmd.effective_confidence,
            vetoed_by_supervisor=True,
        )
