"""Fault-injection tests for the Confidence-Coupled Safety Envelope.

These tests are implementation evidence for the confidence-coupled safety
architecture: the envelope must degrade or halt deterministically when
confidence or signal quality collapses, and the independent supervisor must
veto any command beyond absolute hardware bounds - even a NOMINAL one.
"""

from __future__ import annotations

import numpy as np

from limen.config import BenchmarkConfig
from limen.inference import Prediction
from limen.safety import (
    ConfidenceCoupledSafetyEnvelope,
    EnvelopeState,
    IndependentHardwareSupervisor,
    SignalQualityGate,
    ActuatorCommand,
)
from limen.signals import generate_segment, inject_faults


def _cfg() -> BenchmarkConfig:
    return BenchmarkConfig()


def test_full_confidence_clean_signal_stays_nominal() -> None:
    cfg = _cfg()
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("knee_flexion", cfg.signal, np.random.default_rng(7))
    cmd = env.evaluate(
        Prediction(intent="knee_flexion", confidence=0.95), sqg.assess(seg.emg[2000:2200])
    )
    assert cmd.envelope_state is EnvelopeState.NOMINAL
    assert cmd.torque_nm > 0 and cmd.velocity_rad_s > 0


def test_electrode_dropout_triggers_persistent_safe_halt() -> None:
    cfg = _cfg()
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("knee_extension", cfg.signal, np.random.default_rng(11))
    faulty = inject_faults(seg, cfg.fault, cfg.signal)

    win = int(0.2 * cfg.signal.emg_sample_rate_hz)
    start = int(3.2 * cfg.signal.emg_sample_rate_hz)
    dropout_block = faulty.emg[start : start + win]
    dropout_cmd = env.evaluate(
        Prediction(intent="knee_extension", confidence=0.9), sqg.assess(dropout_block)
    )
    assert dropout_cmd.envelope_state is EnvelopeState.DEGRADED
    assert dropout_cmd.torque_nm < 25.0

    artifact_block = faulty.emg[4500:4700]
    states = []
    for _ in range(cfg.safety.persistence_windows + 1):
        cmd = env.evaluate(
            Prediction(intent="knee_extension", confidence=0.9), sqg.assess(artifact_block)
        )
        states.append(cmd.envelope_state)
    assert states[-1] is EnvelopeState.SAFE_HALT
    assert states.index(EnvelopeState.SAFE_HALT) >= cfg.safety.persistence_windows - 1


def test_degraded_confidence_damps_output() -> None:
    cfg = _cfg()
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("knee_flexion", cfg.signal, np.random.default_rng(13))
    block = seg.emg[2000:2200]
    q = sqg.assess(block)
    target = (cfg.safety.t_low + cfg.safety.t_high) / 2.0
    conf = min(1.0, target / q.reliability)
    cmd = env.evaluate(Prediction(intent="knee_flexion", confidence=conf), q)
    assert cmd.envelope_state is EnvelopeState.DEGRADED
    nominal_torque = 25.0
    assert cmd.torque_nm == nominal_torque * cfg.safety.degraded_force_scale
    assert cmd.velocity_rad_s == 3.0 * cfg.safety.degraded_velocity_scale


def test_supervisor_vetoes_beyond_hard_bounds() -> None:
    cfg = _cfg()
    sup = IndependentHardwareSupervisor(cfg.safety)
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("knee_flexion", cfg.signal, np.random.default_rng(17))
    q = sqg.assess(seg.emg[2000:2200])

    cmd = env.evaluate(Prediction(intent="knee_flexion", confidence=0.99), q)
    assert not cmd.vetoed_by_supervisor

    over = ActuatorCommand(
        intent="knee_flexion",
        torque_nm=cfg.safety.hard_max_torque_nm * 2,
        velocity_rad_s=cfg.safety.hard_max_velocity_rad_s * 3,
        envelope_state=EnvelopeState.NOMINAL,
        effective_confidence=1.0,
        vetoed_by_supervisor=False,
    )
    bounded = sup.supervise(over)
    assert bounded.vetoed_by_supervisor
    assert bounded.torque_nm == cfg.safety.hard_max_torque_nm
    assert bounded.velocity_rad_s == cfg.safety.hard_max_velocity_rad_s


def test_unknown_intent_fails_safe() -> None:
    cfg = _cfg()
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("rest", cfg.signal, np.random.default_rng(19))
    q = sqg.assess(seg.emg[:200])
    cmd = env.evaluate(Prediction(intent="backflip", confidence=0.99), q)
    assert cmd.envelope_state is EnvelopeState.SAFE_HALT
    assert cmd.torque_nm == 0.0 and cmd.velocity_rad_s == 0.0


def test_recovery_requires_high_confidence() -> None:
    cfg = _cfg()
    env = ConfidenceCoupledSafetyEnvelope(cfg.safety)
    sqg = SignalQualityGate()
    seg = generate_segment("knee_flexion", cfg.signal, np.random.default_rng(23))
    block = seg.emg[2000:2200]
    q = sqg.assess(block)

    for _ in range(cfg.safety.persistence_windows + 1):
        env.evaluate(Prediction(intent="knee_flexion", confidence=0.10), q)
    assert env.state is EnvelopeState.SAFE_HALT

    env.evaluate(Prediction(intent="knee_flexion", confidence=0.60), q)
    assert env.state is EnvelopeState.DEGRADED
    cmd = env.evaluate(Prediction(intent="knee_flexion", confidence=0.95), q)
    assert cmd.envelope_state is EnvelopeState.NOMINAL
