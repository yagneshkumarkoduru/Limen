"""Tests for the real-data pipeline: units, gate calibration, protocols, gate logic.

These tests pin the fixes for the previously broken real-data path: the
volts-to-microvolts conversion, the channel-count-aware artifact detector,
the amplitude-scaled fault injection, trial-disjoint and subject-disjoint
protocol splits, and the research gate's accuracy floor.
"""

from __future__ import annotations

import numpy as np

from limen.benchmark import RunResult
from limen.datasets_real import (
    GESTURES,
    SUBJECTS,
    loso_split,
    subject_znormalize,
    within_subject_split,
)
from limen.report import ReportContext, build_report_dict
from limen.config import BenchmarkConfig
from limen.safety import SignalQualityGate
from limen.signals import Segment, inject_faults
from limen.telemetry import Telemetry, WindowRecord


# ---------------------------------------------------------------------------
# SignalQualityGate units and artifact calibration
# ---------------------------------------------------------------------------

def test_gate_flags_volts_scale_as_flatline() -> None:
    """A window in volts (std ~0.2) is below the 5 uV floor: flagged flatlined.

    This documents the original bug: raw dataset values in volts were compared
    against a microvolt threshold, so every channel read as flatlined. The
    adapter now converts to microvolts before the gate sees the data.
    """
    rng = np.random.default_rng(3)
    volts_block = rng.standard_normal((200, 2)) * 0.2
    q = SignalQualityGate().assess(volts_block)
    assert q.flatline_channels == (0, 1)
    assert q.reliability < 1.0


def test_gate_accepts_microvolt_scale() -> None:
    rng = np.random.default_rng(3)
    uv_block = rng.standard_normal((200, 2)) * 2.0e5  # real electrode scale in uV
    q = SignalQualityGate().assess(uv_block)
    assert q.flatline_channels == ()
    assert q.reliability == 1.0
    assert not q.artifact_detected


def test_gate_detects_common_mode_artifact_two_channels() -> None:
    rng = np.random.default_rng(5)
    t = np.arange(200) / 1000.0
    clean = rng.standard_normal((200, 2)) * 2.0e5
    q_clean = SignalQualityGate().assess(clean)
    assert not q_clean.artifact_detected

    artifact = clean + (8.0 * 2.0e5 * np.sin(2 * np.pi * 7.0 * t))[:, None]
    q_art = SignalQualityGate().assess(artifact)
    assert q_art.artifact_detected
    assert q_art.reliability <= 0.3 + 1e-9


def test_gate_detects_common_mode_artifact_eight_channels() -> None:
    rng = np.random.default_rng(7)
    t = np.arange(200) / 1000.0
    clean = rng.standard_normal((200, 8)) * 120.0
    assert not SignalQualityGate().assess(clean).artifact_detected
    artifact = clean + (960.0 * np.sin(2 * np.pi * 7.0 * t))[:, None]
    assert SignalQualityGate().assess(artifact).artifact_detected


def test_gate_does_not_flag_correlated_but_natural_activity() -> None:
    """Correlated physiological activity up to rho ~0.85 must not flag.

    Calibration basis: on 27,000 clean UCI windows the implied rho never
    exceeded 0.856; the detector threshold sits at 0.90.
    """
    rng = np.random.default_rng(11)
    common = rng.standard_normal((200, 1)) * 1.0
    shared = common + rng.standard_normal((200, 1)) * 0.6  # rho ~ 0.73
    ch2 = common * 0.8 + rng.standard_normal((200, 1)) * 1.2
    block = 2.0e5 * np.hstack([shared, ch2])
    q = SignalQualityGate().assess(block)
    assert not q.artifact_detected


# ---------------------------------------------------------------------------
# Fault injection scaling
# ---------------------------------------------------------------------------

def _real_scale_segment() -> Segment:
    rng = np.random.default_rng(13)
    emg = rng.standard_normal((6000, 2)) * 2.8e5  # real electrode scale, 6 s at 1 kHz
    return Segment(
        label="cyl",
        emg=emg,
        imu=np.zeros((600, 3)),
        duration_s=6.0,
        subject="female_1",
        trial=0,
    )


def test_fault_injection_scales_to_recording_amplitude() -> None:
    cfg = BenchmarkConfig()
    seg = _real_scale_segment()
    nominal = float(np.median(np.std(seg.emg, axis=0)))
    faulty = inject_faults(seg, cfg.fault, cfg.signal, nominal_amplitude=nominal)
    burst = faulty.emg[4500:4700]
    clean_block = seg.emg[100:300]
    assert np.std(burst) > 3.0 * np.std(clean_block)

    # Without the measured nominal the burst is invisible at this scale:
    # the non-dropped channel (index 0; the dropout takes index 1) is unchanged.
    unscaled = inject_faults(seg, cfg.fault, cfg.signal)
    clean_window_std = float(np.std(seg.emg[4500:4700, 0]))
    unscaled_window_std = float(np.std(unscaled.emg[4500:4700, 0]))
    assert abs(unscaled_window_std - clean_window_std) / clean_window_std < 0.05


def test_fault_injection_preserves_provenance() -> None:
    cfg = BenchmarkConfig()
    seg = _real_scale_segment()
    faulty = inject_faults(seg, cfg.fault, cfg.signal, nominal_amplitude=2.8e5)
    assert faulty.subject == "female_1"
    assert faulty.trial == 0


# ---------------------------------------------------------------------------
# Protocol splits
# ---------------------------------------------------------------------------

def _metadata_segments() -> list[Segment]:
    out: list[Segment] = []
    for subject in SUBJECTS:
        for label in GESTURES:
            for trial in range(30):
                out.append(
                    Segment(
                        label=label,
                        emg=np.zeros((50, 2)),
                        imu=np.zeros((5, 3)),
                        duration_s=0.05,
                        subject=subject,
                        trial=trial,
                    )
                )
    return out


def test_within_subject_split_is_trial_disjoint_and_deterministic() -> None:
    segments = _metadata_segments()
    a = within_subject_split(segments, train_fraction=0.6, seed=42)
    b = within_subject_split(segments, train_fraction=0.6, seed=42)
    for subject in SUBJECTS:
        train, eval_ = a[subject]
        train_keys = {(s.label, s.trial) for s in train}
        eval_keys = {(s.label, s.trial) for s in eval_}
        assert not (train_keys & eval_keys), f"trial leakage for {subject}"
        assert len(eval_) == 6 * 12 and len(train) == 6 * 18
    for subject in SUBJECTS:
        assert {(s.label, s.trial) for s in a[subject][0]} == {
            (s.label, s.trial) for s in b[subject][0]
        }


def test_loso_split_has_no_subject_leakage() -> None:
    segments = _metadata_segments()
    folds = loso_split(segments)
    assert set(folds) == set(SUBJECTS)
    for held_out, (train, eval_) in folds.items():
        assert {s.subject for s in eval_} == {held_out}
        assert held_out not in {s.subject for s in train}
        assert {s.subject for s in train} == set(SUBJECTS) - {held_out}


def test_subject_znormalize_standardizes_each_subject_independently() -> None:
    rng = np.random.default_rng(17)
    segments = []
    for i, subject in enumerate(("female_1", "male_1")):
        scale = 1.0e5 * (i + 1)
        offset = 3.0e4 * (i + 1)
        emg = rng.standard_normal((400, 2)) * scale + offset
        segments.append(
            Segment(
                label="cyl",
                emg=emg,
                imu=np.zeros((40, 3)),
                duration_s=0.4,
                subject=subject,
                trial=0,
            )
        )
    out = subject_znormalize(segments)
    # mean removed per subject; std equalized to the dataset-wide median scale
    for seg in out:
        assert np.allclose(seg.emg.mean(axis=0), 0.0, atol=1e-6)
    scales = [float(np.std(s.emg)) for s in out]
    assert max(scales) / min(scales) < 1.5
    # original segments untouched (Segment is frozen)
    assert not np.allclose(segments[1].emg.mean(axis=0), 0.0, atol=1e-6)


def test_subject_znormalize_preserves_labels_and_provenance() -> None:
    seg = Segment(
        label="hook",
        emg=np.ones((50, 2)) * 5.0e4,
        imu=np.zeros((5, 3)),
        duration_s=0.05,
        subject="male_2",
        trial=7,
    )
    out = subject_znormalize([seg])
    assert out[0].label == "hook"
    assert out[0].subject == "male_2"
    assert out[0].trial == 7


# ---------------------------------------------------------------------------
# Research gate
# ---------------------------------------------------------------------------

def _record(true_label: str, predicted: str, state: str = "nominal") -> WindowRecord:
    return WindowRecord(
        true_label=true_label,
        predicted_intent=predicted,
        confidence=0.5,
        reliability=1.0,
        effective_confidence=0.5,
        envelope_state=state,
        vetoed=False,
        commanded_torque_nm=0.0,
        preprocess_ns=1000,
        inference_ns=1000,
        decision_ns=1000,
    )


def _chance_result() -> RunResult:
    telemetry = Telemetry()
    for i in range(600):
        true = GESTURES[i % len(GESTURES)]
        predicted = GESTURES[(i + 1) % len(GESTURES)]  # never correct
        telemetry.add(_record(true, predicted, state="degraded" if i % 2 else "nominal"))
    return RunResult(
        model_name="chance-model",
        condition="clean",
        telemetry=telemetry,
        n_train_windows=100,
        n_eval_windows=600,
    )


def test_research_gate_fails_at_chance() -> None:
    cfg = BenchmarkConfig(intent_classes=GESTURES)
    report = build_report_dict(
        cfg,
        [_chance_result()],
        ReportContext(dataset_name="t", dataset_license="t", dataset_kind="real", protocol="t"),
    )
    assert report["research_gate"]["status"] == "FAIL"
    assert report["research_gate"]["checks"]["accuracy_above_chance"] is False


def test_false_activation_rate_none_without_rest_windows() -> None:
    telemetry = Telemetry()
    telemetry.add(_record("cyl", "cyl"))
    assert telemetry.false_activation_rate() is None
