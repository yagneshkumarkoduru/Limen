"""Tests for the domain adaptation stack: CORAL math, leakage guards, DANN."""

from __future__ import annotations

import numpy as np
import torch

from limen.dann import DannNet, GradientReversal, dann_schedule_lambda, recalibrate_batchnorm
from limen.domain_adaptation import (
    calibration_split,
    coral_transform,
    envelope_rms_normalize,
)
from limen.signals import Segment


# ---------------------------------------------------------------------------
# CORAL math
# ---------------------------------------------------------------------------

def test_coral_aligned_covariance_matches_target() -> None:
    """After centering and applying the CORAL map, source covariance equals target."""
    rng = np.random.default_rng(7)
    source = rng.standard_normal((500, 8)) @ np.diag(np.linspace(0.5, 2.0, 8))
    target = rng.standard_normal((400, 8)) @ rng.standard_normal((8, 8))
    transform = coral_transform(source, target)
    aligned = (source - source.mean(axis=0)) @ transform
    cov_aligned = np.cov(aligned, rowvar=False)
    cov_target = np.cov(target, rowvar=False)
    assert np.allclose(cov_aligned, cov_target, rtol=1e-6, atol=1e-6)


def test_coral_is_deterministic() -> None:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((200, 6))
    b = rng.standard_normal((150, 6))
    assert np.allclose(coral_transform(a, b), coral_transform(a, b))


def test_coral_rejects_shape_mismatch() -> None:
    a = np.zeros((10, 4))
    b = np.zeros((10, 5))
    try:
        coral_transform(a, b)
    except ValueError:
        return
    raise AssertionError("expected ValueError on feature dimension mismatch")


# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------

def _subject_segments(subject: str) -> list[Segment]:
    return [
        Segment(
            label=label,
            emg=np.ones((50, 2)),
            imu=np.zeros((5, 3)),
            duration_s=0.05,
            subject=subject,
            trial=trial,
        )
        for label in ("cyl", "hook", "tip", "palm", "spher", "lat")
        for trial in range(30)
    ]


def test_calibration_split_partitions_trials_disjointly() -> None:
    segs = _subject_segments("female_1")
    cal, test = calibration_split(segs, calibration_fraction=0.1)
    cal_trials = {(s.label, s.trial) for s in cal}
    test_trials = {(s.label, s.trial) for s in test}
    assert not (cal_trials & test_trials), "calibration/test trial leakage"
    assert len(cal) + len(test) == len(segs)
    # first trials per gesture calibrate: deterministic, stratified
    for label in ("cyl", "hook", "tip", "palm", "spher", "lat"):
        cal_t = sorted(s.trial for s in cal if s.label == label)
        assert cal_t == list(range(3)), f"{label}: calibration is not the first trials"


def test_calibration_split_rejects_multi_subject_input() -> None:
    segs = _subject_segments("female_1") + _subject_segments("male_1")
    try:
        calibration_split(segs)
    except ValueError:
        return
    raise AssertionError("expected ValueError for multi-subject input")


def test_envelope_rms_normalize_equalizes_subject_amplitude() -> None:
    rng = np.random.default_rng(11)
    segs = []
    for i, subject in enumerate(("female_1", "male_1")):
        emg = np.abs(rng.standard_normal((400, 2))) * (100.0 * (i + 1))
        segs.append(
            Segment(
                label="cyl", emg=emg, imu=np.zeros((40, 3)), duration_s=0.4,
                subject=subject, trial=0,
            )
        )
    out = envelope_rms_normalize(segs)
    levels = [float(np.abs(s.emg).mean()) for s in out]
    assert abs(levels[0] - levels[1]) / levels[0] < 0.01


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

def test_gradient_reversal_negates_and_scales_gradient() -> None:
    grl = GradientReversal(lambda_=0.5)
    x = torch.ones(4, requires_grad=True)
    y = grl(x).sum()
    y.backward()
    assert torch.allclose(x.grad, torch.full((4,), -0.5))


def test_gradient_reversal_forward_is_identity() -> None:
    grl = GradientReversal(lambda_=1.0)
    x = torch.randn(8)
    assert torch.allclose(grl(x), x)


def test_dann_lambda_schedule_is_monotonic_and_bounded() -> None:
    lams = [dann_schedule_lambda(e, 60, 1.0) for e in range(60)]
    assert all(0.0 <= lam <= 1.0 for lam in lams)
    assert all(b >= a for a, b in zip(lams, lams[1:]))
    assert lams[0] < 0.01 and lams[-1] > 0.99


# ---------------------------------------------------------------------------
# Batch-norm recalibration
# ---------------------------------------------------------------------------

def test_bn_recalibration_uses_only_supplied_windows() -> None:
    torch.manual_seed(0)
    model = DannNet(n_classes=6, n_domains=6, n_channels=2)
    cal = torch.randn(64, 2, 200) * 5.0
    recalibrate_batchnorm(model, cal)
    bn = next(m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d))
    # after recalibration the first BN's running stats reflect the cal data scale
    assert bn.running_var.mean() > 1.0
    assert model.training is False, "model must be back in eval mode after recalibration"


def test_bn_recalibration_does_not_touch_weights() -> None:
    torch.manual_seed(0)
    model = DannNet(n_classes=6, n_domains=6, n_channels=2)
    before = [p.clone() for p in model.parameters()]
    recalibrate_batchnorm(model, torch.randn(32, 2, 200))
    after = list(model.parameters())
    for b, a in zip(before, after):
        assert torch.allclose(b, a), "recalibration must not change weight tensors"


def test_dann_forward_shapes() -> None:
    model = DannNet(n_classes=6, n_domains=6, n_channels=2)
    task_logits, domain_logits = model(torch.randn(4, 2, 200))
    assert task_logits.shape == (4, 6)
    assert domain_logits.shape == (4, 6)
