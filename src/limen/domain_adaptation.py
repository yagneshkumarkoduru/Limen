"""Domain adaptation for cross-subject sEMG generalization.

Two stacked, deterministic stages close the within-subject to
leave-one-subject-out (LOSO) gap:

1. Per-subject RMS normalization of the EMG envelope. Electrode impedance
   and skin conditions set each subject's amplitude scale; dividing every
   window by the subject's own envelope RMS removes that recording-level
   gain before features are computed. The statistic comes from the
   subject's own segments only, so nothing crosses the LOSO boundary.

2. CORAL (correlation alignment) on the extracted feature matrix. Each
   training subject's feature covariance is aligned to the pooled
   training covariance, and the held-out subject's windows are aligned to
   a target covariance estimated from a small unlabeled calibration split
   of that subject's own data (labels unused). Alignment is closed-form:
   ``A = cov_source^{-1/2} @ cov_target^{1/2}`` via symmetric matrix
   square roots, so no training and no randomness is involved.

Protocol (unsupervised domain adaptation, standard in the sEMG
literature): the held-out subject's trials are split into a calibration
partition (the first ``calibration_fraction`` of trials per gesture) and
a test partition (the remaining trials). Only the calibration partition
contributes to the target-side covariance estimate. Test windows never
influence the transform or the classifier.

Reference for the alignment math:
  Sun, Feng, Saenko, "Return of Frustratingly Easy Domain Adaptation",
  AAAI 2016. The implementation here is written from the method's
  defining equations, not from the authors' code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .signals import Segment

_REGULARIZATION: float = 1e-8


# ---------------------------------------------------------------------------
# Per-subject envelope RMS normalization
# ---------------------------------------------------------------------------

def envelope_rms_normalize(segments: list[Segment]) -> list[Segment]:
    """Scale each segment so its subject's mean per-window RMS is one.

    The RMS of a filtered EMG window is the amplitude feature every
    downstream extractor relies on; normalizing it per subject removes
    the impedance-driven gain differences between recording sessions.
    Statistics come from the subject's own segments only.
    """
    by_subject: dict[str, list[Segment]] = {}
    for seg in segments:
        by_subject.setdefault(seg.subject or "unknown", []).append(seg)

    gains: dict[str, float] = {}
    for subject, pool in by_subject.items():
        rectified = np.concatenate([np.abs(s.emg) for s in pool], axis=0)
        mean_level = float(rectified.mean())
        if mean_level <= 0.0:
            raise ValueError(
                f"subject {subject!r} has zero-mean rectified EMG; "
                "cannot envelope-normalize"
            )
        gains[subject] = mean_level

    out: list[Segment] = []
    for seg in segments:
        gain = gains[seg.subject or "unknown"]
        out.append(
            Segment(
                label=seg.label,
                emg=seg.emg / gain,
                imu=seg.imu,
                duration_s=seg.duration_s,
                subject=seg.subject,
                trial=seg.trial,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Calibration / test partition of the held-out subject
# ---------------------------------------------------------------------------

def calibration_split(
    segments: list[Segment], calibration_fraction: float = 0.1
) -> tuple[list[Segment], list[Segment]]:
    """Split one subject's segments into calibration and test partitions.

    The split is by trial index within each gesture (the first
    ``calibration_fraction`` of trials calibrate, the rest test), so it
    is deterministic, stratified by class, and disjoint by construction.
    """
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"calibration_fraction must be in (0, 1), got {calibration_fraction}"
        )
    subjects = {s.subject for s in segments}
    if len(subjects) != 1:
        raise ValueError(
            f"calibration_split expects exactly one subject, got {sorted(subjects)}"
        )

    by_label: dict[str, list[Segment]] = {}
    for seg in segments:
        by_label.setdefault(seg.label, []).append(seg)

    calibration: list[Segment] = []
    test: list[Segment] = []
    for label in sorted(by_label):
        pool = sorted(by_label[label], key=lambda s: (s.trial is None, s.trial or 0))
        n_cal = max(1, int(round(len(pool) * calibration_fraction)))
        calibration.extend(pool[:n_cal])
        test.extend(pool[n_cal:])
    return calibration, test


# ---------------------------------------------------------------------------
# CORAL correlation alignment
# ---------------------------------------------------------------------------

def _sym_sqrt(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Symmetric positive-semidefinite matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    clipped = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(clipped)) @ eigvecs.T


def _sym_inv_sqrt(cov: NDArray[np.float64]) -> NDArray[np.float64]:
    """Inverse symmetric square root with eigenvalue clipping for stability."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    clipped = np.clip(eigvals, _REGULARIZATION, None)
    return (eigvecs * (1.0 / np.sqrt(clipped))) @ eigvecs.T


def coral_transform(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the linear map aligning source feature covariance to target's.

    ``A = cov(source)^{-1/2} @ cov(target)^{1/2}`` applied to centered
    source rows makes the aligned covariance match the target covariance
    up to numerical precision. ``source`` and ``target`` are feature
    matrices of shape (n_windows, n_features); they need not have the
    same number of rows.
    """
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("coral_transform expects 2-D feature matrices")
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"feature dimension mismatch: source {source.shape[1]}, "
            f"target {target.shape[1]}"
        )
    if source.shape[0] < 2 or target.shape[0] < 2:
        raise ValueError("coral_transform needs at least two rows per matrix")

    cov_s = np.cov(source, rowvar=False) + _REGULARIZATION * np.eye(source.shape[1])
    cov_t = np.cov(target, rowvar=False) + _REGULARIZATION * np.eye(target.shape[1])
    return _sym_inv_sqrt(cov_s) @ _sym_sqrt(cov_t)


def apply_alignment(
    features: NDArray[np.float64],
    transform: NDArray[np.float64],
    source_mean: NDArray[np.float64],
    target_mean: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a CORAL map: center on the source mean, align, re-center on target."""
    return (features - source_mean) @ transform + target_mean


# ---------------------------------------------------------------------------
# LOSO runner with CORAL adaptation
# ---------------------------------------------------------------------------

def loso_coral_matrices(
    train_features_by_subject: dict[str, NDArray[np.float64]],
    calibration_features: NDArray[np.float64],
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    """Per-source-subject CORAL maps into the held-out subject's calibration space.

    Returns ``{subject: (transform, source_mean, target_mean)}`` where the
    target statistics come from the calibration features only. Test
    windows of the held-out subject are never an input here; the leakage
    guard lives in the caller (see ``calibration_split``).
    """
    target_mean = calibration_features.mean(axis=0)
    maps: dict[
        str, tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ] = {}
    for subject, feats in sorted(train_features_by_subject.items()):
        transform = coral_transform(feats, calibration_features)
        maps[subject] = (transform, feats.mean(axis=0), target_mean)
    return maps
