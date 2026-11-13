"""Real data adapter: UCI sEMG for Basic Hand Movements (Dataset 313, Database 1).

Loads all five subjects, six gestures, and thirty trials per gesture from
.mat files, resamples from 500 Hz to 1000 Hz, converts electrode potentials
from volts to microvolts, and presents each trial as a
:class:`~limen.signals.Segment` with subject/trial provenance.

Signal construction per trial:
- EMG: ``resample_poly(x, 2, 1)`` to 1000 Hz, then ``* 1e6`` to convert the
  dataset's volts to the microvolt unit system the SignalQualityGate
  documents. The two recorded channels are kept as recorded; earlier versions
  expanded them to eight virtual channels with a fixed mixing matrix, which
  created artificial common-mode coupling and made the safety gate flag every
  window as an artifact. No mixing is applied here.
- IMU: the dataset contains no IMU; the field is a zero-filled placeholder
  (marked unavailable) and no metric depends on it.

Dataset citation:
  C. Sapsanis, G. Georgoulas, A. Tzes, "sEMG for Basic Hand Movements",
  UCI Machine Learning Repository, 2013 (DOI 10.24432/C5TK53; CC BY 4.0).

Files are not redistributed here.  Point ``UCISEMGAdapter`` at the extracted
copy of ``semg+for+basic+hand+movements.zip``, which should produce a
``Database 1/`` subdirectory containing five ``<subject>.mat`` files.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.signal import resample_poly

from .benchmark import replay
from .config import BenchmarkConfig, SignalConfig
from .inference import LdaBaseline, RandomForestBaseline, ShrinkageLdaBaseline
from .preprocessing import EmgFilter, extract_features, extract_features_rich
from .report import ReportContext, build_report_dict
from .signals import Segment

# ---------------------------------------------------------------------------
# Dataset constants
# ---------------------------------------------------------------------------

SUBJECTS: tuple[str, ...] = ("female_1", "female_2", "female_3", "male_1", "male_2")
GESTURES: tuple[str, ...] = ("cyl", "hook", "tip", "palm", "spher", "lat")

_UCI_HZ: int = 500          # native sample rate of the UCI dataset
_TARGET_HZ: int = 1000      # resampled target, matching SignalConfig default
_VOLTS_TO_UV: float = 1e6   # dataset records electrode potentials in volts

# Proxy actuator command for UCI gesture classes. The safety envelope maps
# each recognized intent to a bounded actuator request; unknown intents halt.
_UCI_PROXY_COMMAND: dict[str, tuple[float, float]] = {g: (25.0, 3.0) for g in GESTURES}


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def uci_benchmark_config() -> BenchmarkConfig:
    """Return a :class:`~limen.config.BenchmarkConfig` matched to the UCI dataset.

    Differences from the default Limen config:
    - ``intent_classes``: the six UCI gesture names instead of knee-joint classes.
    - ``n_emg_channels``: 2 (the recorded channels; no mixing).
    - ``n_imu_channels``: 3 (zero-filled placeholder; the dataset has no IMU).
    - ``segment_seconds``: 6.0 s (3000 raw samples upsampled by 2).
    - ``segments_per_class``: 30 (one per trial).
    """
    return BenchmarkConfig(
        intent_classes=GESTURES,
        signal=SignalConfig(
            emg_sample_rate_hz=_TARGET_HZ,
            imu_sample_rate_hz=100,
            n_emg_channels=2,
            n_imu_channels=3,
            segment_seconds=6.0,
            rest_seconds=2.0,
            segments_per_class=30,
        ),
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class UCISEMGAdapter:
    """Dataset adapter for UCI sEMG for Basic Hand Movements, Database 1.

    Each trial in the dataset is presented as one
    :class:`~limen.signals.Segment` carrying its subject and trial index.
    Loading all five subjects, six gestures, and thirty trials yields 900
    segments in total, ordered by subject, then gesture, then trial index.

    Args:
        data_dir: Path to the extracted dataset root.  Must contain a
            ``Database 1/`` subdirectory with the five subject ``.mat`` files.

    Raises:
        FileNotFoundError: If ``data_dir/Database 1/`` is absent, or any
            subject file is missing when :meth:`load` is called.
    """

    name = "uci-semg-basic-hand-movements-db1"
    license = "CC BY 4.0 (DOI 10.24432/C5TK53)"

    def __init__(self, data_dir: str | Path) -> None:
        self._root = Path(data_dir)
        db1 = self._root / "Database 1"
        if not db1.is_dir():
            raise FileNotFoundError(
                f"'Database 1' subdirectory not found under {self._root}. "
                "Download and extract the UCI sEMG dataset, then point "
                "--data-dir at the extracted root directory."
            )
        self._db1 = db1

    def load(self, cfg: BenchmarkConfig) -> list[Segment]:
        """Load all subjects, gestures, and trials as a flat Segment list."""
        if cfg.signal.n_emg_channels != 2:
            raise ValueError(
                f"UCISEMGAdapter requires n_emg_channels == 2, "
                f"got {cfg.signal.n_emg_channels}.  Use uci_benchmark_config()."
            )

        segments: list[Segment] = []
        for subject in SUBJECTS:
            mat = self._load_mat(subject)
            for gesture in GESTURES:
                ch1 = np.asarray(mat[f"{gesture}_ch1"], dtype=np.float64)
                ch2 = np.asarray(mat[f"{gesture}_ch2"], dtype=np.float64)
                for trial in range(ch1.shape[0]):
                    segments.append(
                        _build_segment(gesture, ch1[trial], ch2[trial], subject, trial)
                    )

        if not segments:
            raise RuntimeError(
                "UCISEMGAdapter.load produced zero segments; verify data_dir contents."
            )
        return segments

    def _load_mat(self, subject: str) -> dict:
        path = self._db1 / f"{subject}.mat"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing subject file: {path}. "
                "Ensure all five .mat files are present in 'Database 1/'."
            )
        return loadmat(str(path))


def _build_segment(
    label: str,
    raw_ch1: NDArray[np.float64],
    raw_ch2: NDArray[np.float64],
    subject: str,
    trial: int,
) -> Segment:
    """Construct one Segment from a single UCI trial.

    Steps: resample 500 Hz -> 1000 Hz, convert volts to microvolts, stack the
    two recorded channels, and attach a zero-filled IMU placeholder.
    """
    r1 = resample_poly(raw_ch1, 2, 1).astype(np.float64) * _VOLTS_TO_UV
    r2 = resample_poly(raw_ch2, 2, 1).astype(np.float64) * _VOLTS_TO_UV
    n = min(len(r1), len(r2))  # guard against off-by-one differences

    emg = np.column_stack([r1[:n], r2[:n]])
    imu = np.zeros((n // 10, 3), dtype=np.float64)  # dataset has no IMU

    return Segment(
        label=label,
        emg=emg,
        imu=imu,
        duration_s=n / float(_TARGET_HZ),
        subject=subject,
        trial=trial,
    )


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------

def within_subject_split(
    segments: list[Segment], train_fraction: float = 0.6, seed: int = 20261011
) -> dict[str, tuple[list[Segment], list[Segment]]]:
    """Per-subject trial split; no window from an eval trial appears in train.

    Trials are permuted per (subject, gesture) with a deterministic generator,
    so the split is reproducible and stratified by class.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, tuple[list[Segment], list[Segment]]] = {}
    for subject in SUBJECTS:
        train: list[Segment] = []
        eval_: list[Segment] = []
        for gesture in GESTURES:
            class_segments = [s for s in segments if s.subject == subject and s.label == gesture]
            order = rng.permutation(len(class_segments))
            n_train = max(1, int(round(len(class_segments) * train_fraction)))
            for rank, idx in enumerate(order):
                (train if rank < n_train else eval_).append(class_segments[int(idx)])
        out[subject] = (train, eval_)
    return out


def loso_split(segments: list[Segment]) -> dict[str, tuple[list[Segment], list[Segment]]]:
    """Leave-one-subject-out split: eval subject never appears in train."""
    out: dict[str, tuple[list[Segment], list[Segment]]] = {}
    for held_out in SUBJECTS:
        train = [s for s in segments if s.subject != held_out]
        eval_ = [s for s in segments if s.subject == held_out]
        out[held_out] = (train, eval_)
    return out


def subject_znormalize(segments: list[Segment]) -> list[Segment]:
    """Per-subject, per-channel z-normalization of the raw EMG stream.

    Cross-subject amplitude and offset differences (electrode placement,
    skin impedance, subcutaneous fat) dominate the time-domain feature
    vector and are the largest single cause of the within-subject to
    leave-one-subject-out accuracy drop. Standardizing each subject's
    channels to zero mean and unit variance removes that recording-level
    shift while preserving within-subject gesture structure. Statistics
    come from the subject's own segments only, so no information crosses
    the LOSO train/eval boundary beyond the subject's own recordings.

    The standardized stream is rescaled to the dataset-wide median
    per-channel amplitude so that downstream absolute thresholds (the
    SignalQualityGate flatline floor, in microvolts) remain calibrated.
    """
    by_subject: dict[str, list[Segment]] = {}
    for seg in segments:
        key = seg.subject or "unknown"
        by_subject.setdefault(key, []).append(seg)
    stats: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for key, pool in by_subject.items():
        flat = np.concatenate([s.emg for s in pool], axis=0)
        stats[key] = (flat.mean(axis=0), flat.std(axis=0) + 1e-12)
    global_scale = float(
        np.median(np.concatenate([v[1] for v in stats.values()]))
    )
    out: list[Segment] = []
    for seg in segments:
        mean, std = stats[seg.subject or "unknown"]
        out.append(
            dataclasses.replace(seg, emg=(seg.emg - mean) / std * global_scale)
        )
    return out


# ---------------------------------------------------------------------------
# Protocol runner
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _FeatureSpec:
    name: str
    fn: Callable[..., NDArray[np.float64]]
    kind: str  # "time" or "rich"


_FEATURE_SPECS: tuple[_FeatureSpec, ...] = (
    _FeatureSpec("lda-time", extract_features, "time"),
    _FeatureSpec("lda-rich", extract_features_rich, "rich"),
    _FeatureSpec("rf-rich", extract_features_rich, "rich"),
)


def _window_features(
    segments: list[Segment], emg_filter: EmgFilter, cfg: BenchmarkConfig, spec: _FeatureSpec
) -> tuple[NDArray[np.float64], list[str], list[str]]:
    """Feature rows for a segment list, with label and group (subject|trial)."""
    pre, sig = cfg.preprocess, cfg.signal
    win = int(pre.window_ms * sig.emg_sample_rate_hz / 1000)
    stride = int(pre.stride_ms * sig.emg_sample_rate_hz / 1000)
    imu_ratio = sig.emg_sample_rate_hz // sig.imu_sample_rate_hz

    rows: list[NDArray[np.float64]] = []
    labels: list[str] = []
    groups: list[str] = []
    for seg in segments:
        filtered = emg_filter.apply(seg.emg)
        for start in range(0, filtered.shape[0] - win + 1, stride):
            emg_block = filtered[start : start + win]
            i0 = start // imu_ratio
            imu_block = seg.imu[i0 : max(i0 + 1, (start + win) // imu_ratio)]
            if spec.kind == "rich":
                rows.append(spec.fn(emg_block, imu_block, fs=float(sig.emg_sample_rate_hz)))
            else:
                rows.append(spec.fn(emg_block, imu_block))
            labels.append(seg.label)
            groups.append(f"{seg.subject}|{seg.trial}")
    return np.vstack(rows), labels, groups


def _make_model(name: str, seed: int) -> LdaBaseline | ShrinkageLdaBaseline | RandomForestBaseline:
    if name == "lda-time":
        return LdaBaseline()
    if name == "lda-rich":
        return ShrinkageLdaBaseline()
    if name == "rf-rich":
        return RandomForestBaseline(seed=seed)
    raise ValueError(f"unknown model {name!r}")


def _cv_accuracy(
    model_name: str,
    x: NDArray[np.float64],
    y: list[str],
    groups: list[str],
    seed: int,
) -> float:
    """Stratified group CV accuracy (groups keep trial windows together)."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = 5
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, test_idx in splitter.split(x, y, groups):
        model = _make_model(model_name, seed)
        model.fit(x[train_idx], [y[i] for i in train_idx])
        correct = sum(
            1 for i in test_idx if model.predict(x[i]).intent == y[i]
        )
        scores.append(correct / len(test_idx))
    return float(np.mean(scores))


def _nominal_amplitude(segments: list[Segment]) -> float:
    """Median per-channel std across eval segments (fault-injection scale)."""
    medians = [float(np.median(np.std(seg.emg, axis=0))) for seg in segments]
    return float(np.median(medians))


def _per_subject_accuracy(result) -> dict[str, float]:
    by_subject: dict[str, list[int]] = {}
    for r in result.telemetry.records:
        key = r.subject or "unknown"
        by_subject.setdefault(key, [0, 0])
        by_subject[key][1] += 1
        if r.predicted_intent == r.true_label:
            by_subject[key][0] += 1
    return {k: round(v[0] / max(1, v[1]), 6) for k, v in by_subject.items()}


def run_real_protocols(
    data_dir: str | Path,
    out_dir: str | Path,
    quick: bool = False,
    seed: int = 20261011,
    normalize_subjects: bool = True,
) -> dict:
    """Run within-subject and leave-one-subject-out protocols on the UCI dataset.

    Trial-disjoint, subject-disjoint evaluation; model selection only from
    training-fold cross-validation; fault injection scaled to the recording's
    own amplitude. When ``normalize_subjects`` is true each subject's raw EMG
    is z-normalized per channel before windowing (domain adaptation against
    cross-subject recording shift). Writes ``real_semg_within_subject.json``,
    ``real_semg_loso.json`` and ``real_semg_summary.json`` under ``out_dir``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = uci_benchmark_config()
    adapter = UCISEMGAdapter(data_dir)
    segments = adapter.load(cfg)
    if normalize_subjects:
        segments = subject_znormalize(segments)
    emg_filter = EmgFilter.design(cfg.preprocess, cfg.signal)
    command_map = dict(_UCI_PROXY_COMMAND)

    specs = list(_FEATURE_SPECS)
    within = within_subject_split(segments, train_fraction=0.6, seed=seed)
    loso = loso_split(segments)
    if quick:
        specs = specs[:1]
        first_subject = SUBJECTS[0]
        within = {first_subject: within[first_subject]}
        loso = {first_subject: loso[first_subject]}

    # ---- within-subject -----------------------------------------------------
    within_train = [s for pair in within.values() for s in pair[0]]
    within_eval = [s for pair in within.values() for s in pair[1]]
    rich_fn = partial(extract_features_rich, fs=float(cfg.signal.emg_sample_rate_hz))
    within_results = []
    within_selection: dict[str, float] = {}
    for spec in specs:
        x_tr, y_tr, g_tr = _window_features(within_train, emg_filter, cfg, spec)
        cv = _cv_accuracy(spec.name, x_tr, y_tr, g_tr, seed)
        within_selection[spec.name] = round(cv, 6)
        model = _make_model(spec.name, seed)
        model.fit(x_tr, y_tr)
        for condition in ("clean", "fault_injected"):
            within_results.append(
                replay(
                    cfg=cfg,
                    model=model,
                    model_name=spec.name,
                    eval_segments=within_eval,
                    emg_filter=emg_filter,
                    condition=condition,
                    n_train_windows=len(y_tr),
                    feature_fn=rich_fn if spec.kind == "rich" else extract_features,
                    command_map=command_map,
                    nominal_amplitude=_nominal_amplitude(within_eval),
                )
            )

    ctx = ReportContext(
        dataset_name=adapter.name,
        dataset_license=adapter.license,
        dataset_kind="real",
        protocol=f"within-subject, trial-disjoint, {int(0.6*30)}/{30-int(0.6*30)} trials per class",
    )
    within_report = build_report_dict(cfg, within_results, ctx)
    within_report["cv_accuracy_train_only"] = within_selection
    within_report["per_subject_accuracy"] = {
        r.model_name + "|" + r.condition: _per_subject_accuracy(r) for r in within_results
    }
    best_model = max(within_selection, key=lambda k: within_selection[k])
    within_report["selected_model_by_cv"] = best_model
    (out_dir / "real_semg_within_subject.json").write_text(
        json.dumps(within_report, indent=2) + "\n", encoding="utf-8"
    )

    # ---- leave-one-subject-out ---------------------------------------------
    loso_results = []
    loso_selection: dict[str, dict[str, float]] = {}
    for held_out, (train_segs, eval_segs) in loso.items():
        fold_cv: dict[str, float] = {}
        fold_features: dict[str, tuple] = {}
        for spec in specs:
            x_tr, y_tr, g_tr = _window_features(train_segs, emg_filter, cfg, spec)
            fold_features[spec.name] = (x_tr, y_tr)
            fold_cv[spec.name] = _cv_accuracy(spec.name, x_tr, y_tr, g_tr, seed)
        loso_selection[held_out] = {k: round(v, 6) for k, v in fold_cv.items()}
        for spec in specs:
            x_tr, y_tr = fold_features[spec.name]
            model = _make_model(spec.name, seed)
            model.fit(x_tr, y_tr)
            for condition in ("clean", "fault_injected"):
                loso_results.append(
                    replay(
                        cfg=cfg,
                        model=model,
                        model_name=f"{spec.name}|held_out={held_out}",
                        eval_segments=eval_segs,
                        emg_filter=emg_filter,
                        condition=condition,
                        n_train_windows=len(y_tr),
                        feature_fn=rich_fn if spec.kind == "rich" else extract_features,
                        command_map=command_map,
                        nominal_amplitude=_nominal_amplitude(eval_segs),
                    )
                )

    loso_ctx = ReportContext(
        dataset_name=adapter.name,
        dataset_license=adapter.license,
        dataset_kind="real",
        protocol="leave-one-subject-out (all trials of the held-out subject)",
    )
    loso_report = build_report_dict(cfg, loso_results, loso_ctx)
    loso_report["cv_accuracy_train_only"] = loso_selection
    (out_dir / "real_semg_loso.json").write_text(
        json.dumps(loso_report, indent=2) + "\n", encoding="utf-8"
    )

    # ---- summary ------------------------------------------------------------
    def _selected_fold_mean(report: dict, selection: dict[str, dict[str, float]], condition: str) -> dict:
        """Mean over folds of the model each fold's train-only CV selected."""
        per_fold: dict[str, dict[str, dict]] = {}
        for r in report["runs"]:
            if r["condition"] != condition:
                continue
            model, held = r["model"].split("|held_out=")
            per_fold.setdefault(held, {})[model] = r
        accs, rejs, acc_accepted, chosen = [], [], [], {}
        for held, cv in selection.items():
            best = max(cv, key=lambda k: cv[k])
            chosen[held] = best
            run = per_fold.get(held, {}).get(best)
            if run is None:
                continue
            accs.append(run["accuracy"])
            rejs.append(run["rejection_rate"])
            if run["accuracy_on_accepted"] is not None:
                acc_accepted.append(run["accuracy_on_accepted"])
        return {
            "n_runs": len(accs),
            "selected_model_per_fold": chosen,
            "accuracy_mean": round(float(np.mean(accs)), 6) if accs else None,
            "accuracy_std": round(float(np.std(accs)), 6) if accs else None,
            "rejection_mean": round(float(np.mean(rejs)), 6) if rejs else None,
            "accuracy_on_accepted_mean": (
                round(float(np.mean(acc_accepted)), 6) if acc_accepted else None
            ),
        }

    def _runs_mean(report: dict, model_prefix: str, condition: str) -> dict[str, float | int | None]:
        runs = [
            r
            for r in report["runs"]
            if r["model"].startswith(model_prefix) and r["condition"] == condition
        ]
        accs = [r["accuracy"] for r in runs]
        rej = [r["rejection_rate"] for r in runs]
        acc_accepted = [
            r["accuracy_on_accepted"] for r in runs if r["accuracy_on_accepted"] is not None
        ]
        return {
            "n_runs": len(accs),
            "accuracy_mean": round(float(np.mean(accs)), 6) if accs else None,
            "accuracy_std": round(float(np.std(accs)), 6) if accs else None,
            "rejection_mean": round(float(np.mean(rej)), 6) if rej else None,
            "accuracy_on_accepted_mean": (
                round(float(np.mean(acc_accepted)), 6) if acc_accepted else None
            ),
        }

    summary = {
        "dataset": adapter.name,
        "protocols": {
            "within_subject": {
                "cv_accuracy_train_only": within_selection,
                "selected_model_by_cv": best_model,
                "runs": {
                    f"{spec.name}|{cond}": _runs_mean(within_report, spec.name, cond)
                    for spec in specs
                    for cond in ("clean", "fault_injected")
                },
            },
            "loso": {
                f"{spec.name}|{cond}": _runs_mean(loso_report, spec.name, cond)
                for spec in specs
                for cond in ("clean", "fault_injected")
            },
            "loso_selected_by_cv": {
                cond: _selected_fold_mean(loso_report, loso_selection, cond)
                for cond in ("clean", "fault_injected")
            },
        },
        "preprocessing": {
            "subject_znormalization": bool(normalize_subjects),
            "domain_adaptation": (
                "per-subject per-channel z-normalization of raw EMG; "
                "statistics from the subject's own segments only"
            ),
        },
    }
    (out_dir / "real_semg_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
