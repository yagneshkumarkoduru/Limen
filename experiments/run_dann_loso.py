"""DANN + CORAL + per-subject BN leave-one-subject-out benchmark.

Runs the three-stage domain adaptation stack on the UCI sEMG dataset:

- Stage 1: CORAL pre-alignment of source-subject windows into the
  held-out subject's calibration covariance space.
- Stage 2: domain-adversarial training (task classifier + subject
  discriminator with gradient reversal).
- Stage 3: per-subject batch-norm recalibration from the held-out
  subject's unlabeled calibration windows.

Strict protocol: the held-out subject's trials are split into a
calibration partition (first trials per gesture, labels unused) and a
test partition. Test windows never influence CORAL, adversarial
training, or batch-norm statistics.

Usage:
    python experiments/run_dann_loso.py --data-dir data [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from limen.dann import (  # noqa: E402
    DannNet,
    DannTrainConfig,
    predict_labels,
    recalibrate_batchnorm,
    train_dann,
)
from limen.datasets_real import (  # noqa: E402
    GESTURES,
    SUBJECTS,
    UCISEMGAdapter,
    uci_benchmark_config,
)
from limen.domain_adaptation import (  # noqa: E402
    calibration_split,
    coral_transform,
    envelope_rms_normalize,
)
from limen.preprocessing import EmgFilter  # noqa: E402

CLASS_TO_IDX = {g: i for i, g in enumerate(GESTURES)}
SUBJECT_TO_DOMAIN = {s: i for i, s in enumerate(SUBJECTS)}


def _raw_windows(segments, emg_filter, cfg) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Filtered EMG windows as (n_windows, n_channels, n_samples), labels, subjects."""
    win = int(cfg.preprocess.window_ms * cfg.signal.emg_sample_rate_hz / 1000)
    stride = int(cfg.preprocess.stride_ms * cfg.signal.emg_sample_rate_hz / 1000)
    xs, ys, subs = [], [], []
    for seg in segments:
        filtered = emg_filter.apply(seg.emg)
        for start in range(0, filtered.shape[0] - win + 1, stride):
            xs.append(filtered[start : start + win].T.copy())
            ys.append(CLASS_TO_IDX[seg.label])
            subs.append(seg.subject or "unknown")
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64), subs


def _coral_align(
    source_x: np.ndarray, calibration_x: np.ndarray
) -> np.ndarray:
    """Align source windows into the calibration covariance space (flattened CORAL)."""
    n, c, t = source_x.shape
    src_flat = source_x.reshape(n, -1).astype(np.float64)
    cal_flat = calibration_x.reshape(calibration_x.shape[0], -1).astype(np.float64)
    transform = coral_transform(src_flat, cal_flat)
    src_mean = src_flat.mean(axis=0)
    cal_mean = cal_flat.mean(axis=0)
    aligned = (src_flat - src_mean) @ transform + cal_mean
    return aligned.reshape(n, c, t).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_PROJECT_ROOT / "data"))
    parser.add_argument("--out-dir", default=str(_PROJECT_ROOT / "results"))
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = uci_benchmark_config()
    adapter = UCISEMGAdapter(args.data_dir)
    segments = envelope_rms_normalize(adapter.load(cfg))
    emg_filter = EmgFilter.design(cfg.preprocess, cfg.signal)

    train_cfg = DannTrainConfig(epochs=args.epochs, seed=args.seed, device=args.device)
    held_domain = len(SUBJECTS)  # target domain index: one past source domains

    folds: dict[str, dict] = {}
    for held_out in SUBJECTS:
        held_segments = [s for s in segments if s.subject == held_out]
        source_segments = [s for s in segments if s.subject != held_out]
        cal_segments, test_segments = calibration_split(
            held_segments, calibration_fraction=args.calibration_fraction
        )

        src_x, src_y, src_subs = _raw_windows(source_segments, emg_filter, cfg)
        cal_x, _, _ = _raw_windows(cal_segments, emg_filter, cfg)
        test_x, test_y, _ = _raw_windows(test_segments, emg_filter, cfg)

        src_aligned = _coral_align(src_x, cal_x)
        src_domain = np.asarray([SUBJECT_TO_DOMAIN[s] for s in src_subs], dtype=np.int64)

        model = DannNet(
            n_classes=len(GESTURES),
            n_domains=len(SUBJECTS) + 1,
            n_channels=cfg.signal.n_emg_channels,
        )
        train_dann(
            model,
            torch.from_numpy(src_aligned),
            torch.from_numpy(src_y),
            torch.from_numpy(src_domain),
            torch.from_numpy(cal_x),
            target_domain=held_domain,
            cfg=train_cfg,
        )

        # Stage 3: per-subject BN recalibration from unlabeled calibration windows
        cal_tensor = torch.from_numpy(cal_x).to(args.device)
        recalibrate_batchnorm(model, cal_tensor)

        preds = predict_labels(model, torch.from_numpy(test_x).to(args.device))
        acc = float(np.mean(preds == test_y))
        folds[held_out] = {
            "accuracy": round(acc, 6),
            "n_test_windows": int(test_x.shape[0]),
            "n_calibration_windows": int(cal_x.shape[0]),
            "n_train_windows": int(src_x.shape[0]),
        }
        print(f"[{held_out}] LOSO accuracy {acc:.4f} ({test_x.shape[0]} test windows)")

    accs = [f["accuracy"] for f in folds.values()]
    report = {
        "experiment": "dann+coral+per-subject-bn loso",
        "seed": args.seed,
        "epochs": args.epochs,
        "calibration_fraction": args.calibration_fraction,
        "protocol": (
            "leave-one-subject-out; held-out subject split by trial into "
            "calibration (first trials per gesture, labels unused) and test "
            "partitions; test windows never influence CORAL, training, or "
            "batch-norm statistics"
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "per_subject": folds,
        "aggregate": {
            "accuracy_mean": round(float(np.mean(accs)), 6),
            "accuracy_std": round(float(np.std(accs)), 6),
            "n_folds": len(accs),
        },
    }
    out_path = out_dir / "real_semg_loso_dann.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nAggregate LOSO accuracy: {report['aggregate']['accuracy_mean']:.4f} "
          f"+/- {report['aggregate']['accuracy_std']:.4f}")
    print(f"Report written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
