"""Limen safety benchmark on the UCI sEMG real dataset (honest protocols).

Runs two evaluation protocols on real surface-EMG recordings from the UCI
"sEMG for Basic Hand Movements" dataset (Dataset 313, Database 1):

- within-subject, trial-disjoint: for each subject, 60% of trials per gesture
  train, 40% evaluate; no window from an evaluation trial appears in training.
- leave-one-subject-out: the held-out subject's trials are never seen in
  training.

Model selection uses training-fold cross-validation only (stratified group CV
over trials). Fault injection (electrode dropout, motion artifact) is scaled
to the recording's own amplitude, and the safety gate's artifact threshold is
calibrated on clean data statistics.

Dataset citation:
  C. Sapsanis, G. Georgoulas, A. Tzes, "sEMG for Basic Hand Movements",
  UCI Machine Learning Repository, 2013 (DOI 10.24432/C5TK53; CC BY 4.0).

Usage:
    python experiments/run_real_semg_benchmark.py --data-dir data
    python experiments/run_real_semg_benchmark.py --data-dir data --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the package importable when this script is run directly from the repo.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from limen.datasets_real import GESTURES, UCISEMGAdapter, run_real_protocols  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Limen safety benchmark on the UCI sEMG hand-movement dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The dataset must be downloaded separately from the UCI ML Repository\n"
            "(DOI 10.24432/C5TK53; CC BY 4.0) and extracted so that --data-dir\n"
            "contains a 'Database 1/' subdirectory with five subject .mat files."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=str(_PROJECT_ROOT / "data"),
        metavar="PATH",
        help="Path to the extracted UCI sEMG dataset root (must contain 'Database 1/').",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "results"),
        metavar="PATH",
        help="Directory for the protocol reports (default: results/).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Trial run: first subject only, first model only (sanity check).",
    )
    args = parser.parse_args(argv)

    print(f"Dataset : {args.data_dir}")
    print(f"Output  : {args.out_dir}")
    print(f"Classes : {', '.join(GESTURES)}")
    print(f"Mode    : {'quick (trial)' if args.quick else 'full'}")
    print()

    adapter = UCISEMGAdapter(args.data_dir)  # raises FileNotFoundError if missing
    print(f"Adapter : {adapter.name}  |  {adapter.license}")
    print("Loading : 5 subjects x 6 gestures x 30 trials = 900 segments ...")

    summary = run_real_protocols(
        data_dir=args.data_dir, out_dir=args.out_dir, quick=args.quick
    )

    print()
    print("=== within-subject (trial-disjoint) ===")
    ws = summary["protocols"]["within_subject"]
    print(f"selected model by train-only CV: {ws['selected_model_by_cv']}")
    for key, stats in ws["runs"].items():
        print(
            f"  {key:<24} acc {stats['accuracy_mean']:.4f} "
            f"+/- {stats['accuracy_std']:.4f}  rejection {stats['rejection_mean']:.4f}"
        )
    print()
    print("=== leave-one-subject-out ===")
    for key, stats in summary["protocols"]["loso"].items():
        print(
            f"  {key:<24} acc {stats['accuracy_mean']:.4f} "
            f"+/- {stats['accuracy_std']:.4f}  rejection {stats['rejection_mean']:.4f}"
        )
    print()
    print(f"Reports written under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
