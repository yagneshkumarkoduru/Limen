"""Command-line entry point: run the full Limen safety benchmark.

Usage (from the Limen folder, with the project venv active):
    python -m limen [--out out/]

Executes four runs - both baselines (classical LDA, temporal) under both
conditions (clean, fault_injected) - and writes JSON + Markdown reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .benchmark import replay, split_segments, training_matrix
from .config import BenchmarkConfig
from .datasets import SyntheticAdapter
from .inference import LdaBaseline, TemporalBaseline
from .preprocessing import EmgFilter
from .report import ReportContext, build_report_dict, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="limen", description="Limen safety benchmark")
    parser.add_argument("--out", default="out", help="output directory for reports")
    args = parser.parse_args(argv)

    cfg = BenchmarkConfig()
    adapter = SyntheticAdapter()
    segments = adapter.load(cfg)
    train_segments, eval_segments = split_segments(segments, cfg)
    emg_filter = EmgFilter.design(cfg.preprocess, cfg.signal)

    x_train, y_train = training_matrix(train_segments, emg_filter, cfg)

    models: list[tuple[str, LdaBaseline | TemporalBaseline]] = []
    lda = LdaBaseline()
    lda.fit(x_train, y_train)
    models.append(("classical-lda", lda))
    temporal = TemporalBaseline(history=5, hysteresis=4)
    temporal.fit(x_train, y_train)
    models.append(("temporal-lda-smoothing", temporal))

    results = []
    for name, model in models:
        for condition in ("clean", "fault_injected"):
            results.append(
                replay(
                    cfg=cfg,
                    model=model,
                    model_name=name,
                    eval_segments=eval_segments,
                    emg_filter=emg_filter,
                    condition=condition,
                    n_train_windows=len(y_train),
                )
            )

    ctx = ReportContext(dataset_name=adapter.name, dataset_license=adapter.license)
    report = build_report_dict(cfg, results, ctx)
    json_path, md_path = write_report(report, Path(args.out))

    for r in results:
        lat = r.telemetry.latency_summary()["end_to_end"]
        print(
            f"{r.model_name:24s} {r.condition:15s} acc={r.telemetry.accuracy():.4f} "
            f"far={r.telemetry.false_activation_rate():.4f} "
            f"p95={lat['p95_ms']}ms events={r.telemetry.envelope_event_counts()}"
        )
    print(f"\nReport written: {json_path} , {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
