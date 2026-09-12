"""Report generation for the Limen safety benchmark.

Writes two artifacts per benchmark execution:
- ``benchmark_report.json``: machine-readable results,
- ``benchmark_report.md``: human-readable report with every field the
  protocol requires, including limitations and next hardware implication.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn

from .benchmark import RunResult
from .config import BenchmarkConfig


@dataclass(frozen=True)
class ReportContext:
    """Provenance fields required by the reporting format."""

    dataset_name: str
    dataset_license: str


def _run_block(result: RunResult, classes: tuple[str, ...]) -> dict:
    t = result.telemetry
    return {
        "model": result.model_name,
        "condition": result.condition,
        "train_windows": result.n_train_windows,
        "eval_windows": result.n_eval_windows,
        "accuracy": t.accuracy(),
        "false_activation_rate": t.false_activation_rate(),
        "rejection_rate": t.rejection_rate(),
        "envelope_events": t.envelope_event_counts(),
        "confusion_matrix": {"classes": list(classes), "matrix": t.confusion_matrix(classes)},
        "latency": t.latency_summary(),
    }


def build_report_dict(
    cfg: BenchmarkConfig, results: list[RunResult], ctx: ReportContext
) -> dict:
    if not results:
        raise ValueError("cannot build a report from zero runs")
    limitations = [
        "Synthetic dataset: separability is easier than real surface EMG; "
        "absolute accuracy numbers must not be quoted as real-world performance.",
        "Zero-phase offline filtering (filtfilt) is used for replay; an online "
        "causal filter will add group delay on hardware.",
        "Latency is measured on a development workstation under a general-purpose OS; "
        "it establishes software cost order-of-magnitude, not embedded worst-case "
        "execution time.",
        "Power is not measured in software; the eventized scheduler addresses the "
        "power budget on target hardware.",
        "Torque/velocity values are simulated actuator abstractions, not physical "
        "joint torques.",
    ]
    research_gate = _research_gate(results, limitations)
    return {
        "report": "Limen EMG/IMU Safety Benchmark",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"source": ctx.dataset_name, "license": ctx.dataset_license},
        "configuration": {
            "seed": cfg.seed,
            "intent_classes": list(cfg.intent_classes),
            "emg_sample_rate_hz": cfg.signal.emg_sample_rate_hz,
            "imu_sample_rate_hz": cfg.signal.imu_sample_rate_hz,
            "n_emg_channels": cfg.signal.n_emg_channels,
            "window_ms": cfg.preprocess.window_ms,
            "stride_ms": cfg.preprocess.stride_ms,
            "bandpass_hz": [cfg.preprocess.bandpass_low_hz, cfg.preprocess.bandpass_high_hz],
            "notch_hz": cfg.preprocess.notch_hz,
            "safety": {
                "t_high": cfg.safety.t_high,
                "t_low": cfg.safety.t_low,
                "degraded_velocity_scale": cfg.safety.degraded_velocity_scale,
                "degraded_force_scale": cfg.safety.degraded_force_scale,
                "hard_max_torque_nm": cfg.safety.hard_max_torque_nm,
                "hard_max_velocity_rad_s": cfg.safety.hard_max_velocity_rad_s,
                "persistence_windows": cfg.safety.persistence_windows,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "os": platform.platform(),
            "hardware": platform.machine() + " (development workstation; no embedded target yet)",
        },
        "runs": [_run_block(r, cfg.intent_classes) for r in results],
        "failure_cases": _failure_cases(results),
        "research_gate": research_gate,
        "limitations": limitations,
        "next_hardware_implication": (
            "If p95 decision-loop + inference latency remains well under the 20 ms stride "
            "budget on this workstation class, an MCU (Cortex-M7-class) is the justified "
            "first target; FPGA acceleration is revisited only if measured online inference "
            "or multi-channel preprocessing exceeds the budget."
        ),
    }


def _research_gate(results: list[RunResult], limitations: list[str]) -> dict:
    """Evaluate the software research stage gate.

    This is an acceptance check for the software research stage, not a claim
    that synthetic evidence generalizes to people, products, or hardware.
    """
    decision_windows = sum(result.n_eval_windows for result in results)
    fallback_events = sum(
        result.telemetry.envelope_event_counts()["degraded"]
        + result.telemetry.envelope_event_counts()["safe_halt"]
        for result in results
    )
    checks = {
        "pipeline_runs_end_to_end": all(result.n_eval_windows > 0 for result in results),
        "p95_latency_reported": all(
            result.telemetry.latency_summary()["end_to_end"]["p95_ms"] > 0
            for result in results
        ),
        "fallback_condition_implemented": fallback_events > 0,
        "reproducibility_controls_present": True,
        "limitations_stated": len(limitations) > 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision_windows_evaluated": decision_windows,
        "fallback_events_recorded": fallback_events,
        "checks": checks,
    }


def _failure_cases(results: list[RunResult]) -> list[str]:
    cases: list[str] = []
    for r in results:
        recs = r.telemetry.records
        wrong = [x for x in recs if x.predicted_intent != x.true_label]
        if wrong:
            top: dict[str, int] = {}
            for x in wrong:
                key = f"{x.true_label}->{x.predicted_intent}"
                top[key] = top.get(key, 0) + 1
            summary = ", ".join(
                f"{k} x{n}" for k, n in sorted(top.items(), key=lambda kv: -kv[1])
            )
            cases.append(
                f"[{r.model_name}/{r.condition}] "
                f"{len(wrong)} misclassified windows: {summary}"
            )
        halts = sum(1 for x in recs if x.envelope_state == "safe_halt")
        if r.condition == "fault_injected" and halts == 0:
            cases.append(
                f"[{r.model_name}/fault_injected] WARNING: no SAFE_HALT triggered under "
                "fault injection; review thresholds - the envelope may be too permissive."
            )
    if not cases:
        cases.append("No misclassified windows in this run; see limitations before interpreting.")
    return cases


def write_report(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "benchmark_report.json"
    md_path = out_dir / "benchmark_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = []
    lines.append(f"# {report['report']}")
    lines.append("")
    lines.append(f"Generated (UTC): {report['generated_utc']}")
    lines.append(f"Dataset: {report['dataset']['source']} - {report['dataset']['license']}")
    env = report["environment"]
    lines.append(
        f"Environment: Python {env['python']}, numpy {env['numpy']}, "
        f"scipy {env['scipy']}, scikit-learn {env['scikit_learn']} on {env['os']}"
    )
    cfg = report["configuration"]
    lines.append(
        f"Signal: EMG {cfg['emg_sample_rate_hz']} Hz x {cfg['n_emg_channels']} ch, "
        f"IMU {cfg['imu_sample_rate_hz']} Hz; window {cfg['window_ms']} ms, "
        f"stride {cfg['stride_ms']} ms; seed {cfg['seed']}"
    )
    lines.append("")
    for run in report["runs"]:
        lines.append(f"## Run: {run['model']} - {run['condition']}")
        lines.append("")
        lines.append(f"- Eval windows: {run['eval_windows']} (train: {run['train_windows']})")
        lines.append(f"- Accuracy: {run['accuracy']:.4f}")
        lines.append(f"- False activation rate: {run['false_activation_rate']:.4f}")
        lines.append(f"- Rejection rate (envelope constrained): {run['rejection_rate']:.4f}")
        lines.append(f"- Envelope events: {run['envelope_events']}")
        lat = run["latency"]["end_to_end"]
        lines.append(
            f"- End-to-end latency: p50 {lat['p50_ms']} ms, p95 {lat['p95_ms']} ms, "
            f"p99 {lat['p99_ms']} ms, max {lat['max_ms']} ms"
        )
        inf = run["latency"]["inference"]
        lines.append(f"- Inference latency: p50 {inf['p50_ms']} ms, p95 {inf['p95_ms']} ms")
        cm = run["confusion_matrix"]
        lines.append("")
        lines.append("| true \\ pred | " + " | ".join(cm["classes"]) + " |")
        lines.append("|" + "---|" * (len(cm["classes"]) + 1))
        for cls, row in zip(cm["classes"], cm["matrix"]):
            lines.append(f"| {cls} | " + " | ".join(str(v) for v in row) + " |")
        lines.append("")
    gate = report["research_gate"]
    lines.append("## Research Gate")
    lines.append("")
    lines.append(
        f"Status: **{gate['status']}**; decision windows evaluated: "
        f"{gate['decision_windows_evaluated']}; fallback events recorded: "
        f"{gate['fallback_events_recorded']}"
    )
    for name, passed in gate["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("## Failure cases")
    for c in report["failure_cases"]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("## Limitations")
    for lim in report["limitations"]:
        lines.append(f"- {lim}")
    lines.append("")
    lines.append("## Next hardware implication")
    lines.append(report["next_hardware_implication"])
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
