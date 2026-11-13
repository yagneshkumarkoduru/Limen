"""Merge pre-fix (before) LOSO metrics into the regenerated summary as a
before/after comparison block. Run after run_real_semg_benchmark.py.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    before = json.loads((RESULTS / "_before_summary_snapshot.json").read_text())
    summary_path = RESULTS / "real_semg_summary.json"
    summary = json.loads(summary_path.read_text())

    def loso_acc(block: dict, key: str) -> float | None:
        entry = block.get(key)
        return entry.get("accuracy_mean") if entry else None

    after = summary["protocols"]["loso"]
    comparison = {}
    for key in sorted(before["loso"]):
        b = before["loso"][key].get("accuracy_mean")
        a = after.get(key, {}).get("accuracy_mean")
        comparison[key] = {
            "before_accuracy_mean": b,
            "after_accuracy_mean": a,
            "delta": round(a - b, 6) if (a is not None and b is not None) else None,
        }

    summary["before_after"] = {
        "change": (
            "per-subject per-channel z-normalization of raw EMG plus "
            "train-only CV model selection added to the LOSO protocol"
        ),
        "loso_accuracy": comparison,
        "loso_selected_by_cv": summary["protocols"].get("loso_selected_by_cv"),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "_before_summary_snapshot.json").unlink()
    print("summary updated with before/after")


if __name__ == "__main__":
    main()
