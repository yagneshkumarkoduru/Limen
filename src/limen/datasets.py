"""Dataset adapters for the Limen replay benchmark.

The benchmark consumes a uniform ``list[Segment]`` stream. The synthetic
adapter is the default, fully reproducible source. Real public datasets
(e.g., NinaPro, CapgMyo) plug in through the same protocol - and fail loudly
when their files are absent rather than silently substituting synthetic data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .config import BenchmarkConfig
from .signals import Segment, generate_dataset


class DatasetAdapter(Protocol):
    """Uniform interface between data sources and the benchmark runner."""

    name: str
    license: str

    def load(self, cfg: BenchmarkConfig) -> list[Segment]: ...


class SyntheticAdapter:
    """Deterministic synthetic EMG/IMU source (default benchmark dataset)."""

    name = "limen-synthetic-emg-imu"
    license = "Generated in-repo (deterministic, seeded); no external license terms"

    def load(self, cfg: BenchmarkConfig) -> list[Segment]:
        segments = generate_dataset(cfg)
        if not segments:
            raise RuntimeError("synthetic generator produced zero segments")
        return segments


class NinaProAdapter:
    """Placeholder adapter for the NinaPro DB (sEMG hand/wrist gestures).

    Not wired into the default benchmark: the dataset requires a manual
    download and license acceptance. When the files are present under
    ``data/ninapro``, this adapter will parse them; until then it refuses
    explicitly.
    """

    name = "ninapro-db"
    license = "NinaPro research license (manual acceptance required)"

    def __init__(self, root: Path) -> None:
        self._root = root

    def load(self, cfg: BenchmarkConfig) -> list[Segment]:
        raise FileNotFoundError(
            f"NinaPro files not found under {self._root}. Download the dataset, "
            "accept its license, place .mat files there, and implement parsing "
            "before selecting this adapter. The benchmark never falls back to "
            "synthetic data implicitly."
        )
