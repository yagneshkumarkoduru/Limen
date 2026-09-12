"""Preprocessing: filtering, windowing, and feature extraction.

All operations are deterministic given the same input samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import signal as sp_signal

from .config import PreprocessConfig, SignalConfig
from .signals import Segment


@dataclass(frozen=True)
class EmgFilter:
    """Precomputed band-pass + notch filter coefficients for the EMG stream."""

    band_b: NDArray[np.float64]
    band_a: NDArray[np.float64]
    notch_b: NDArray[np.float64]
    notch_a: NDArray[np.float64]

    @classmethod
    def design(cls, pre: PreprocessConfig, sig: SignalConfig) -> "EmgFilter":
        nyq = sig.emg_sample_rate_hz / 2.0
        if not (0 < pre.bandpass_low_hz < pre.bandpass_high_hz < nyq):
            raise ValueError(
                f"Band-pass [{pre.bandpass_low_hz}, {pre.bandpass_high_hz}] Hz "
                f"invalid for Nyquist {nyq} Hz"
            )
        band_b, band_a = sp_signal.butter(
            pre.filter_order,
            [pre.bandpass_low_hz / nyq, pre.bandpass_high_hz / nyq],
            btype="band",
        )
        notch_b, notch_a = sp_signal.iirnotch(
            pre.notch_hz, Q=30.0, fs=sig.emg_sample_rate_hz
        )
        return cls(band_b=band_b, band_a=band_a, notch_b=notch_b, notch_a=notch_a)

    def apply(self, emg: NDArray[np.float64]) -> NDArray[np.float64]:
        """Zero-phase filtering along the sample axis (offline replay semantics)."""
        x = sp_signal.filtfilt(self.band_b, self.band_a, emg, axis=0)
        return sp_signal.filtfilt(self.notch_b, self.notch_a, x, axis=0)


@dataclass(frozen=True)
class Window:
    """One decision window: features plus the raw EMG block for quality checks."""

    label: str
    features: NDArray[np.float64]
    emg_block: NDArray[np.float64]
    start_s: float
    end_s: float


def extract_features(
    emg_block: NDArray[np.float64], imu_block: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Time-domain feature vector per window.

    EMG features per channel: MAV, RMS, waveform length, zero crossings,
    slope-sign changes. IMU features per channel: mean and standard deviation.
    """
    if emg_block.ndim != 2:
        raise ValueError(f"emg_block must be 2-D, got shape {emg_block.shape}")
    n = emg_block.shape[0]
    if n < 2:
        raise ValueError("emg_block too short for feature extraction")

    rect = np.abs(emg_block)
    mav = rect.mean(axis=0)
    rms = np.sqrt((emg_block**2).mean(axis=0))
    wl = np.abs(np.diff(emg_block, axis=0)).sum(axis=0)
    zc = ((emg_block[:-1] * emg_block[1:]) < 0).sum(axis=0).astype(np.float64)
    ssc = (
        ((emg_block[1:-1] - emg_block[:-2]) * (emg_block[1:-1] - emg_block[2:]) > 0)
        .sum(axis=0)
        .astype(np.float64)
    )
    imu_mean = imu_block.mean(axis=0)
    imu_std = imu_block.std(axis=0)
    return np.concatenate([mav, rms, wl, zc, ssc, imu_mean, imu_std])


def window_segment(
    segment: Segment,
    emg_filtered: NDArray[np.float64],
    pre: PreprocessConfig,
    sig: SignalConfig,
) -> list[Window]:
    """Slice a segment into overlapping decision windows with features."""
    win = int(pre.window_ms * sig.emg_sample_rate_hz / 1000)
    stride = int(pre.stride_ms * sig.emg_sample_rate_hz / 1000)
    imu_ratio = sig.emg_sample_rate_hz // sig.imu_sample_rate_hz
    if emg_filtered.shape[0] < win:
        raise ValueError(
            f"Segment {segment.label!r} too short: "
            f"{emg_filtered.shape[0]} samples < window {win}"
        )

    windows: list[Window] = []
    for start in range(0, emg_filtered.shape[0] - win + 1, stride):
        emg_block = emg_filtered[start : start + win]
        imu_start = start // imu_ratio
        imu_end = max(imu_start + 1, (start + win) // imu_ratio)
        imu_block = segment.imu[imu_start:imu_end]
        windows.append(
            Window(
                label=segment.label,
                features=extract_features(emg_block, imu_block),
                emg_block=emg_block,
                start_s=start / sig.emg_sample_rate_hz,
                end_s=(start + win) / sig.emg_sample_rate_hz,
            )
        )
    return windows
