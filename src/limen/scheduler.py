"""Cycle-grid multi-rate fabric for deterministic replay and HIL planning.

Supervision runs on a fixed continuous cadence while the costlier
inference path is eventized: an inference cycle only fires when at least
one meaningful signal event arrived since the previous inference slot.
The fabric never touches a wall clock, so a given event list and horizon
always reproduce the identical emission stream. Within one timestamp the
emission order is:

    safety tick -> inference (when activated) -> telemetry tick

This is a virtual fabric. It makes no real-time scheduling claims on a
general-purpose operating system.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FabricEventKind(enum.Enum):
    """Category of one emitted fabric event."""

    SAFETY_TICK = "safety_tick"
    INFERENCE = "inference"
    TELEMETRY_TICK = "telemetry_tick"

    @property
    def is_periodic_supervision(self) -> bool:
        """True for the mandatory supervision cadence channel."""
        return self is FabricEventKind.SAFETY_TICK

    @property
    def is_event_driven(self) -> bool:
        """True for channels gated by meaningful signal activity."""
        return self is FabricEventKind.INFERENCE


@dataclass(frozen=True)
class SignalEvent:
    """One observed input event offered to the fabric."""

    timestamp_us: int; meaningful: bool

    def __post_init__(self) -> None:
        if self.timestamp_us < 0:
            raise ValueError("signal timestamp_us must be non-negative")


@dataclass(frozen=True)
class FabricEvent:
    """One control-fabric emission."""

    timestamp_us: int; kind: FabricEventKind


@dataclass(frozen=True, slots=True)
class FabricRun:
    """Emission stream plus deterministic accounting."""

    events: tuple[FabricEvent, ...]; safety_ticks: int; inference_events: int
    telemetry_ticks: int; missed_safety_deadlines: int


class EventizedMultiRateFabric:
    """Multi-rate virtual fabric with a mandatory supervision cadence.

    Supervision floor: the documented safety cadence runs at 1 kHz, and
    any period under 100 us exceeds every supported supervision rate, so
    the constructor refuses it.
    """

    MIN_SAFETY_PERIOD_US = 100

    def __init__(self, safety_period_us: int = 1_000, inference_period_us: int = 20_000, telemetry_period_us: int = 20_000) -> None:
        for label in ("safety_period_us", "inference_period_us", "telemetry_period_us"):
            candidate = {"safety_period_us": safety_period_us, "inference_period_us": inference_period_us, "telemetry_period_us": telemetry_period_us}[label]
            if not isinstance(candidate, int) or candidate <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if safety_period_us < self.MIN_SAFETY_PERIOD_US:
            raise ValueError(f"safety_period_us must be at least {self.MIN_SAFETY_PERIOD_US} us")
        if inference_period_us % safety_period_us != 0:
            raise ValueError("safety period must not be slower than inference period")
        self._periods = (safety_period_us, inference_period_us, telemetry_period_us)

    @staticmethod
    def _validate_run_inputs(signal_events: tuple[SignalEvent, ...], end_time_us: int) -> None:
        if not isinstance(end_time_us, int) or end_time_us < 0:
            raise ValueError("end_time_us must be a non-negative integer")
        for event in signal_events:
            if not isinstance(event, SignalEvent):
                raise TypeError("signal_events must contain only SignalEvent values")
        horizon_us = end_time_us
        previous_stamp: int | None = None
        for event in signal_events:
            stamp = event.timestamp_us
            if stamp == previous_stamp:
                raise ValueError("duplicate signal_event timestamps are not supported")
            if previous_stamp is not None and stamp < previous_stamp:
                raise ValueError("signal_events must be ordered by timestamp")
            if stamp > horizon_us:
                raise ValueError("signal event occurs after end_time_us")
            previous_stamp = stamp

    def _grid(self, end_time_us: int, extra_instants: set[int]) -> list[tuple[int, tuple[bool, bool, bool]]]:
        """Merge the three periodic channels plus ``extra_instants`` into one grid.

        Produces (timestamp, (is_safety, is_inference, is_telemetry)) rows
        covering every channel tick and every extra instant from zero
        through the horizon, in ascending order.
        """
        periods = self._periods
        schedule: dict[int, list[bool]] = {}
        for instant in extra_instants:
            schedule[instant] = [False, False, False]
        for channel, period in enumerate(periods):
            slot = 0
            while slot * period <= end_time_us:
                instant = slot * period
                flags = schedule.setdefault(instant, [False, False, False])
                flags[channel] = True
                slot += 1
        merged = sorted(schedule.items())
        return [(when, (flags[0], flags[1], flags[2])) for when, flags in merged]

    def run(self, signal_events: tuple[SignalEvent, ...], end_time_us: int) -> FabricRun:
        """Emit the fabric stream from time zero through ``end_time_us``.

        The input list must be timestamp ordered with unique timestamps.
        A signal event sharing a tick's timestamp is folded in before the
        tick fires, so a meaningful event at exactly an inference slot
        activates that slot's inference cycle.
        """
        self._validate_run_inputs(signal_events, end_time_us)

        activation_by_time: dict[int, bool] = {}
        for event in signal_events:
            activation_by_time[event.timestamp_us] = event.meaningful

        stream: list[FabricEvent] = []
        counts = [0, 0, 0]
        queued_request = False

        grid = self._grid(end_time_us, set(activation_by_time))
        for instant, (is_safety, is_inference, is_telemetry) in grid:
            activation = activation_by_time.get(instant, None)
            if activation is not None:
                queued_request = queued_request or activation

            if is_safety:
                stream.append(FabricEvent(instant, FabricEventKind.SAFETY_TICK))
                counts[0] += 1

            if is_inference and queued_request:
                stream.append(FabricEvent(instant, FabricEventKind.INFERENCE))
                counts[1] += 1
                queued_request = False

            if is_telemetry:
                stream.append(FabricEvent(instant, FabricEventKind.TELEMETRY_TICK))
                counts[2] += 1

        expected_safety = end_time_us // self._periods[0] + 1
        missed = expected_safety - counts[0]
        if missed < 0:
            missed = 0
        summary = FabricRun(
            events=tuple(stream), safety_ticks=counts[0], inference_events=counts[1],
            telemetry_ticks=counts[2], missed_safety_deadlines=missed)
        return summary
