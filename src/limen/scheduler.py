"""Deterministic eventized multi-rate control scheduler.

Keeps safety supervision on a continuous mandatory period while allowing
higher-cost inference to run only after meaningful signal events. This is
a virtual scheduler for replay and HIL planning; it does not claim real-time
guarantees on a general-purpose operating system.

The scheduler has no wall-clock calls. Given the same signal events and end
time, it emits the same ordered event stream:

    signal ingestion -> safety tick -> inference (if pending) -> telemetry

Safety is always scheduled at its configured period. Inference is eventized:
no meaningful event means no inference event.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FabricEventKind(enum.Enum):
    SAFETY_TICK = "safety_tick"
    INFERENCE = "inference"
    TELEMETRY_TICK = "telemetry_tick"


@dataclass(frozen=True)
class SignalEvent:
    """An input event observed by the scheduler."""

    timestamp_us: int
    meaningful: bool

    def __post_init__(self) -> None:
        if self.timestamp_us < 0:
            raise ValueError("signal timestamp_us must be non-negative")


@dataclass(frozen=True)
class FabricEvent:
    """One scheduled control-fabric event."""

    timestamp_us: int
    kind: FabricEventKind


@dataclass(frozen=True)
class FabricRun:
    """Scheduler output and deterministic accounting."""

    events: tuple[FabricEvent, ...]
    safety_ticks: int
    inference_events: int
    telemetry_ticks: int
    missed_safety_deadlines: int


class EventizedMultiRateFabric:
    """Virtual multi-rate scheduler with mandatory safety ticks."""

    def __init__(
        self,
        safety_period_us: int = 1_000,
        inference_period_us: int = 20_000,
        telemetry_period_us: int = 20_000,
    ) -> None:
        periods = {
            "safety_period_us": safety_period_us,
            "inference_period_us": inference_period_us,
            "telemetry_period_us": telemetry_period_us,
        }
        for name, value in periods.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if safety_period_us > inference_period_us:
            raise ValueError("safety period must not be slower than inference period")
        self._safety_period_us = safety_period_us
        self._inference_period_us = inference_period_us
        self._telemetry_period_us = telemetry_period_us

    def run(self, signal_events: tuple[SignalEvent, ...], end_time_us: int) -> FabricRun:
        """Run from time zero through ``end_time_us`` inclusive.

        Input events must be timestamp ordered. Same-timestamp events are
        consumed before scheduled work so a meaningful event can activate an
        inference cycle due at that exact timestamp.
        """
        if not isinstance(end_time_us, int) or end_time_us < 0:
            raise ValueError("end_time_us must be a non-negative integer")
        if any(not isinstance(event, SignalEvent) for event in signal_events):
            raise TypeError("signal_events must contain only SignalEvent values")
        for previous, current in zip(signal_events, signal_events[1:]):
            if current.timestamp_us < previous.timestamp_us:
                raise ValueError("signal_events must be ordered by timestamp")
        if any(event.timestamp_us > end_time_us for event in signal_events):
            raise ValueError("signal event occurs after end_time_us")

        next_safety = 0
        next_inference = 0
        next_telemetry = 0
        input_index = 0
        pending_inference = False
        events: list[FabricEvent] = []

        while True:
            candidates = [next_safety, next_inference, next_telemetry]
            if input_index < len(signal_events):
                candidates.append(signal_events[input_index].timestamp_us)
            timestamp = min(candidates)
            if timestamp > end_time_us:
                break

            while (
                input_index < len(signal_events)
                and signal_events[input_index].timestamp_us == timestamp
            ):
                if signal_events[input_index].meaningful:
                    pending_inference = True
                input_index += 1

            if next_safety == timestamp:
                events.append(FabricEvent(timestamp, FabricEventKind.SAFETY_TICK))
                next_safety += self._safety_period_us

            if next_inference == timestamp:
                if pending_inference:
                    events.append(FabricEvent(timestamp, FabricEventKind.INFERENCE))
                    pending_inference = False
                next_inference += self._inference_period_us

            if next_telemetry == timestamp:
                events.append(FabricEvent(timestamp, FabricEventKind.TELEMETRY_TICK))
                next_telemetry += self._telemetry_period_us

        safety_ticks = sum(event.kind is FabricEventKind.SAFETY_TICK for event in events)
        inference_events = sum(event.kind is FabricEventKind.INFERENCE for event in events)
        telemetry_ticks = sum(event.kind is FabricEventKind.TELEMETRY_TICK for event in events)
        expected_safety = end_time_us // self._safety_period_us + 1
        return FabricRun(
            events=tuple(events),
            safety_ticks=safety_ticks,
            inference_events=inference_events,
            telemetry_ticks=telemetry_ticks,
            missed_safety_deadlines=max(0, expected_safety - safety_ticks),
        )
