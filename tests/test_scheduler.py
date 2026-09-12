"""Tests for the eventized multi-rate control scheduler."""

from __future__ import annotations

import pytest

from limen.scheduler import EventizedMultiRateFabric, FabricEventKind, SignalEvent


def test_safety_ticks_are_continuous_without_inference_events() -> None:
    run = EventizedMultiRateFabric().run((), 20_000)
    assert run.safety_ticks == 21
    assert run.inference_events == 0
    assert run.telemetry_ticks == 2
    assert run.missed_safety_deadlines == 0
    assert all(event.kind is not FabricEventKind.INFERENCE for event in run.events)


def test_meaningful_signal_activates_one_due_inference_cycle() -> None:
    run = EventizedMultiRateFabric().run(
        (SignalEvent(3_500, True), SignalEvent(7_000, False)),
        20_000,
    )
    assert run.inference_events == 1
    inference = [event for event in run.events if event.kind is FabricEventKind.INFERENCE]
    assert [event.timestamp_us for event in inference] == [20_000]
    assert run.safety_ticks == 21
    assert run.missed_safety_deadlines == 0


def test_same_timestamp_signal_is_consumed_before_inference() -> None:
    run = EventizedMultiRateFabric().run((SignalEvent(20_000, True),), 20_000)
    at_boundary = [event.kind for event in run.events if event.timestamp_us == 20_000]
    assert at_boundary == [
        FabricEventKind.SAFETY_TICK,
        FabricEventKind.INFERENCE,
        FabricEventKind.TELEMETRY_TICK,
    ]


def test_scheduler_is_deterministic_and_rejects_unordered_input() -> None:
    fabric = EventizedMultiRateFabric(safety_period_us=500, inference_period_us=2_000)
    events = (SignalEvent(100, True), SignalEvent(1_200, True))
    assert fabric.run(events, 4_000) == fabric.run(events, 4_000)
    with pytest.raises(ValueError, match="ordered"):
        fabric.run((SignalEvent(1_000, True), SignalEvent(0, True)), 2_000)
