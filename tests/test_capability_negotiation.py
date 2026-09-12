"""Tests for signed capability manifests and conservative refusal paths."""

from __future__ import annotations

from dataclasses import replace

from limen.protocol import (
    CapabilityManifest,
    CapabilityNegotiator,
    NegotiationPolicy,
    SensorCapability,
)


SECRET = b"limen-test-key"


def manifest() -> CapabilityManifest:
    return CapabilityManifest(
        manifest_version="1.0",
        device_id="assistive_servo_01",
        device_type="knee_assist",
        protocol_version="limen-1",
        commands=("hold_position", "set_torque"),
        sensors=(
            SensorCapability("emg_primary", 1000.0),
            SensorCapability("imu_joint", 100.0),
        ),
        max_torque_nm=25.0,
        max_velocity_rad_s=3.0,
        fault_behaviors=("safe_halt", "hold_position"),
    )


def policy() -> NegotiationPolicy:
    return NegotiationPolicy(
        protocol_version="limen-1",
        required_commands=("hold_position", "set_torque"),
        minimum_sensor_rates_hz={"emg_primary": 500.0, "imu_joint": 50.0},
        required_fault_behaviors=("safe_halt", "hold_position"),
        required_torque_nm=20.0,
        required_velocity_rad_s=2.0,
        max_allowed_torque_nm=30.0,
        max_allowed_velocity_rad_s=4.0,
    )


def test_signed_manifest_is_accepted() -> None:
    result = CapabilityNegotiator(policy(), SECRET).negotiate(manifest().signed_mapping(SECRET))
    assert result.accepted
    assert result.device_id == "assistive_servo_01"
    assert result.reasons == ("accepted",)
    assert result.agreed_limits == {"max_torque_nm": 30.0, "max_velocity_rad_s": 4.0}


def test_tampering_is_refused_before_capability_checks() -> None:
    raw = manifest().signed_mapping(SECRET)
    raw["max_torque_nm"] = 29.0
    result = CapabilityNegotiator(policy(), SECRET).negotiate(raw)
    assert not result.accepted
    assert result.device_id is None
    assert result.reasons[0].startswith("manifest_invalid:manifest signature")


def test_missing_command_and_sensor_are_refused_explicitly() -> None:
    altered = replace(
        manifest(),
        commands=("hold_position",),
        sensors=(SensorCapability("emg_primary", 100.0),),
    )
    result = CapabilityNegotiator(policy(), SECRET).negotiate(altered.signed_mapping(SECRET))
    assert not result.accepted
    assert "missing_commands:set_torque" in result.reasons
    assert "sensor_rate_below_minimum:emg_primary" in result.reasons
    assert "missing_sensor:imu_joint" in result.reasons


def test_over_limit_device_is_refused_even_when_it_has_required_capability() -> None:
    altered = replace(manifest(), max_torque_nm=35.0)
    result = CapabilityNegotiator(policy(), SECRET).negotiate(altered.signed_mapping(SECRET))
    assert not result.accepted
    assert result.reasons == ("actuator_torque_exceeds_safety_ceiling",)


def test_negotiation_reasons_are_stable() -> None:
    raw = manifest().signed_mapping(SECRET)
    first = CapabilityNegotiator(policy(), SECRET).negotiate(raw)
    second = CapabilityNegotiator(policy(), SECRET).negotiate(raw)
    assert first == second
