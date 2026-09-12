"""Strict Limen partner-profile schema and validation.

The profile is the single source of truth for a partner device's sensors,
intent classes, actuator limits, and fault paths. The parser rejects unknown
fields and physically incoherent values instead of silently applying defaults.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ProfileValidationError(ValueError):
    """Raised when a partner profile cannot be safely compiled."""


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_SENSOR_TYPES = frozenset({"emg", "imu", "pressure", "machine_state"})
_FALLBACK_ACTIONS = frozenset({"safe_halt", "hold_position", "zero_output"})


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileValidationError(f"{context} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(value))
    extra = sorted(set(value) - required)
    if missing:
        raise ProfileValidationError(f"{context} missing required fields: {', '.join(missing)}")
    if extra:
        raise ProfileValidationError(f"{context} has unknown fields: {', '.join(extra)}")


def _string(value: Any, context: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise ProfileValidationError(f"{context} must be a non-empty string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ProfileValidationError(f"{context} has invalid identifier {value!r}")
    return value


def _positive_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ProfileValidationError(f"{context} must be finite and greater than zero")
    return result


def _nonnegative_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileValidationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ProfileValidationError(f"{context} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class SensorSpec:
    """One sensor declaration in a partner profile."""

    name: str
    sensor_type: str
    channels: int
    sample_rate_hz: float
    noise_threshold: float

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "SensorSpec":
        data = _mapping(value, f"sensors[{index}]")
        _exact_keys(
            data,
            {"name", "type", "channels", "sample_rate_hz", "noise_threshold"},
            f"sensors[{index}]",
        )
        name = _string(data["name"], f"sensors[{index}].name", _IDENTIFIER)
        sensor_type = _string(data["type"], f"sensors[{index}].type")
        if sensor_type not in _SENSOR_TYPES:
            raise ProfileValidationError(
                f"sensors[{index}].type {sensor_type!r} is not one of {sorted(_SENSOR_TYPES)}"
            )
        channels = data["channels"]
        if isinstance(channels, bool) or not isinstance(channels, int) or not 1 <= channels <= 64:
            raise ProfileValidationError(
                f"sensors[{index}].channels must be an integer from 1 to 64"
            )
        rate = _positive_number(data["sample_rate_hz"], f"sensors[{index}].sample_rate_hz")
        noise = _nonnegative_number(data["noise_threshold"], f"sensors[{index}].noise_threshold")
        if rate > 1_000_000.0:
            raise ProfileValidationError(
                f"sensors[{index}].sample_rate_hz exceeds supported limit"
            )
        return cls(
            name=name,
            sensor_type=sensor_type,
            channels=channels,
            sample_rate_hz=rate,
            noise_threshold=noise,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.sensor_type,
            "channels": self.channels,
            "sample_rate_hz": self.sample_rate_hz,
            "noise_threshold": self.noise_threshold,
        }


@dataclass(frozen=True)
class ActuatorSpec:
    """Hard physical limits for the partner actuator."""

    name: str
    max_torque_nm: float
    max_velocity_rad_s: float
    position_min_rad: float
    position_max_rad: float

    @classmethod
    def from_mapping(cls, value: Any) -> "ActuatorSpec":
        data = _mapping(value, "actuator")
        _exact_keys(
            data,
            {"name", "max_torque_nm", "max_velocity_rad_s", "position_min_rad", "position_max_rad"},
            "actuator",
        )
        name = _string(data["name"], "actuator.name", _IDENTIFIER)
        max_torque = _positive_number(data["max_torque_nm"], "actuator.max_torque_nm")
        max_velocity = _positive_number(data["max_velocity_rad_s"], "actuator.max_velocity_rad_s")
        lower_raw = data["position_min_rad"]
        if isinstance(lower_raw, bool) or not isinstance(lower_raw, (int, float)):
            raise ProfileValidationError("actuator.position_min_rad must be a number")
        lower = float(lower_raw)
        if not math.isfinite(lower):
            raise ProfileValidationError("actuator.position_min_rad must be finite")
        position_max_raw = data["position_max_rad"]
        if isinstance(position_max_raw, bool) or not isinstance(position_max_raw, (int, float)):
            raise ProfileValidationError("actuator.position_max_rad must be a number")
        position_max = float(position_max_raw)
        if not math.isfinite(lower) or lower >= position_max:
            raise ProfileValidationError(
                "actuator.position_min_rad must be less than position_max_rad"
            )
        return cls(
            name=name,
            max_torque_nm=max_torque,
            max_velocity_rad_s=max_velocity,
            position_min_rad=lower,
            position_max_rad=position_max,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "max_torque_nm": self.max_torque_nm,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "position_min_rad": self.position_min_rad,
            "position_max_rad": self.position_max_rad,
        }


@dataclass(frozen=True)
class FaultPath:
    """Deterministic fallback action for one sensor failure."""

    sensor: str
    action: str

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "FaultPath":
        data = _mapping(value, f"fault_paths[{index}]")
        _exact_keys(data, {"sensor", "action"}, f"fault_paths[{index}]")
        sensor = _string(data["sensor"], f"fault_paths[{index}].sensor", _IDENTIFIER)
        action = _string(data["action"], f"fault_paths[{index}].action")
        if action not in _FALLBACK_ACTIONS:
            raise ProfileValidationError(
                f"fault_paths[{index}].action {action!r} is not one of {sorted(_FALLBACK_ACTIONS)}"
            )
        return cls(sensor=sensor, action=action)

    def to_mapping(self) -> dict[str, str]:
        return {"sensor": self.sensor, "action": self.action}


@dataclass(frozen=True)
class PartnerProfile:
    """Validated, immutable partner profile."""

    schema_version: str
    profile_id: str
    sensors: tuple[SensorSpec, ...]
    intent_classes: tuple[str, ...]
    actuator: ActuatorSpec
    fault_paths: tuple[FaultPath, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "PartnerProfile":
        data = _mapping(value, "profile")
        _exact_keys(
            data,
            {"schema_version", "profile_id", "sensors", "intent_classes", "actuator", "fault_paths"},
            "profile",
        )
        schema_version = _string(data["schema_version"], "schema_version")
        if schema_version != "1.0":
            raise ProfileValidationError(
                f"unsupported schema_version {schema_version!r}; expected '1.0'"
            )
        profile_id = _string(data["profile_id"], "profile_id", _IDENTIFIER)

        raw_sensors = data["sensors"]
        if not isinstance(raw_sensors, list) or not raw_sensors:
            raise ProfileValidationError("sensors must be a non-empty array")
        sensors = tuple(
            SensorSpec.from_mapping(item, index) for index, item in enumerate(raw_sensors)
        )
        sensor_names = [sensor.name for sensor in sensors]
        if len(set(sensor_names)) != len(sensor_names):
            raise ProfileValidationError("sensor names must be unique")

        raw_intents = data["intent_classes"]
        if not isinstance(raw_intents, list) or not raw_intents:
            raise ProfileValidationError("intent_classes must be a non-empty array")
        intents = tuple(
            _string(item, f"intent_classes[{index}]") for index, item in enumerate(raw_intents)
        )
        if len(set(intents)) != len(intents):
            raise ProfileValidationError("intent_classes must be unique")
        if len(intents) > 64:
            raise ProfileValidationError("intent_classes cannot contain more than 64 classes")

        actuator = ActuatorSpec.from_mapping(data["actuator"])
        raw_faults = data["fault_paths"]
        if not isinstance(raw_faults, list) or not raw_faults:
            raise ProfileValidationError("fault_paths must be a non-empty array")
        fault_paths = tuple(
            FaultPath.from_mapping(item, index) for index, item in enumerate(raw_faults)
        )
        fault_sensors = [path.sensor for path in fault_paths]
        if len(set(fault_sensors)) != len(fault_sensors):
            raise ProfileValidationError("fault_paths must contain one path per sensor")
        if set(fault_sensors) != set(sensor_names):
            missing = sorted(set(sensor_names) - set(fault_sensors))
            unknown = sorted(set(fault_sensors) - set(sensor_names))
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unknown:
                details.append(f"unknown {', '.join(unknown)}")
            raise ProfileValidationError(
                "fault_paths do not cover sensors: " + "; ".join(details)
            )

        return cls(
            schema_version=schema_version,
            profile_id=profile_id,
            sensors=sensors,
            intent_classes=intents,
            actuator=actuator,
            fault_paths=fault_paths,
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "PartnerProfile":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise FileNotFoundError(f"unable to read profile {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ProfileValidationError(f"invalid JSON in {path}: {exc}") from exc
        return cls.from_mapping(raw)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "sensors": [sensor.to_mapping() for sensor in self.sensors],
            "intent_classes": list(self.intent_classes),
            "actuator": self.actuator.to_mapping(),
            "fault_paths": [path.to_mapping() for path in self.fault_paths],
        }
