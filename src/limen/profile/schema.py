"""Partner device profile model and strict input gate.

A profile declares everything Limen needs to trust a partner device: its
sensing channels, the intents it distinguishes, the physical envelope of its
actuator, and the deterministic response to each sensor loss. Validation is
table driven: every field carries a declared rule, and the gate rejects both
absent and unlisted keys rather than guessing intent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields as dataclass_fields
from pathlib import Path
from typing import Any, Callable, Mapping

_POSITIVE_INFINITY = float("inf")


def _finite(magnitude: float) -> bool:
    return magnitude == magnitude and abs(magnitude) != _POSITIVE_INFINITY


class ProfileValidationError(ValueError):
    """Raised when a partner profile cannot be safely compiled."""


_NAME_RULE = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_VALID_SENSOR_KINDS = frozenset({"emg", "imu", "pressure", "machine_state"})
_VALID_FAULT_RESPONSES = frozenset({"safe_halt", "hold_position", "zero_output"})
_SAMPLE_RATE_CEILING_HZ = 1_000_000.0
_CHANNEL_FLOOR = 1
_CHANNEL_CEILING = 64
_INTENT_CEILING = 64


def _reject(message: str) -> None:
    raise ProfileValidationError(message)


def _object_guard(node: Any, where: str) -> Mapping[str, Any]:
    if isinstance(node, Mapping):
        return node
    _reject(f"{where} must be an object")
    raise AssertionError("unreachable")


def _key_guard(node: Mapping[str, Any], allowed: set[str], where: str) -> None:
    absent = sorted(allowed.difference(node))
    surplus = sorted(set(node).difference(allowed))
    if absent:
        _reject(f"{where} missing required fields: {', '.join(absent)}")
    if surplus:
        _reject(f"{where} has unknown fields: {', '.join(surplus)}")


def _text_guard(node: Any, where: str, rule: re.Pattern[str] | None = None) -> str:
    if not isinstance(node, str) or not node:
        _reject(f"{where} must be a non-empty string")
    if rule is not None and rule.fullmatch(node) is None:
        _reject(f"{where} has invalid identifier {node!r}")
    return node


def _numeric_guard(node: Any, where: str, floor: float | None, open_floor: bool) -> float:
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        _reject(f"{where} must be a number")
    magnitude = float(node)
    if floor is None:
        if not _finite(magnitude):
            _reject(f"{where} must be finite")
        return magnitude
    wording = "greater than zero" if open_floor else "non-negative"
    out_of_range = magnitude <= floor if open_floor else magnitude < floor
    if not _finite(magnitude) or out_of_range:
        _reject(f"{where} must be finite and {wording}")
    return magnitude


# Field-rule tables: each row binds one declared key to its guard. Extending a
# section means adding one row and one dataclass attribute, nothing else.
_Rule = tuple[str, Callable[[Any, str], Any]]


def _run_rules(node: Mapping[str, Any], rules: tuple[_Rule, ...], where: str) -> dict[str, Any]:
    _key_guard(node, {key for key, _ in rules}, where)
    outcome: dict[str, Any] = {}
    for key, guard in rules:
        label = f"{where}.{key}" if where else key
        outcome[key] = guard(node[key], label)
    return outcome


@dataclass(frozen=True)
class SensorSpec:
    """One sensing channel of the partner device."""

    name: str
    sensor_type: str
    channels: int
    sample_rate_hz: float
    noise_threshold: float

    @classmethod
    def from_mapping(cls, raw: Any, slot: int) -> "SensorSpec":
        where = f"sensors[{slot}]"
        node = _object_guard(raw, where)
        checked = _run_rules(
            node,
            (
                ("name", lambda raw, at: _text_guard(raw, at, _NAME_RULE)),
                ("type", _text_guard),
                ("channels", lambda raw, at: raw),
                ("sample_rate_hz", lambda raw, at: _numeric_guard(raw, at, 0.0, True)),
                ("noise_threshold", lambda raw, at: _numeric_guard(raw, at, 0.0, False)),
            ),
            where,
        )

        kind = checked["type"]
        if kind not in _VALID_SENSOR_KINDS:
            _reject(f"{where}.type {kind!r} is not one of {sorted(_VALID_SENSOR_KINDS)}")

        count = checked["channels"]
        invalid_count = (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not _CHANNEL_FLOOR <= count <= _CHANNEL_CEILING
        )
        if invalid_count:
            _reject(f"{where}.channels must be an integer from 1 to 64")

        if checked["sample_rate_hz"] > _SAMPLE_RATE_CEILING_HZ:
            _reject(f"{where}.sample_rate_hz exceeds supported limit")

        return cls(checked["name"], kind, count, checked["sample_rate_hz"], checked["noise_threshold"])

    def to_mapping(self) -> dict[str, Any]:
        ordered = (
            ("name", self.name),
            ("type", self.sensor_type),
            ("channels", self.channels),
            ("sample_rate_hz", self.sample_rate_hz),
            ("noise_threshold", self.noise_threshold),
        )
        return dict(ordered)


@dataclass(frozen=True)
class ActuatorSpec:
    """Physical envelope the partner actuator must never exceed."""

    name: str
    max_torque_nm: float
    max_velocity_rad_s: float
    position_min_rad: float
    position_max_rad: float

    @classmethod
    def from_mapping(cls, raw: Any) -> "ActuatorSpec":
        node = _object_guard(raw, "actuator")
        checked = _run_rules(
            node,
            (
                ("name", lambda raw, at: _text_guard(raw, at, _NAME_RULE)),
                ("max_torque_nm", lambda raw, at: _numeric_guard(raw, at, 0.0, True)),
                ("max_velocity_rad_s", lambda raw, at: _numeric_guard(raw, at, 0.0, True)),
                ("position_min_rad", lambda raw, at: _numeric_guard(raw, at, None, False)),
                ("position_max_rad", lambda raw, at: _numeric_guard(raw, at, None, False)),
            ),
            where="actuator",
        )

        if checked["position_min_rad"] >= checked["position_max_rad"]:
            _reject("actuator.position_min_rad must be less than position_max_rad")

        return cls(checked["name"], checked["max_torque_nm"], checked["max_velocity_rad_s"], checked["position_min_rad"], checked["position_max_rad"])

    def to_mapping(self) -> dict[str, Any]:
        ordered = (
            ("name", self.name),
            ("max_torque_nm", self.max_torque_nm),
            ("max_velocity_rad_s", self.max_velocity_rad_s),
            ("position_min_rad", self.position_min_rad),
            ("position_max_rad", self.position_max_rad),
        )
        return dict(ordered)


@dataclass(frozen=True)
class FaultPath:
    """Predetermined device response when one sensor is lost."""

    sensor: str
    action: str

    @classmethod
    def from_mapping(cls, raw: Any, slot: int) -> "FaultPath":
        where = f"fault_paths[{slot}]"
        node = _object_guard(raw, where)
        checked = _run_rules(
            node,
            (
                ("sensor", lambda raw, at: _text_guard(raw, at, _NAME_RULE)),
                ("action", _text_guard),
            ),
            where,
        )

        response = checked["action"]
        if response not in _VALID_FAULT_RESPONSES:
            _reject(
                f"{where}.action {response!r} is not one of {sorted(_VALID_FAULT_RESPONSES)}"
            )
        return cls(checked["sensor"], response)

    def to_mapping(self) -> dict[str, str]:
        return dict((("sensor", self.sensor), ("action", self.action)))


def _text_array(node: Any, where: str) -> tuple[str, ...]:
    if not isinstance(node, list) or not node:
        _reject(f"{where} must be a non-empty array")
    return tuple(_text_guard(item, f"{where}[{slot}]") for slot, item in enumerate(node))


def _has_repeat(items: tuple[str, ...]) -> bool:
    return len(set(items)) != len(items)


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
    def from_mapping(cls, raw: Any) -> "PartnerProfile":
        node = _object_guard(raw, "profile")
        declared = {entry.name for entry in dataclass_fields(cls)}
        _key_guard(node, declared, "profile")

        version = _text_guard(node["schema_version"], "schema_version")
        if version != "1.0":
            _reject(f"unsupported schema_version {version!r}; expected '1.0'")
        identifier = _text_guard(node["profile_id"], "profile_id", _NAME_RULE)

        sensor_block = node["sensors"]
        if not isinstance(sensor_block, list) or not sensor_block:
            _reject("sensors must be a non-empty array")
        sensors = tuple(
            SensorSpec.from_mapping(entry, slot) for slot, entry in enumerate(sensor_block)
        )
        names = tuple(unit.name for unit in sensors)
        if _has_repeat(names):
            _reject("sensor names must be unique")

        intents = _text_array(node["intent_classes"], "intent_classes")
        if _has_repeat(intents):
            _reject("intent_classes must be unique")
        if len(intents) > _INTENT_CEILING:
            _reject("intent_classes cannot contain more than 64 classes")

        actuator = ActuatorSpec.from_mapping(node["actuator"])

        fault_block = node["fault_paths"]
        if not isinstance(fault_block, list) or not fault_block:
            _reject("fault_paths must be a non-empty array")
        fault_paths = tuple(
            FaultPath.from_mapping(entry, slot) for slot, entry in enumerate(fault_block)
        )
        covered = tuple(route.sensor for route in fault_paths)
        if _has_repeat(covered):
            _reject("fault_paths must contain one path per sensor")
        if set(covered) != set(names):
            gaps = sorted(set(names).difference(covered))
            stray = sorted(set(covered).difference(names))
            parts = []
            if gaps:
                parts.append(f"missing {', '.join(gaps)}")
            if stray:
                parts.append(f"unknown {', '.join(stray)}")
            _reject("fault_paths do not cover sensors: " + "; ".join(parts))

        return cls(version, identifier, sensors, intents, actuator, fault_paths)

    @classmethod
    def from_json_file(cls, source: Path) -> "PartnerProfile":
        try:
            text = source.read_text(encoding="utf-8")
        except OSError as failure:
            raise FileNotFoundError(f"unable to read profile {source}: {failure}") from failure
        try:
            document = json.JSONDecoder().decode(text)
        except json.JSONDecodeError as failure:
            raise ProfileValidationError(f"invalid JSON in {source}: {failure}") from failure
        return cls.from_mapping(document)

    def to_mapping(self) -> dict[str, Any]:
        pairs = (
            ("schema_version", self.schema_version),
            ("profile_id", self.profile_id),
            ("sensors", [unit.to_mapping() for unit in self.sensors]),
            ("intent_classes", list(self.intent_classes)),
            ("actuator", self.actuator.to_mapping()),
            ("fault_paths", [route.to_mapping() for route in self.fault_paths]),
        )
        return dict(pairs)
