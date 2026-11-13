"""Limen device capability exchange with keyed-manifest integrity checks.

Every device presents a manifest describing what it can do. The manifest
travels with an HMAC-SHA256 tag over a canonical serialization so that lab
integrations detect tampering without a full identity stack. The admission
pipeline evaluates a verified manifest against a host policy and either
admits the device with agreed ceilings or refuses it with deterministic,
machine-readable grounds.
"""

from __future__ import annotations

import hashlib, hmac, json, math, re
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Callable, Mapping


class ManifestValidationError(ValueError):
    """Raised when a capability manifest is malformed."""


_NAME_RULE = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
_TAG_RULE = re.compile(r"[0-9a-f]{64}")


def _fail(problem: str) -> None:
    raise ManifestValidationError(problem)


def _require_mapping(item: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(item, Mapping):
        _fail(f"{where} must be an object")
    return item


def _require_exact_fields(item: Mapping[str, Any], wanted: set[str], where: str) -> None:
    absent = sorted(wanted - set(item))
    surplus = sorted(set(item) - wanted)
    if absent:
        _fail(f"{where} missing required fields: {', '.join(absent)}")
    if surplus:
        _fail(f"{where} has unknown fields: {', '.join(surplus)}")


def _require_name(item: Any, where: str) -> str:
    if not isinstance(item, str) or _NAME_RULE.fullmatch(item) is None:
        _fail(f"{where} must be a valid identifier")
    return item


def _require_quantity(item: Any, where: str) -> float:
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        _fail(f"{where} must be a number")
    amount = float(item)
    if not math.isfinite(amount) or amount <= 0.0:
        _fail(f"{where} must be finite and greater than zero")
    return amount


def _freeze_string_list(item: Any, where: str, unique_error: str) -> tuple[str, ...]:
    if not isinstance(item, list) or not item:
        _fail(f"{where} must be a non-empty array")
    frozen = tuple(
        _require_name(entry, f"{where}[{position}]")
        for position, entry in enumerate(item)
    )
    if len(set(frozen)) != len(frozen):
        _fail(unique_error)
    return frozen


def _serialize_stable(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _digest_payload(payload: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, _serialize_stable(payload), hashlib.sha256).hexdigest()


def _tags_match(first: str, second: str) -> bool:
    return hmac.compare_digest(first, second)


def _require_key(secret: bytes, purpose: str) -> None:
    if len(secret) == 0:
        raise ValueError(f"manifest {purpose} secret must not be empty")


def _manifest_field_names() -> set[str]:
    return {entry.name for entry in dataclass_fields(CapabilityManifest)} | {"signature"}


@dataclass(frozen=True)
class SensorCapability:
    name: str; sample_rate_hz: float

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "SensorCapability":
        where = f"sensors[{index}]"
        item = _require_mapping(value, where)
        _require_exact_fields(item, {"name", "sample_rate_hz"}, where)
        parsed_name = _require_name(item["name"], f"{where}.name")
        parsed_rate = _require_quantity(item["sample_rate_hz"], f"{where}.sample_rate_hz")
        return cls(parsed_name, parsed_rate)

    def to_mapping(self) -> dict[str, Any]:
        return dict(zip(("name", "sample_rate_hz"), (self.name, self.sample_rate_hz)))


@dataclass(frozen=True)
class CapabilityManifest:
    """Validated device capability declaration."""

    manifest_version: str; device_id: str; device_type: str; protocol_version: str
    commands: tuple[str, ...]; sensors: tuple[SensorCapability, ...]
    max_torque_nm: float; max_velocity_rad_s: float
    fault_behaviors: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityManifest":
        item = _require_mapping(value, "manifest")
        _require_exact_fields(item, _manifest_field_names(), "manifest")
        cls._gate_schema(item)
        parsed = {
            "manifest_version": item["manifest_version"],
            "device_id": _require_name(item["device_id"], "device_id"),
            "device_type": _require_name(item["device_type"], "device_type"),
            "protocol_version": item["protocol_version"],
            "commands": _freeze_string_list(
                item["commands"], "commands", "commands must be unique"
            ),
            "sensors": cls._build_sensors(item["sensors"]),
            "max_torque_nm": _require_quantity(item["max_torque_nm"], "max_torque_nm"),
            "max_velocity_rad_s": _require_quantity(
                item["max_velocity_rad_s"], "max_velocity_rad_s"
            ),
            "fault_behaviors": _freeze_string_list(
                item["fault_behaviors"], "fault_behaviors", "fault_behaviors must be unique"
            ),
        }
        return cls(**parsed)

    @staticmethod
    def _gate_schema(item: Mapping[str, Any]) -> None:
        edition = item["manifest_version"]
        if edition != "1.0":
            _fail(f"unsupported manifest_version {edition!r}; expected '1.0'")
        dialect = item["protocol_version"]
        if not isinstance(dialect, str) or not dialect:
            _fail("protocol_version must be a non-empty string")
        tag = item["signature"]
        if not isinstance(tag, str) or _TAG_RULE.fullmatch(tag) is None:
            _fail("signature must be a lowercase HMAC-SHA256 hex string")

    @staticmethod
    def _build_sensors(raw: Any) -> tuple[SensorCapability, ...]:
        if not isinstance(raw, list) or not raw:
            _fail("sensors must be a non-empty array")
        built = tuple(
            SensorCapability.from_mapping(entry, position)
            for position, entry in enumerate(raw)
        )
        labels = [entry.name for entry in built]
        if len(set(labels)) != len(labels):
            _fail("sensor names must be unique")
        return built

    def unsigned_mapping(self) -> dict[str, Any]:
        envelope: dict[str, Any] = {}
        for entry in dataclass_fields(self):
            content = getattr(self, entry.name)
            if entry.name in ("commands", "fault_behaviors"):
                content = list(content)
            elif entry.name == "sensors":
                content = [unit.to_mapping() for unit in content]
            envelope[entry.name] = content
        return envelope

    def signed_mapping(self, secret: bytes) -> dict[str, Any]:
        _require_key(secret, "signing")
        envelope = self.unsigned_mapping()
        envelope.update(signature=_digest_payload(envelope, secret))
        return envelope

    @classmethod
    def from_signed_mapping(cls, value: Any, secret: bytes) -> "CapabilityManifest":
        _require_key(secret, "verification")
        item = _require_mapping(value, "manifest")
        decoded = cls.from_mapping(item)
        reference = _digest_payload(decoded.unsigned_mapping(), secret)
        if not _tags_match(reference, item["signature"]):
            _fail("manifest signature verification failed")
        return decoded


@dataclass(frozen=True)
class NegotiationPolicy:
    """Minimum and maximum requirements for one controller integration."""

    protocol_version: str; required_commands: tuple[str, ...]
    minimum_sensor_rates_hz: Mapping[str, float]
    required_fault_behaviors: tuple[str, ...]
    required_torque_nm: float; max_allowed_torque_nm: float
    required_velocity_rad_s: float; max_allowed_velocity_rad_s: float

    def __post_init__(self) -> None:
        if len(self.protocol_version) == 0:
            raise ValueError("protocol_version must not be empty")
        figures = (
            ("required_torque_nm", self.required_torque_nm),
            ("required_velocity_rad_s", self.required_velocity_rad_s),
            ("max_allowed_torque_nm", self.max_allowed_torque_nm),
            ("max_allowed_velocity_rad_s", self.max_allowed_velocity_rad_s),
        )
        for label, figure in figures:
            if not math.isfinite(figure):
                raise ValueError(f"{label} must be finite")
            if figure <= 0:
                raise ValueError(f"{label} must be positive")
        torque_ok = self.required_torque_nm <= self.max_allowed_torque_nm
        velocity_ok = self.required_velocity_rad_s <= self.max_allowed_velocity_rad_s
        if not torque_ok:
            raise ValueError("required torque exceeds policy maximum")
        if not velocity_ok:
            raise ValueError("required velocity exceeds policy maximum")


@dataclass(frozen=True)
class NegotiationResult:
    accepted: bool; device_id: str | None; reasons: tuple[str, ...]
    agreed_limits: Mapping[str, float] | None = None


def _assess_dialect(manifest: CapabilityManifest, policy: NegotiationPolicy) -> list[str]:
    if manifest.protocol_version != policy.protocol_version:
        return ["protocol_version_mismatch"]
    return []


def _assess_commands(manifest: CapabilityManifest, policy: NegotiationPolicy) -> list[str]:
    deficit = sorted(set(policy.required_commands) - set(manifest.commands))
    if deficit:
        return ["missing_commands:" + ",".join(deficit)]
    return []


def _assess_sensors(manifest: CapabilityManifest, policy: NegotiationPolicy) -> list[str]:
    findings: list[str] = []
    offered = {entry.name: entry.sample_rate_hz for entry in manifest.sensors}
    for label, floor in sorted(policy.minimum_sensor_rates_hz.items()):
        if label not in offered:
            findings.append(f"missing_sensor:{label}")
        elif offered[label] < floor:
            findings.append(f"sensor_rate_below_minimum:{label}")
    return findings


def _assess_actuators(manifest: CapabilityManifest, policy: NegotiationPolicy) -> list[str]:
    findings: list[str] = []
    pairs = (
        (
            manifest.max_torque_nm,
            policy.required_torque_nm,
            policy.max_allowed_torque_nm,
            "torque",
        ),
        (
            manifest.max_velocity_rad_s,
            policy.required_velocity_rad_s,
            policy.max_allowed_velocity_rad_s,
            "velocity",
        ),
    )
    for offered, floor, ceiling, axis in pairs:
        if offered < floor:
            findings.append(f"actuator_{axis}_below_required")
        if offered > ceiling:
            findings.append(f"actuator_{axis}_exceeds_safety_ceiling")
    return findings


def _assess_faults(manifest: CapabilityManifest, policy: NegotiationPolicy) -> list[str]:
    deficit = sorted(set(policy.required_fault_behaviors) - set(manifest.fault_behaviors))
    if deficit:
        return ["missing_fault_behaviors:" + ",".join(deficit)]
    return []


_EVALUATORS: tuple[
    Callable[[CapabilityManifest, NegotiationPolicy], list[str]], ...
] = (
    _assess_dialect,
    _assess_commands,
    _assess_sensors,
    _assess_actuators,
    _assess_faults,
)


class CapabilityNegotiator:
    """Verify and admit signed manifests through a staged evaluation pipeline."""

    def __init__(self, policy: NegotiationPolicy, secret: bytes) -> None:
        if len(secret) == 0:
            raise ValueError("negotiation secret must not be empty")
        self._secret = secret
        self._policy = policy

    def negotiate(self, raw_manifest: Any) -> NegotiationResult:
        try:
            offer = CapabilityManifest.from_signed_mapping(raw_manifest, self._secret)
        except (ManifestValidationError, ValueError, TypeError) as problem:
            return NegotiationResult(False, None, (f"manifest_invalid:{problem}",))

        verdicts: list[str] = []
        for stage in _EVALUATORS:
            verdicts.extend(stage(offer, self._policy))

        if verdicts:
            return NegotiationResult(False, offer.device_id, tuple(verdicts))
        ceilings = dict(
            max_torque_nm=self._policy.max_allowed_torque_nm,
            max_velocity_rad_s=self._policy.max_allowed_velocity_rad_s,
        )
        return NegotiationResult(True, offer.device_id, ("accepted",), ceilings)
