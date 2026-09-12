"""Signed capability manifests and conservative Limen negotiation protocol.

Uses HMAC-SHA256 with a pre-shared key so that the manifest cannot be
changed silently in a lab integration. This is not a replacement for device
identity, key provisioning, or a production PKI. Negotiation refuses
unknown, unsigned, under-capable, or over-limit devices with stable reasons.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


class ManifestValidationError(ValueError):
    """Raised when a capability manifest is malformed."""


_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{context} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing:
        raise ManifestValidationError(
            f"{context} missing required fields: {', '.join(missing)}"
        )
    if extra:
        raise ManifestValidationError(f"{context} has unknown fields: {', '.join(extra)}")


def _identifier(value: Any, context: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ManifestValidationError(f"{context} must be a valid identifier")
    return value


def _positive(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestValidationError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ManifestValidationError(f"{context} must be finite and greater than zero")
    return number


@dataclass(frozen=True)
class SensorCapability:
    name: str
    sample_rate_hz: float

    @classmethod
    def from_mapping(cls, value: Any, index: int) -> "SensorCapability":
        data = _object(value, f"sensors[{index}]")
        _keys(data, {"name", "sample_rate_hz"}, f"sensors[{index}]")
        return cls(
            name=_identifier(data["name"], f"sensors[{index}].name"),
            sample_rate_hz=_positive(data["sample_rate_hz"], f"sensors[{index}].sample_rate_hz"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"name": self.name, "sample_rate_hz": self.sample_rate_hz}


@dataclass(frozen=True)
class CapabilityManifest:
    """Validated device capability declaration."""

    manifest_version: str
    device_id: str
    device_type: str
    protocol_version: str
    commands: tuple[str, ...]
    sensors: tuple[SensorCapability, ...]
    max_torque_nm: float
    max_velocity_rad_s: float
    fault_behaviors: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Any) -> "CapabilityManifest":
        data = _object(value, "manifest")
        _keys(
            data,
            {
                "manifest_version",
                "device_id",
                "device_type",
                "protocol_version",
                "commands",
                "sensors",
                "max_torque_nm",
                "max_velocity_rad_s",
                "fault_behaviors",
                "signature",
            },
            "manifest",
        )
        version = data["manifest_version"]
        protocol = data["protocol_version"]
        if version != "1.0":
            raise ManifestValidationError(
                f"unsupported manifest_version {version!r}; expected '1.0'"
            )
        if not isinstance(protocol, str) or not protocol:
            raise ManifestValidationError("protocol_version must be a non-empty string")
        device_id = _identifier(data["device_id"], "device_id")
        device_type = _identifier(data["device_type"], "device_type")

        commands_raw = data["commands"]
        if not isinstance(commands_raw, list) or not commands_raw:
            raise ManifestValidationError("commands must be a non-empty array")
        commands = tuple(
            _identifier(item, f"commands[{index}]") for index, item in enumerate(commands_raw)
        )
        if len(set(commands)) != len(commands):
            raise ManifestValidationError("commands must be unique")

        sensors_raw = data["sensors"]
        if not isinstance(sensors_raw, list) or not sensors_raw:
            raise ManifestValidationError("sensors must be a non-empty array")
        sensors = tuple(
            SensorCapability.from_mapping(item, index) for index, item in enumerate(sensors_raw)
        )
        sensor_names = [sensor.name for sensor in sensors]
        if len(set(sensor_names)) != len(sensor_names):
            raise ManifestValidationError("sensor names must be unique")

        faults_raw = data["fault_behaviors"]
        if not isinstance(faults_raw, list) or not faults_raw:
            raise ManifestValidationError("fault_behaviors must be a non-empty array")
        faults = tuple(
            _identifier(item, f"fault_behaviors[{index}]")
            for index, item in enumerate(faults_raw)
        )
        if len(set(faults)) != len(faults):
            raise ManifestValidationError("fault_behaviors must be unique")
        signature = data["signature"]
        if not isinstance(signature, str) or not re.fullmatch(r"[0-9a-f]{64}", signature):
            raise ManifestValidationError(
                "signature must be a lowercase HMAC-SHA256 hex string"
            )

        return cls(
            manifest_version=version,
            device_id=device_id,
            device_type=device_type,
            protocol_version=protocol,
            commands=commands,
            sensors=sensors,
            max_torque_nm=_positive(data["max_torque_nm"], "max_torque_nm"),
            max_velocity_rad_s=_positive(data["max_velocity_rad_s"], "max_velocity_rad_s"),
            fault_behaviors=faults,
        )

    def unsigned_mapping(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "device_id": self.device_id,
            "device_type": self.device_type,
            "protocol_version": self.protocol_version,
            "commands": list(self.commands),
            "sensors": [sensor.to_mapping() for sensor in self.sensors],
            "max_torque_nm": self.max_torque_nm,
            "max_velocity_rad_s": self.max_velocity_rad_s,
            "fault_behaviors": list(self.fault_behaviors),
        }

    def signed_mapping(self, secret: bytes) -> dict[str, Any]:
        if not secret:
            raise ValueError("manifest signing secret must not be empty")
        payload = self.unsigned_mapping()
        signature = hmac.new(secret, _canonical_bytes(payload), hashlib.sha256).hexdigest()
        return {**payload, "signature": signature}

    @classmethod
    def from_signed_mapping(cls, value: Any, secret: bytes) -> "CapabilityManifest":
        if not secret:
            raise ValueError("manifest verification secret must not be empty")
        data = _object(value, "manifest")
        parsed = cls.from_mapping(data)
        expected = hmac.new(
            secret, _canonical_bytes(parsed.unsigned_mapping()), hashlib.sha256
        ).hexdigest()
        actual = data["signature"]
        if not hmac.compare_digest(expected, actual):
            raise ManifestValidationError("manifest signature verification failed")
        return parsed


@dataclass(frozen=True)
class NegotiationPolicy:
    """Minimum and maximum requirements for one controller integration."""

    protocol_version: str
    required_commands: tuple[str, ...]
    minimum_sensor_rates_hz: Mapping[str, float]
    required_fault_behaviors: tuple[str, ...]
    required_torque_nm: float
    required_velocity_rad_s: float
    max_allowed_torque_nm: float
    max_allowed_velocity_rad_s: float

    def __post_init__(self) -> None:
        if not self.protocol_version:
            raise ValueError("protocol_version must not be empty")
        if self.required_torque_nm <= 0 or self.required_velocity_rad_s <= 0:
            raise ValueError("required actuator ranges must be positive")
        if self.max_allowed_torque_nm <= 0 or self.max_allowed_velocity_rad_s <= 0:
            raise ValueError("maximum actuator limits must be positive")
        if self.required_torque_nm > self.max_allowed_torque_nm:
            raise ValueError("required torque exceeds policy maximum")
        if self.required_velocity_rad_s > self.max_allowed_velocity_rad_s:
            raise ValueError("required velocity exceeds policy maximum")


@dataclass(frozen=True)
class NegotiationResult:
    accepted: bool
    device_id: str | None
    reasons: tuple[str, ...]
    agreed_limits: Mapping[str, float] | None = None


class CapabilityNegotiator:
    """Verify and negotiate a signed manifest with stable refusal ordering."""

    def __init__(self, policy: NegotiationPolicy, secret: bytes) -> None:
        if not secret:
            raise ValueError("negotiation secret must not be empty")
        self._policy = policy
        self._secret = secret

    def negotiate(self, raw_manifest: Any) -> NegotiationResult:
        try:
            manifest = CapabilityManifest.from_signed_mapping(raw_manifest, self._secret)
        except (ManifestValidationError, ValueError, TypeError) as exc:
            return NegotiationResult(False, None, (f"manifest_invalid:{exc}",))

        reasons: list[str] = []
        if manifest.protocol_version != self._policy.protocol_version:
            reasons.append("protocol_version_mismatch")

        missing_commands = sorted(
            set(self._policy.required_commands) - set(manifest.commands)
        )
        if missing_commands:
            reasons.append("missing_commands:" + ",".join(missing_commands))

        by_name = {sensor.name: sensor for sensor in manifest.sensors}
        for name, minimum in sorted(self._policy.minimum_sensor_rates_hz.items()):
            capability = by_name.get(name)
            if capability is None:
                reasons.append(f"missing_sensor:{name}")
            elif capability.sample_rate_hz < minimum:
                reasons.append(f"sensor_rate_below_minimum:{name}")

        if manifest.max_torque_nm < self._policy.required_torque_nm:
            reasons.append("actuator_torque_below_required")
        if manifest.max_velocity_rad_s < self._policy.required_velocity_rad_s:
            reasons.append("actuator_velocity_below_required")
        if manifest.max_torque_nm > self._policy.max_allowed_torque_nm:
            reasons.append("actuator_torque_exceeds_safety_ceiling")
        if manifest.max_velocity_rad_s > self._policy.max_allowed_velocity_rad_s:
            reasons.append("actuator_velocity_exceeds_safety_ceiling")

        missing_faults = sorted(
            set(self._policy.required_fault_behaviors) - set(manifest.fault_behaviors)
        )
        if missing_faults:
            reasons.append("missing_fault_behaviors:" + ",".join(missing_faults))

        if reasons:
            return NegotiationResult(False, manifest.device_id, tuple(reasons))
        return NegotiationResult(
            accepted=True,
            device_id=manifest.device_id,
            reasons=("accepted",),
            agreed_limits={
                "max_torque_nm": self._policy.max_allowed_torque_nm,
                "max_velocity_rad_s": self._policy.max_allowed_velocity_rad_s,
            },
        )
