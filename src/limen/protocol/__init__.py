"""Limen capability negotiation protocol.

Provides:
- ``CapabilityManifest``: validated device capability declaration
- ``CapabilityNegotiator``: verify and negotiate a signed manifest
- ``NegotiationPolicy``: minimum and maximum requirements
- ``NegotiationResult``: accepted/refused result with stable reasons
- ``SensorCapability``: one sensor capability entry
- ``ManifestValidationError``: raised on malformed manifests
"""

from .negotiation import (
    CapabilityManifest,
    CapabilityNegotiator,
    ManifestValidationError,
    NegotiationPolicy,
    NegotiationResult,
    SensorCapability,
)

__all__ = [
    "CapabilityManifest",
    "CapabilityNegotiator",
    "ManifestValidationError",
    "NegotiationPolicy",
    "NegotiationResult",
    "SensorCapability",
]
