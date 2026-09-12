"""Limen profile compiler and schema.

Provides:
- ``PartnerProfile``: validated, immutable device profile
- ``ProfileCompiler``: compiles a profile into simulation, firmware, RTL, and test targets
- ``ProfileValidationError``: raised on invalid profile input
"""

from .compiler import CompilationArtifacts, ProfileCompiler
from .schema import (
    ActuatorSpec,
    FaultPath,
    PartnerProfile,
    ProfileValidationError,
    SensorSpec,
)

__all__ = [
    "PartnerProfile",
    "SensorSpec",
    "ActuatorSpec",
    "FaultPath",
    "ProfileValidationError",
    "ProfileCompiler",
    "CompilationArtifacts",
]
