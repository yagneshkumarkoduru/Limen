"""Tests for strict profile validation and deterministic target generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from limen.profile import PartnerProfile, ProfileCompiler, ProfileValidationError


EXAMPLE = Path(__file__).parents[1] / "examples" / "controller_v0.json"


def profile_mapping() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def test_example_profile_validates() -> None:
    profile = PartnerProfile.from_mapping(profile_mapping())
    assert profile.profile_id == "assistive_controller_v0"
    assert len(profile.sensors) == 2
    assert profile.actuator.position_min_rad < profile.actuator.position_max_rad


def test_unknown_fields_are_rejected() -> None:
    raw = profile_mapping()
    raw["unexpected"] = True
    with pytest.raises(ProfileValidationError, match="unknown fields"):
        PartnerProfile.from_mapping(raw)


def test_missing_sensor_fault_path_is_rejected() -> None:
    raw = profile_mapping()
    raw["fault_paths"] = raw["fault_paths"][:1]
    with pytest.raises(ProfileValidationError, match="do not cover sensors"):
        PartnerProfile.from_mapping(raw)


def test_invalid_position_type_is_reported_as_profile_error() -> None:
    raw = profile_mapping()
    raw["actuator"]["position_min_rad"] = "-1.2"
    with pytest.raises(ProfileValidationError, match="position_min_rad"):
        PartnerProfile.from_mapping(raw)


def test_target_generation_is_deterministic(tmp_path: Path) -> None:
    profile = PartnerProfile.from_mapping(profile_mapping())
    compiler = ProfileCompiler()
    first = compiler.compile(profile)
    second = compiler.compile(profile)
    assert dict(first.files) == dict(second.files)
    assert set(first.files) == {
        "simulation_profile.json",
        "firmware_profile.h",
        "rtl_profile.vh",
        "fault_test_vectors.json",
    }

    written = first.write(tmp_path)
    assert {path.name for path in written} == {path.split("/")[-1] for path in first.files}
    simulation = json.loads((tmp_path / "simulation_profile.json").read_text(encoding="utf-8"))
    vectors = json.loads((tmp_path / "fault_test_vectors.json").read_text(encoding="utf-8"))
    assert simulation["profile"]["profile_id"] == profile.profile_id
    assert len(vectors["vectors"]) == len(profile.sensors) + 2


def test_generated_reference_targets_contain_profile_limits() -> None:
    profile = PartnerProfile.from_mapping(profile_mapping())
    files = ProfileCompiler().compile(profile).files
    assert "LIMEN_PROFILE_MAX_TORQUE_NM (25.000000F)" in files["firmware_profile.h"]
    assert "LIMEN_ASSISTIVE_CONTROLLER_V0_MAX_TORQUE_MNM = 25000" in files["rtl_profile.vh"]
