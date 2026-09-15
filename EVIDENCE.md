# Limen Evidence Record

Maintained by: Koduru Yagnesh Kumar
Updated: 2026-09-12
Status: Software simulation stage. No hardware runs yet.

This file documents every number used in project descriptions, README
claims, and admissions materials. No claim appears in public without a
source entry here.

---

## 1. Python Benchmark Results (seed 20260722)

Run date: 2026-08-22 (original simulation run from backup archive)
Environment: Python 3.11.15, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.6.1
Dataset: limen-synthetic-emg-imu (deterministic in-repo generator, seeded)
Signal: EMG 1000 Hz x 8 channels, IMU 100 Hz, window 200 ms, stride 20 ms

### 1a. Clean condition

| Model | Eval windows | Accuracy | False activation rate | Rejection rate |
|---|---|---|---|---|
| classical-lda | 4365 | 1.0000 | 0.0000 | 0.0000 |
| temporal-lda-smoothing | 4365 | 1.0000 | 0.0000 | 0.0000 |

Envelope events (clean): 4365 nominal, 0 degraded, 0 safe_halt
End-to-end latency (classical-lda, clean): p50 0.3242 ms, p95 0.8399 ms, p99 1.1774 ms, max 3.1236 ms

### 1b. Fault-injected condition

| Model | Eval windows | Accuracy | Rejection rate | Safe-halt events |
|---|---|---|---|---|
| classical-lda | 4365 | 0.9883 | 0.0708 | 300 |
| temporal-lda-smoothing | 4365 | 0.9885 | 0.0779 | 300 |

Fault schedule: channel 1 dropout at 3.0 s, motion artifact at 4.5 s
(amplitude 8x nominal), burst duration 250 ms.

SAFE_HALT events confirm the envelope correctly detects and responds to
injected degradation. Research gate status: PASS.

### Interpretation limits

- These numbers use synthetic data where class separability is higher than
  real surface EMG. They demonstrate the envelope mechanism, not real-world
  accuracy.
- Latency is measured on a development workstation, not an embedded target.
- Zero-phase filtfilt is used for offline replay; a causal filter adds delay.
- Torque and velocity values are simulated actuator abstractions.

---

## 2. Safety Envelope Tests (Python)

Location: tests/test_safety_envelope.py
Result: 6/6 pass

| Test | What it proves |
|---|---|
| test_full_confidence_clean_signal_stays_nominal | C_eff >= T_high -> NOMINAL state, positive commands |
| test_electrode_dropout_triggers_persistent_safe_halt | Dropout -> DEGRADED then SAFE_HALT after persistence_windows |
| test_degraded_confidence_damps_output | T_low <= C_eff < T_high -> DEGRADED, scaled torque and velocity |
| test_supervisor_vetoes_beyond_hard_bounds | Hard supervisor clips over-limit commands regardless of confidence |
| test_unknown_intent_fails_safe | Unknown intent class -> SAFE_HALT immediately |
| test_recovery_requires_high_confidence | SAFE_HALT -> DEGRADED at mid-band, NOMINAL requires C_eff >= T_high |

---

## 3. Scheduler Tests (Python)

Location: tests/test_scheduler.py
Result: 4/4 pass

| Test | What it proves |
|---|---|
| test_safety_ticks_are_continuous_without_inference_events | 21 safety ticks in 20 ms, 0 inference events when no signal |
| test_meaningful_signal_activates_one_due_inference_cycle | Meaningful event triggers exactly one inference cycle |
| test_same_timestamp_signal_consumed_before_inference | Event ordering: safety tick -> inference -> telemetry at same timestamp |
| test_scheduler_is_deterministic_and_rejects_unordered_input | Identical inputs produce identical outputs; out-of-order input raises ValueError |

---

## 4. Profile Compiler Tests (Python)

Location: tests/test_profile_compiler.py
Result: 5/5 pass

| Test | What it proves |
|---|---|
| test_example_profile_validates | Reference profile parses and validates correctly |
| test_unknown_fields_are_rejected | Strict schema rejects extra fields |
| test_missing_sensor_fault_path_is_rejected | All sensors must have a fault path |
| test_invalid_position_type_is_reported_as_profile_error | Type errors produce ProfileValidationError |
| test_target_generation_is_deterministic | Same profile always produces same 4 artifacts |

Profile limits correctly propagated:
- Firmware header contains: `LIMEN_PROFILE_MAX_TORQUE_NM (25.000000F)`
- RTL parameters contain: `LIMEN_ASSISTIVE_CONTROLLER_V0_MAX_TORQUE_MNM = 25000`

---

## 5. Capability Negotiation Tests (Python)

Location: tests/test_capability_negotiation.py
Result: 5/5 pass

| Test | What it proves |
|---|---|
| test_signed_manifest_is_accepted | Valid signed manifest with matching policy -> accepted with agreed limits |
| test_tampering_is_refused_before_capability_checks | HMAC tamper detection fires before any capability check |
| test_missing_command_and_sensor_are_refused_explicitly | Refusal reasons are explicit and enumerated |
| test_over_limit_device_is_refused | Device exceeding safety ceiling is refused even if it has required capabilities |
| test_negotiation_reasons_are_stable | Same input always produces same reasons (deterministic refusal) |

---

## 6. Firmware Conformance (C)

Location: firmware/tests/test_conformance.c + golden_conformance.h
Result (from backup archive): 30/30 checks pass
Model CRC32: 0x5D9CA66A (verified at runtime before running vectors)

| Suite | Vectors | Result |
|---|---|---|
| Single window (golden_single) | 18 | PASS |
| Sequence (golden_sequence, stateful supervisor) | 9 | PASS |
| Trace ring buffer wrap and CRC8 validation | 3 | PASS |

The C fixed-point pipeline (features -> SQG -> inference -> supervisor) reproduces
the Python simulation output exactly on 18 independent windows and 9 stateful
sequence windows. This confirms behavioral parity between the Python reference
and the C firmware implementation.

---

## 7. What is NOT yet demonstrated

- Embedded timing on any MCU hardware (Cortex-M class or otherwise)
- Physical actuator trials (real motor or joint)
- Real biosignal datasets (NinaPro or equivalent)
- Power measurement on target silicon
- RTL synthesis, timing closure, or FPGA floorplan
- Any human participant involvement

Do not claim embedded timing, power, or physical system results without
adding them to this file with a hardware run ID and measurement context.
