# Limen

Threshold-gated safety architecture for confidence-coupled embedded control.

## What it is

Limen is a research project studying how to make intent-driven physical
controllers safer when their underlying sensor signals are noisy or
degraded. The core problem: a machine learning classifier that translates
biosignals (EMG, IMU) into control commands produces a confidence score
alongside its prediction. Most embedded control systems ignore that
confidence score at the actuation stage. Limen builds a framework where
the confidence score and an independent estimate of signal quality jointly
determine how much authority the actuator is given - in real time, per
decision window, at 50 Hz.

The name comes from the Latin word for threshold - specifically, the
confidence threshold that separates safe from constrained from halted
operation.

## Three interconnected components

### 1. Confidence-Coupled Safety Envelope

The envelope sits between the inference engine and the actuator and
continuously computes an effective confidence value:

```
C_eff = inference_confidence * signal_reliability
```

Three operating tiers determined by configurable thresholds:

| C_eff range          | Tier        | Action                               |
|----------------------|-------------|--------------------------------------|
| >= T_high (0.75)     | NOMINAL     | Full actuator authority              |
| T_low to T_high      | DEGRADED    | 40% torque, 50% velocity limit       |
| < T_low (persistent) | SAFE_HALT   | Zero velocity, hold position         |

An Independent Hardware Supervisor layer enforces absolute kinematic bounds
regardless of confidence level - even a high-confidence nominal command
cannot exceed the hard maximum torque and velocity limits.

Implemented in Python (`limen/safety.py`) with a bit-exact fixed-point C
reference (`firmware/limen_supervisor.c`) and a synthesizable
SystemVerilog reference (`rtl/limen_safety_supervisor.sv`).

### 2. Intent-Control Profile Compiler

A device profile is a single JSON file that declares sensors, intent
classes, actuator limits, and per-sensor fault paths. The compiler
translates this profile deterministically into four cross-domain targets:

- `simulation_profile.json` - runtime configuration for the Python simulator
- `firmware_profile.h` - C header with structs and `#define` macros
- `rtl_profile.vh` - Verilog localparams
- `fault_test_vectors.json` - auto-generated test scenarios for every fault path

The strict validator rejects unknown fields, missing fault paths, incoherent
actuator limits, and unsupported sensor types rather than silently applying
defaults. Implemented in `limen/profile/`.

### 3. Eventized Multi-Rate Scheduler

The scheduler keeps safety supervision on a continuous mandatory tick
(1 kHz) while inference runs only when a meaningful signal event is present.
This separates the guaranteed safety period from the cost of running the
classifier - relevant for MCU targets where inference cannot run at 1 kHz.

```
signal ingestion -> safety tick -> inference (if event pending) -> telemetry
```

Implemented as a deterministic virtual scheduler in `limen/scheduler.py`.

## Evidence boundary

All current results are from software simulation on a workstation with
synthetic EMG/IMU data.

| Claim | Evidence class | Value |
|---|---|---|
| Safety envelope triggers DEGRADED on electrode dropout | Software fault injection | Yes - see test_safety_envelope.py |
| Safety envelope reaches SAFE_HALT after 3 consecutive low-quality windows | Software fault injection | Yes - persistence_windows=3 |
| Independent supervisor vetoes over-limit commands even at NOMINAL confidence | Software fault injection | Yes - test_supervisor_vetoes_beyond_hard_bounds |
| Pipeline accuracy on clean synthetic data | Software simulation | 1.0000 (LDA, 4365 eval windows, seed 20260722) |
| Pipeline accuracy under fault injection | Software simulation | 0.9883 (LDA), 0.9885 (temporal-LDA) |
| Profile compiler generates deterministic artifacts | Unit test | 28 tests pass |
| Scheduler fires safety ticks continuously regardless of inference events | Unit test | Yes - test_scheduler.py |
| C firmware reproduces Python pipeline outputs exactly | Host conformance | 30/30 golden vectors - limen_conformance.exe |

**Not yet demonstrated:** embedded timing on MCU hardware, physical actuator
trials, real biosignal datasets, power measurement on target silicon.

## Quick start

### Python benchmark

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
python -m limen --out out/
```

Writes `out/benchmark_report.json` and `out/benchmark_report.md`.

### Profile compiler

```powershell
python -m limen.profile compile examples/controller_v0.json --out out/profile
```

### Firmware (host conformance)

```powershell
cmake -S firmware -B firmware/build-host -G Ninja -DCMAKE_C_COMPILER=gcc
cmake --build firmware/build-host
.\firmware\build-host\limen_conformance.exe
```

### Run tests

```powershell
pytest tests -q
```

## Repository structure

```
Limen/
├── src/limen/           # Python package
│   ├── safety.py        # Confidence-Coupled Safety Envelope
│   ├── scheduler.py     # Eventized multi-rate scheduler
│   ├── signals.py       # Deterministic EMG/IMU generator + fault injection
│   ├── preprocessing.py # Band-pass filter, feature extraction
│   ├── inference.py     # LDA baselines with calibrated confidence
│   ├── telemetry.py     # Per-window records and aggregate metrics
│   ├── report.py        # JSON + Markdown report generation
│   ├── benchmark.py     # Replay runner
│   ├── datasets.py      # Dataset adapters
│   ├── profile/         # Profile compiler
│   │   ├── schema.py    # Strict profile validation
│   │   └── compiler.py  # Cross-domain artifact generation
│   └── protocol/        # Capability negotiation
│       └── negotiation.py
├── firmware/            # Fixed-point C reference
│   ├── limen_supervisor.c   # State machine (mirrors Python exactly)
│   ├── limen_sqg.c          # Integer signal-quality gate
│   ├── limen_features.c     # Integer feature extraction
│   ├── limen_inference.c    # Quantized LDA with exp-LUT
│   ├── limen_pipeline.c     # Full decision loop
│   ├── limen_trace.c        # Ring buffer with CRC8
│   └── tests/
│       ├── test_conformance.c
│       └── golden_conformance.h
├── rtl/                 # SystemVerilog reference
│   ├── limen_safety_supervisor.sv
│   └── limen_event_queue.sv
├── tests/               # Python test suite
│   ├── test_safety_envelope.py
│   ├── test_scheduler.py
│   ├── test_profile_compiler.py
│   └── test_capability_negotiation.py
└── examples/
    └── controller_v0.json
```

## Design decisions

**Why a separate signal quality gate?**
Inference confidence alone is not sufficient for safety decisions. A classifier
can be highly confident about a misprediction on corrupted input. The SQG
detects flatline channels, saturation clipping, and motion artifacts before
the confidence computation, acting as an independent reliability modifier.

**Why fixed-point C?**
Embedded MCU targets (Cortex-M class) typically lack an FPU or have a slow
one. The fixed-point C implementation is a direct translation of the Python
reference using Q8 and permille arithmetic. The golden vector conformance
test verifies bit-exact agreement between the two implementations.

**Why a profile compiler?**
Hardware integration typically requires the same constants defined separately
in firmware headers, RTL parameters, and simulation configs. Drift between
these creates silent bugs at bring-up. Compiling from a single validated
source eliminates this class of error.

**Why eventized inference?**
On hardware, running the full classification pipeline every 1 ms is too
expensive for the safety tick but 20 ms (50 Hz) is acceptable for intent
decisions. The scheduler keeps safety mandatory at 1 kHz while inference
fires only when a new signal event warrants it.
