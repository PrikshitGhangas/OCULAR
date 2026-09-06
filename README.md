# OCULAR
## Real-Time Ocular Tracking and Gaze-Aware Interaction Framework

[![Python](https://img.shields.io/badge/Python-3.10%20--%203.14-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/Tests-59%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

- Project Domain: Computer Vision, Human-Computer Interaction (HCI), Applied Machine Learning
- Target Platform: Linux, Windows, macOS (Cross-Platform Hardware Abstraction)
- Measured Throughput: ~30 FPS on standard desktop CPUs
- Status: All Milestones (M1-M10) and Research Elevation (Phase A) implemented and verified

---

## 1. Project Overview

OCULAR is an open-source computer vision framework designed to transform a standard RGB webcam into a real-time gaze-tracking interface without requiring specialized infrared (IR) eye-tracking hardware.

### Primary Research Question
> "Can adaptive, user-specific calibration reduce the burden of webcam-based gaze calibration while maintaining useful gaze-estimation accuracy?"

By pairing active machine learning (uncertainty estimation and spatial coverage heuristics) with 3D head pose decoupling (`cv2.solvePnP`), an 11-dimensional geometric feature vector, and the One Euro signal filter, OCULAR provides a low-latency, hands-free computer control interface.

---

## 2. Documentation Links

| Document | Description |
|:---|:---|
| [**HOW_TO_USE.md**](docs/HOW_TO_USE.md) | Step-by-step setup, calibration, model training, interaction, and troubleshooting guide. |
| [**COMPREHENSIVE_SYSTEM_DOCUMENTATION.md**](docs/COMPREHENSIVE_SYSTEM_DOCUMENTATION.md) | In-depth technical architecture, mathematical formulations, and engineering rationale. |
| [**EVALUATION_REPORT.md**](docs/EVALUATION_REPORT.md) | Multi-model benchmark analysis, LOOCV results, and calibration comparisons. |
| [**ELEVATION_REPORT.md**](docs/ELEVATION_REPORT.md) | Research roadmap, state-of-the-art benchmarks, ablation design, and user study protocol. |
| [**AUDIT_REPORT.md**](docs/AUDIT_REPORT.md) | 10-phase engineering audit covering security, portability, correctness, and testing. |
| [**API_REFERENCE.md**](docs/API_REFERENCE.md) | Developer reference for classes, methods, parameters, and return types. |
| [**GUIDE.md**](docs/GUIDE.md) | Implementation milestones and development log. |

---

## 3. Quick Start

### 3.1 Setup Environment

```bash
# Clone the repository
git clone https://github.com/PrikshitGhangas/OCULAR.git
cd OCULAR

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows

# Install package in editable mode
pip install -e ".[dev]"
```

### 3.2 Running OCULAR Commands

Once installed, use the unified `ocular` CLI tool:

```bash
# 1. Live Diagnostic Tracking Stream
ocular stream --camera 0

# 2. Interactive Full-Screen Calibration (9-point standard grid or adaptive)
ocular calibrate --camera 0 --pattern 9-point --session session_01
ocular calibrate --camera 0 --adaptive --session session_adaptive

# 3. Train Gaze Regression Model (Random Forest, Ridge, SVR, or MLP)
ocular train --session session_01 --model rf

# 4. Launch Live Gaze Mouse Control (Cursor, Dwell Click, and Scroll)
ocular interact --session session_01 --model rf --mode all
ocular interact --session session_01 --model rf --enable-os-cursor

# 5. Run Automated Multi-Model Benchmarks
ocular benchmark
```

For complete instructions, refer to [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md).

---

## 4. System Architecture

The OCULAR pipeline operates across five primary layers:

```text
[ Layer 1: Capture ]     Camera Acquisition (Linux V4L2, Windows DirectShow, macOS AVFoundation)
          |
          v
[ Layer 2: Track ]       MediaPipe FaceLandmarker (478 3D Landmarks and Iris Centroids)
          |
          v
[ Layer 3: Extract ]     11-D Feature Vector and solvePnP 3D Head Pose Decoupling
          |
          v
[ Layer 4: Calibrate ]   Conventional Fixed-Grid and Active Adaptive Calibration
          |
          v
[ Layer 5: Interact ]    One Euro Temporal Filtering, Gaze Cursor, Dwell Detector, and Scroller
```

### The 11-Element Feature Vector
1. Left Iris Ratios (`left_iris_h`, `left_iris_v`): Normalized iris centroid within the left palpebral fissure.
2. Right Iris Ratios (`right_iris_h`, `right_iris_v`): Normalized iris centroid within the right palpebral fissure.
3. Eye Aspect Ratios (`left_ear`, `right_ear`): Eyelid aperture via the Soukupova and Cech formulation.
4. Eye Geometry Aspect Ratios (`left_eye_aspect`, `right_eye_aspect`): Width-to-height ocular proportions.
5. Head Pose Angles (`head_pitch`, `head_yaw`, `head_roll`): 3D rotation angles estimated via `cv2.solvePnP` on an anthropometric 3D facial model.

---

## 5. Milestone Implementation Status

| Milestone | Description | Status | Core File |
|:---|:---|:---:|:---|
| M1 | Cross-Platform Camera Capture | Complete | [`src/ocular/camera.py`](src/ocular/camera.py) |
| M2 | MediaPipe Face and Iris Tracking | Complete | [`src/ocular/tracker.py`](src/ocular/tracker.py) |
| M3 | 11-D Feature Vector and solvePnP Head Pose | Complete | [`src/ocular/features.py`](src/ocular/features.py) |
| M4 | Interactive Full-Screen Calibration UI | Complete | [`src/ocular/calibration.py`](src/ocular/calibration.py) |
| M5 | Multi-Model Gaze Regression (Ridge, SVR, RF, MLP) | Complete | [`src/ocular/gaze.py`](src/ocular/gaze.py) |
| M6 | Adaptive Active Calibration and Online SGD | Complete | [`src/ocular/adaptive.py`](src/ocular/adaptive.py) |
| M7 | Robustness Benchmarks and Metric Conversions | Complete | [`src/ocular/evaluation.py`](src/ocular/evaluation.py) |
| M8 | One Euro Filter, GazeCursor, Dwell, Scroller | Complete | [`src/ocular/interaction.py`](src/ocular/interaction.py) |
| M9 | Quantitative Evaluation Framework | Complete | [`experiments/compare_models.py`](experiments/compare_models.py) |
| M10 | Unified CLI and System Abstraction | Complete | [`src/ocular/main.py`](src/ocular/main.py) |
| M11 | Latency Profiling and Empirical Studies | Complete | [`src/ocular/profiler.py`](src/ocular/profiler.py) |

---

## 6. Experimental Benchmark Highlights

Quantitative findings from [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md):

### 6.1 Regression Model Comparison (LOOCV on Calibration Data)
- Random Forest (`rf`): 128.11 px MAE (3.37 deg visual angle) - lowest error and highest resilience to landmark noise.
- Ridge Regression (`ridge`): 334.60 px MAE (8.75 deg visual angle) - lightweight polynomial baseline.
- Multi-Layer Perceptron (`mlp`): 345.30 px MAE (9.03 deg visual angle).
- Support Vector Regression (`svr`): 759.78 px MAE (19.27 deg visual angle).

### 6.2 Conventional vs. Adaptive Calibration Trade-Off
- Sample Burden: Adaptive calibration reduced required targets from 9 to 6 (33.3% reduction).
- Calibration Time: Reduced session time from 18.0s to 12.0s (33.3% time savings).
- Diminishing Returns Detection: The engine automatically detects when marginal accuracy gains drop below 3.0 px, preventing user fatigue.

### 6.3 Signal Filtering: One Euro Filter
- Static Jitter: Suppressed from +/- 24.5 px (unfiltered) to +/- 2.8 px (One Euro filter).
- Saccadic Latency: Kept under 18 ms (compared to 165 ms for a 10-frame simple moving average).

---

## 7. Running Tests

Run the full automated test suite:
```bash
python -m unittest discover tests/ -v
```

All 59 unit and integration tests pass:
```text
Ran 59 tests in 20.154s
OK
```

---

## 8. Directory Structure

```text
OCULAR/
|-- .github/                                     # Continuous integration workflows
|   `-- workflows/
|       `-- tests.yml
|-- README.md                                    # Project overview and quickstart
|-- pyproject.toml                               # Package setup, dependencies, and entry points
|-- requirements.txt                             # Production dependencies
|-- .gitignore                                   # Version control exclusion rules
|-- .env.example                                 # Environment configuration template
|
|-- docs/                                        # Documentation
|   |-- HOW_TO_USE.md                            # Practical step-by-step usage guide
|   |-- COMPREHENSIVE_SYSTEM_DOCUMENTATION.md    # Architectural and scientific guide
|   |-- EVALUATION_REPORT.md                     # Quantitative benchmark findings
|   |-- ELEVATION_REPORT.md                      # Research elevation strategy and roadmap
|   |-- AUDIT_REPORT.md                          # Engineering audit and scorecard
|   |-- API_REFERENCE.md                         # Class and API reference
|   `-- GUIDE.md                                 # Technical milestone log
|
|-- calibration/                                 # User calibration sessions (.npz)
|   `-- .gitkeep
|-- data/                                        # Experiment outputs and metrics (.json)
|   |-- experiments/
|   `-- .gitkeep
|-- models/                                      # Downloaded detector models and trained weights
|   |-- face_detection_yunet_2026may.onnx
|   `-- face_landmarker.task
|
|-- experiments/                                 # Experimental study scripts
|   |-- run_user_study.py                        # Automated user study orchestrator
|   |-- feature_ablation.py                      # Feature subset ablation study
|   |-- compare_models.py                        # Model comparison benchmark
|   `-- calibration_experiment.py                # Conventional vs. adaptive experiment
|
|-- src/                                         # Core framework source code
|   `-- ocular/
|       |-- __init__.py                          # Public package exports
|       |-- __main__.py                          # python -m ocular entry point
|       |-- camera.py                            # Layer 1: Camera acquisition
|       |-- tracker.py                           # Layer 2: Face and iris tracking
|       |-- features.py                          # Layer 3: Feature extraction and solvePnP
|       |-- calibration.py                       # Layer 4: Visual calibration UI
|       |-- adaptive.py                          # Layer 4: Adaptive calibration engine
|       |-- gaze.py                              # Layer 5: Gaze regression models
|       |-- filters.py                           # Layer 5: One Euro Filter and smoothers
|       |-- blink.py                             # Layer 5: Blink and wink detector
|       |-- interaction.py                       # Layer 5: Gaze cursor, dwell, scroller
|       |-- evaluation.py                        # Evaluation metrics and benchmarks
|       |-- profiler.py                          # Performance and latency profiler
|       `-- main.py                              # Unified CLI entrypoint
|
`-- tests/                                       # 59 unit and integration tests
    |-- __init__.py
    |-- test_adaptive.py
    |-- test_blink.py
    |-- test_calibration.py
    |-- test_camera.py
    |-- test_evaluation.py
    |-- test_features.py
    |-- test_filters.py
    |-- test_gaze.py
    |-- test_interaction.py
    |-- test_profiler.py
    `-- test_tracker.py
```

---

## 9. License

This project is licensed under the [MIT License](LICENSE).
