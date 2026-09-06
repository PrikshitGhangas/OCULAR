# OCULAR
## Real-Time Ocular Tracking and Gaze-Aware Interaction Framework

[![Python](https://img.shields.io/badge/Python-3.9%20--%203.14-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/Tests-17%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**Project Domain:** Computer Vision · Human-Computer Interaction (HCI) · Applied Machine Learning  
**Target Platform:** Linux & Windows (Cross-Platform Hardware Abstraction)  
**Measured Throughput:** ~30 FPS on Standard Desktop CPUs  
**Status:** **All Milestones (M1–M10) Fully Implemented and Verified**

---

## 1. Project Overview

**OCULAR** is an open-source, research-driven computer vision framework designed to transform an ordinary, commodity RGB webcam into an accurate, real-time gaze-tracking interface without requiring specialized infrared (IR) eye-tracking hardware.

### Primary Research Question
> **"Can adaptive, user-specific calibration reduce the burden of webcam-based gaze calibration while maintaining useful gaze-estimation accuracy?"**

By pairing active machine learning (uncertainty estimation via Gaussian Processes and regional residual heuristics) with 3D head pose decoupling (`cv2.solvePnP`), an 11-dimensional geometric feature vector, and human-computer interaction signal filtering (the One Euro Filter), OCULAR provides a practical, low-latency, hands-free computer control interface.

---

## 2. Documentation Quick Links

| Document | Description |
|:---|:---|
| [**COMPREHENSIVE_SYSTEM_DOCUMENTATION.md**](docs/COMPREHENSIVE_SYSTEM_DOCUMENTATION.md) | In-depth technical walkthrough, mathematical formulations, and engineering rationale. |
| [**EVALUATION_REPORT.md**](docs/EVALUATION_REPORT.md) | Multi-model benchmark analysis, LOOCV results, and Conventional vs. Adaptive calibration study. |
| [**API_REFERENCE.md**](docs/API_REFERENCE.md) | Complete developer reference of all classes, methods, parameters, and return types. |
| [**GUIDE.md**](docs/GUIDE.md) | Step-by-step milestone roadmap and engineering reference manual. |

---

## 3. Quick Start & Installation

### 3.1 Setup Environment

```bash
# Clone the repository
git clone https://github.com/PrikshitGhangas/OCULAR.git
cd OCULAR

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux / macOS
# .venv\Scripts\activate        # On Windows

# Install package and dependencies in editable mode
pip install -e .
```

### 3.2 Running OCULAR Commands

Once installed, the unified `ocular` CLI tool is available anywhere in your environment:

```bash
# 1. Live Diagnostic Tracking Stream (Inspect face mesh, iris centers, and head pose)
ocular stream

# 2. Interactive Full-Screen Calibration (9-point standard grid)
ocular calibrate --pattern 9-point --output calibration/session_01.npz

# 3. Train Gaze Regression Model (Random Forest or Ridge)
ocular train --input calibration/session_01.npz --model rf --output-dir models/

# 4. Launch Live Hands-Free Gaze Mouse Control (Cursor, Dwell Click & Scroll)
ocular interact --model rf --model-dir models/ --enable-os-cursor

# 5. Run Automated Multi-Model Benchmarks
ocular benchmark
```

---

## 4. System Architecture

The OCULAR pipeline operates across five conceptual layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          THE OCULAR PIPELINE                                │
│                                                                             │
│  [ LAYER 1: SEE ]        Webcam Acquisition (Linux V4L2 / Windows DirectShow)│
│         │                                                                   │
│         ▼                                                                   │
│  [ LAYER 2: UNDERSTAND ] MediaPipe FaceLandmarker (478 3D Landmarks & Iris) │
│         │                                                                   │
│         ▼                                                                   │
│  [ LAYER 3: ESTIMATE ]   11-D Feature Extraction & solvePnP 3D Head Pose    │
│         │                                                                   │
│         ▼                                                                   │
│  [ LAYER 4: PERSONALIZE] Conventional & Adaptive Calibration (Active ML)    │
│         │                                                                   │
│         ▼                                                                   │
│  [ LAYER 5: ACT ]        HCI Interaction (1€ Filter, GazeCursor, Dwell)     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The 11-Element Feature Vector
1. **Iris Horizontal & Vertical Ratios (`left_iris_h_ratio`, `left_iris_v_ratio`):** Normalized sub-pixel iris centroid within left palpebral fissure `[0.0, 1.0]`.
2. **Right Iris Ratios (`right_iris_h_ratio`, `right_iris_v_ratio`):** Normalized iris centroid within right palpebral fissure `[0.0, 1.0]`.
3. **Eye Aspect Ratios (`left_ear`, `right_ear`):** Eyelid openness via the Soukupová & Čech formulation.
4. **Eye Geometry Aspects (`left_eye_aspect`, `right_eye_aspect`):** Width-to-height ocular proportions.
5. **Head Pose Angles (`head_pitch`, `head_yaw`, `head_roll`):** 3D rotation angles estimated via `cv2.solvePnP` on an anthropometric 3D face model, mathematically decoupling head movement from eye rotation.

---

## 5. Milestone Implementation Status

| Milestone | Description | Status | Core File |
|:---|:---|:---:|:---|
| **M1** | Cross-Platform Camera Capture | Complete | [`src/ocular/camera.py`](src/ocular/camera.py) |
| **M2** | MediaPipe Face & Iris Tracking | Complete | [`src/ocular/tracker.py`](src/ocular/tracker.py) |
| **M3** | 11-D Feature Vector & solvePnP Head Pose | Complete | [`src/ocular/features.py`](src/ocular/features.py) |
| **M4** | Interactive Full-Screen Calibration UI | Complete | [`src/ocular/calibration.py`](src/ocular/calibration.py) |
| **M5** | Multi-Model Gaze Regression (Ridge, SVR, RF, MLP) | Complete | [`src/ocular/gaze.py`](src/ocular/gaze.py) |
| **M6** | Adaptive Active Calibration & Online SGD | Complete | [`src/ocular/adaptive.py`](src/ocular/adaptive.py) |
| **M7** | Robustness Benchmarks & Metric Conversions | Complete | [`src/ocular/evaluation.py`](src/ocular/evaluation.py) |
| **M8** | 1€ Signal Filter, GazeCursor, Dwell, Scroller | Complete | [`src/ocular/interaction.py`](src/ocular/interaction.py) |
| **M9** | Quantitative Evaluation Framework | Complete | [`experiments/compare_models.py`](experiments/compare_models.py) |
| **M10** | OS Abstraction, Packaging, & Unified CLI | Complete | [`src/ocular/main.py`](src/ocular/main.py) |

---

## 6. Experimental Benchmark Highlights

Quantitative findings from [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md):

### 6.1 Regression Model Comparison (LOOCV on Calibration Data)
* **Random Forest (`rf`):** **128.11 px MAE** ($\mathbf{3.37^\circ}$ **visual angle**) — *Lowest error and highest resilience to landmark noise*.
* **Ridge Regression (`ridge`):** **334.60 px MAE** ($\mathbf{8.75^\circ}$ **visual angle**) — *Lightweight second-order polynomial baseline*.
* **Multi-Layer Perceptron (`mlp`):** 345.30 px MAE ($9.03^\circ$ visual angle).
* **Support Vector Regression (`svr`):** 759.78 px MAE ($19.27^\circ$ visual angle).

### 6.2 Conventional vs. Adaptive Calibration Trade-Off
* **Sample Burden:** Adaptive calibration reduced required targets from **9 down to 6** (**33.3% reduction**).
* **Calibration Time:** Reduced session time from **18.0s to 12.0s** (**33.3% time savings**).
* **Diminishing Returns Detection:** The engine automatically detected when marginal accuracy gains dropped below 3.0 px, preventing user fatigue.

### 6.3 Signal Filtering: One Euro (1€) Filter
* **Static Jitter:** Suppressed from $\pm 24.5\text{ px}$ (unfiltered) to $\mathbf{\pm 2.8\text{ px}}$ (1€ filter).
* **Saccadic Latency:** Kept under **18 ms** (vs. 165 ms for a 10-frame simple moving average).

---

## 7. Running Tests

Run the test suite across all modules:
```bash
python -m unittest discover tests
```
All **17 tests** pass:
```text
Ran 17 tests in 2.989s
OK
```

---

## 8. Directory Structure

```text
OCULAR/
├── README.md                                    # Project overview & quickstart (This file)
├── pyproject.toml                               # Package setup & entry points
├── requirements.txt                             # Production dependencies
├── .gitignore                                   # Version control exclusion rules
├── .env.example                                 # Environment configuration template
│
├── archive/                                     # Legacy and prototype scripts
│   ├── pupil.py
│   ├── cameraTest.py
│   ├── eyeTest.py
│   └── trackerTest.py
│
├── docs/                                        # Comprehensive project documentation
│   ├── GUIDE.md                                 # Technical roadmap & milestone manual
│   ├── COMPREHENSIVE_SYSTEM_DOCUMENTATION.md    # In-depth architectural & scientific guide
│   ├── EVALUATION_REPORT.md                     # Quantitative benchmark & research findings
│   └── API_REFERENCE.md                         # Detailed developer class & API reference
│
├── calibration/                                 # User calibration sessions (.npz)
│   └── .gitkeep
├── data/                                        # Generated experiment outputs (.json)
│   ├── experiments/
│   └── .gitkeep
├── models/                                      # Models directory
│   ├── face_detection_yunet_2026may.onnx        # YuNet detector model
│   └── face_landmarker.task                     # MediaPipe 478-landmark task model
│
├── experiments/                                 # Evaluation scripts
│   ├── compare_models.py                        # Model comparison benchmark
│   └── calibration_experiment.py                # Conventional vs. Adaptive experiment
│
├── src/                                         # Core framework source code
│   └── ocular/
│       ├── __init__.py                          # Public package exports
│       ├── __main__.py                          # python -m ocular entry point
│       ├── camera.py                            # Layer 1: Camera acquisition
│       ├── tracker.py                           # Layer 2: MediaPipe face/iris tracking
│       ├── features.py                          # Layer 3: Feature extraction & solvePnP
│       ├── calibration.py                       # Layer 4: Fullscreen visual calibration
│       ├── adaptive.py                          # Layer 4: Adaptive calibration engine
│       ├── gaze.py                              # Layer 5: Gaze regression models
│       ├── filters.py                           # Layer 5: One Euro Filter & smoothers
│       ├── blink.py                             # Layer 5: Blink/wink event detector
│       ├── interaction.py                       # Layer 5: Gaze cursor, dwell, scroller
│       ├── evaluation.py                        # Performance metrics & benchmarks
│       └── main.py                              # Unified CLI entrypoint
│
└── tests/                                       # 41 unit and integration tests
    ├── __init__.py
    ├── test_blink.py
    ├── test_calibration.py
    ├── test_camera.py
    ├── test_evaluation.py
    ├── test_features.py
    ├── test_filters.py
    ├── test_gaze.py
    ├── test_interaction.py
    └── test_tracker.py
```

---

## 9. License

This project is licensed under the [MIT License](LICENSE).
