# OCULAR: Step-by-Step Usage Guide

This guide walks through setting up and running OCULAR, from initial camera checks to live gaze interaction, benchmarking, and experimental studies.

---

## 1. Prerequisites and Installation

### System Requirements
- Operating System: Linux, Windows 10/11, or macOS 12+
- Python: 3.10 to 3.14
- Hardware: Standard USB or integrated webcam (720p at 30 fps recommended), standard monitor

### Environment Setup

1. Clone the repository and navigate into the project directory:
   ```bash
   git clone https://github.com/PrikshitGhangas/OCULAR.git
   cd OCULAR
   ```

2. Create and activate a virtual environment:
   - On Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. Install production dependencies and the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -e .
   ```

4. (Optional) Install development dependencies for running tests:
   ```bash
   pip install -e ".[dev]"
   ```

5. Verify the installation by running the test suite:
   ```bash
   python -m unittest discover tests/ -v
   ```
   All 59 unit and integration tests should pass.

---

## 2. Environment Configuration

Copy the example configuration file:
```bash
cp .env.example .env
```

Adjust the values in `.env` if your environment requires non-default settings:
- `OCULAR_CAMERA`: Set to `0` for default webcam, `1` for an external webcam, or a device path like `/dev/video0` on Linux.
- `OCULAR_PATTERN`: Default calibration grid (`5-point`, `9-point`, `13-point`, or `16-point`).
- `OCULAR_MODEL`: Default regression algorithm (`rf`, `ridge`, `svr`, or `mlp`).

---

## 3. Step 1: Diagnostics and Face Tracking

Before calibrating, verify that your webcam is recognized and that MediaPipe can detect your facial landmarks and iris centers.

Run the streaming diagnostic tool:
```bash
ocular stream --camera 0
```

### What to check:
- Green dots should outline your face geometry (forehead, jawline, lips, nose).
- Red contours should align closely with both eye openings.
- Blue circles should sit precisely on the center of each iris.
- The top-left HUD displays current FPS and tracking confidence.
- Head pose Euler angles (pitch, yaw, roll) appear in real time.

Press `q` in the video window to close the stream.

### Tips for clean tracking:
- Position the camera directly in front of you, centered above or below your screen.
- Avoid strong backlight (such as an open window behind your chair).
- Keep your head roughly 50 to 70 cm away from the display.

---

## 4. Step 2: Screen Calibration

Calibration collects paired samples of your eye features while you fixate on specific target coordinates on screen.

### Option A: Standard Fixed-Grid Calibration
Run a standard 9-point calibration:
```bash
ocular calibrate --camera 0 --pattern 9-point --dwell 1.8 --session my_session
```

Parameters:
- `--pattern`: Choose from `5-point`, `9-point`, `13-point`, or `16-point`. `9-point` offers the best balance between speed and coverage.
- `--dwell`: Seconds to hold gaze on each target (default: 1.8s). The system ignores the initial saccade and blinks, recording only stable fixations.
- `--session`: Identifier name. Data saves automatically to `calibration/<session>.npz`.

### Option B: Active Adaptive Calibration
If you want the system to choose target points based on spatial coverage and uncertainty:
```bash
ocular calibrate --camera 0 --adaptive --dwell 1.8 --session adaptive_session
```
In adaptive mode:
- Phase 1 presents a 5-point seed grid (center and corners).
- Phase 2 selects the next most informative points across the display until convergence or budget limit (up to 12 points).

### Calibration Procedure:
1. A fullscreen dark window will open.
2. An animated target dot with an expanding progress ring will appear.
3. Fixate steadily on the center of the dot without tilting your head excessively.
4. When the progress ring fills, the target will move to the next coordinate.
5. Once complete, summary quality metrics (mean LOOCV pixel error and visual degrees) print to the terminal.

---

## 5. Step 3: Training the Gaze Regression Model

Train an estimator mapping the extracted eye features to screen pixel coordinates:

```bash
ocular train --session my_session --model rf
```

Supported model architectures (`--model` / `-m`):
- `rf` (Random Forest, default): Handles non-linear eye-to-screen mappings and head pose drift effectively.
- `ridge` (Polynomial Ridge Regression): Fast, lightweight, and mathematically stable on smaller calibration grids.
- `svr` (Support Vector Regression): Radial basis function kernel for continuous curvature.
- `mlp` (Multi-Layer Perceptron): Neural network regressor (recommended for 16-point grids with abundant data).

Output:
Trained weight files and configurations are saved to `models/my_session/`.

---

## 6. Step 4: Live Gaze Interaction

Once a model is trained, launch the real-time interaction controller:

```bash
ocular interact --session my_session --model rf --mode all
```

### Available Interaction Modes (`--mode`):
- `all`: Enables the visual gaze pointer, dwell-click detection, and boundary scrolling.
- `cursor`: Gaze pointer only (no dwell selection).
- `dwell`: Dwell selection with visual progress rings.
- `scroll`: Peripheral window scrolling when reading documents.

### Controlling the Real Operating System Cursor
By default, OCULAR displays a diagnostic visualizer window. To enable actual mouse pointer takeover:
```bash
ocular interact --session my_session --model rf --enable-os-cursor
```

Safety features:
- PyAutoGUI failsafe is enabled: move your physical mouse forcefully to any corner of the screen to break cursor lock.
- Refractory cooldown: After a dwell selection triggers, a 1000 ms cooldown prevents accidental double clicks (mitigating the Midas Touch problem).
- Deliberate blink click: Keeping both eyes closed for 8 or more consecutive frames triggers a primary click event.

Press `q` to terminate the session.

---

## 7. Step 5: Automated Benchmarks and Profiling

### Model Comparison Benchmark
Evaluate all regression models (Ridge, SVR, Random Forest, MLP) against your recorded calibration session:
```bash
ocular benchmark
```
This runs Leave-One-Out Cross-Validation (LOOCV) and generates a comparative table showing mean pixel error, median error, 95th percentile error, and angular error in visual degrees. Results are written to `data/experiments/model_benchmark.json`.

### Pipeline Latency Profiling
Measure per-stage latency (camera read, landmark inference, feature extraction, model prediction, filtering):
```bash
python -c "
from ocular import Camera, FaceTracker, FeatureExtractor, GazeRegressor, PerformanceProfiler
profiler = PerformanceProfiler()
cam = Camera(0)
cam.open()
tracker = FaceTracker()
for _ in range(50):
    with profiler.measure('camera_read'):
        ret, frame = cam.read()
    if ret:
        with profiler.measure('tracker'):
            found = tracker.process(frame)
cam.release()
tracker.release()
profiler.report()
"
```

---

## 8. Step 6: Running Empirical Experiments

### Feature Subset Ablation Study
Evaluate the impact of different feature subsets (iris-only vs. iris + EAR vs. iris + head pose vs. full 11-D vector):
```bash
python -m experiments.feature_ablation --input calibration/my_session.npz --model rf
```
The script evaluates model accuracy across five feature subsets and prints a comparative ranking.

### Structured User Study Protocol
To conduct a controlled evaluation across participants:
```bash
python -m experiments.run_user_study --participant P01 --camera 0
```
This guides the participant through:
1. Conventional 9-point calibration.
2. 16-point held-out validation grid.
3. Model evaluation (Ridge and Random Forest).
4. Automated logging of structured metrics to `data/experiments/study_P01_<timestamp>.json`.

---

## 9. Troubleshooting and FAQs

### Problem: Camera fails to open (`Failed to open camera 0`)
- On Linux: Verify your user belongs to the `video` group:
  ```bash
  sudo usermod -aG video $USER
  ```
  Check available device nodes using `ls -l /dev/video*` or `v4l2-ctl --list-devices`.
- On macOS: Ensure your terminal emulator has camera permissions under System Settings > Privacy & Security > Camera.
- On Windows: Confirm no other application (Teams, Zoom, browser) is actively locking the webcam.

### Problem: High gaze jitter or erratic cursor jumps
- Increase the filtering strength by adjusting One Euro Filter parameters in `src/ocular/interaction.py` or `.env`.
- Check lighting: Uneven shadows across the bridge of your nose can degrade iris contour detection.
- Re-run calibration with `--dwell 2.0` to ensure stable fixations.

### Problem: `ModuleNotFoundError: No module named 'ocular'`
- Ensure your virtual environment is active.
- Re-run `pip install -e .` from the project root.
