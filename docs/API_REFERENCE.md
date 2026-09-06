# OCULAR: API Reference Manual

A comprehensive developer reference documenting all classes, methods, arguments, and return types within the `ocular` package.

---

## Table of Contents
1. [Camera Acquisition (`ocular.Camera`)](#1-camera-acquisition-ocularcamera)
2. [Face & Iris Tracking (`ocular.FaceTracker`)](#2-face--iris-tracking-ocularfacetracker)
3. [Feature Extraction (`ocular.FeatureExtractor`)](#3-feature-extraction-ocularfeatureextractor)
4. [Calibration Sessions (`ocular.CalibrationSession`)](#4-calibration-sessions-ocularcalibrationsession)
5. [Adaptive Calibration (`ocular.AdaptiveCalibrationEngine`)](#5-adaptive-calibration-ocularadaptivecalibrationengine)
6. [Online Model Adaptation (`ocular.OnlineAdaptiveRefiner`)](#6-online-model-adaptation-ocularonlineadaptiverefiner)
7. [Gaze Regression (`ocular.GazeRegressor`)](#7-gaze-regression-oculargazeregressor)
8. [Signal Filters (`ocular.OneEuroFilter`, `ocular.PointFilter2D`)](#8-signal-filters-ocularoneeurofilter-ocularpointfilter2d)
9. [Blink Detection (`ocular.BlinkDetector`)](#9-blink-detection-ocularblinkdetector)
10. [Interaction Controller (`ocular.InteractionController`)](#10-interaction-controller-ocularinteractioncontroller)
11. [Evaluation Framework (`ocular.EvaluationFramework`)](#11-evaluation-framework-ocularevaluationframework)
12. [Performance Profiler (`ocular.PerformanceProfiler`)](#12-performance-profiler-ocularperformanceprofiler)

---

## 1. Camera Acquisition (`ocular.Camera`)

```python
from ocular import Camera
```

### Constructor
```python
Camera(source=0, width=1280, height=720, fps=30)
```
* **Parameters:**
  * `source` *(int or str)*: Video device index (e.g. `0`) or V4L2 device path (e.g. `"/dev/video0"`).
  * `width` *(int)*: Desired capture frame width (default: `1280`).
  * `height` *(int)*: Desired capture frame height (default: `720`).
  * `fps` *(int)*: Desired frame rate (default: `30`).

### Methods
* `open() -> bool`: Initializes capture backend (V4L2 on Linux, DirectShow on Windows). Returns `True` if successfully opened.
* `is_opened() -> bool`: Returns `True` if camera stream is active and valid.
* `read() -> tuple[bool, numpy.ndarray]`: Reads next BGR frame.
* `release() -> None`: Releases underlying camera hardware handle.
* Context manager supported: `with Camera(0) as cam:` automatically handles open and release.

---

## 2. Face & Iris Tracking (`ocular.FaceTracker`)

```python
from ocular import FaceTracker
```

### Constructor
```python
FaceTracker(max_faces=1, detection_confidence=0.5, tracking_confidence=0.5, model_path=None)
```
* **Parameters:**
  * `max_faces` *(int)*: Maximum number of faces to detect (default: `1`).
  * `detection_confidence` *(float)*: Minimum detection threshold `[0.0, 1.0]`.
  * `tracking_confidence` *(float)*: Minimum tracking threshold `[0.0, 1.0]`.
  * `model_path` *(str, optional)*: Path to `face_landmarker.task`. If `None`, uses `models/face_landmarker.task` (auto-downloads if absent).

### Key Attributes
* `LEFT_IRIS_CENTER = 468`
* `RIGHT_IRIS_CENTER = 473`
* `LEFT_EYE_INNER = 133`, `LEFT_EYE_OUTER = 33`
* `RIGHT_EYE_INNER = 362`, `RIGHT_EYE_OUTER = 263`

### Methods
* `process(frame: numpy.ndarray) -> bool`: Ingests BGR image, extracts 478 landmarks. Returns `True` if face detected.
* `get_landmark_px(index: int, frame_shape: tuple) -> tuple[int, int] | None`: Returns $(x, y)$ integer pixel coordinate for landmark index.
* `get_landmark_normalized(index: int) -> tuple[float, float, float] | None`: Returns normalized $(x, y, z)$ coordinate.
* `get_iris_centers(frame_shape: tuple) -> tuple[tuple[int, int], tuple[int, int]]`: Returns `(left_iris_px, right_iris_px)`.
* `draw_debug_overlay(frame: numpy.ndarray) -> numpy.ndarray`: Draws iris centers, eyelid contours, and pose points.
* `release() -> None`: Closes MediaPipe pipeline.

---

## 3. Feature Extraction (`ocular.FeatureExtractor`)

```python
from ocular import FeatureExtractor
```

### Methods
* `extract(landmarks, frame_shape: tuple) -> numpy.ndarray | None`:  
  Computes normalized 11-element feature vector:
  ```python
  [
      left_iris_h_ratio, left_iris_v_ratio,
      right_iris_h_ratio, right_iris_v_ratio,
      left_ear, right_ear,
      left_eye_aspect, right_eye_aspect,
      head_pitch, head_yaw, head_roll
  ]
  ```
* `estimate_head_pose(landmarks, frame_shape: tuple) -> tuple[float, float, float, np.ndarray, np.ndarray]`:  
  Returns `(pitch_deg, yaw_deg, roll_deg, rvec, tvec)`.
* `draw_head_pose_axes(frame: np.ndarray, rvec, tvec, length=100) -> np.ndarray`:  
  Projects 3D RGB axes (Red=X, Green=Y, Blue=Z) from the nose tip.

---

## 4. Calibration Sessions (`ocular.CalibrationSession`)

```python
from ocular import CalibrationSession
```

### Constructor
```python
CalibrationSession(screen_w=1920, screen_h=1080)
```

### Static / Class Methods
* `get_screen_resolution() -> tuple[int, int]`: Auto-detects primary display dimensions.
* `generate_grid(screen_w, screen_h, pattern='9-point', margin=0.10) -> list[tuple[int, int]]`:  
  Generates target screen coordinates. `pattern` options: `'5-point'`, `'9-point'`, `'13-point'`, `'16-point'`.
* `save_session(filepath: str, X: np.ndarray, y: np.ndarray, metadata: dict = None) -> None`:  
  Serializes calibration dataset to compressed `.npz`.
* `load_session(filepath: str) -> tuple[np.ndarray, np.ndarray, dict]`:  
  Loads `X`, `y`, and metadata dictionary from `.npz`.

### Methods
* `run_interactive(camera, tracker, extractor, points=None, dwell_seconds=1.8, prep_seconds=0.6) -> tuple[np.ndarray, np.ndarray]`:  
  Runs full-screen interactive calibration UI. Filters blinks and saccades.

---

## 5. Adaptive Calibration (`ocular.AdaptiveCalibrationEngine`)

```python
from ocular import AdaptiveCalibrationEngine
```

### Constructor
```python
AdaptiveCalibrationEngine(screen_w=1920, screen_h=1080, strategy='hybrid')
```
* **Strategies:** `'hybrid'` (coverage + error), `'uncertainty'` (Gaussian Process), `'error'` (regional residuals).

### Methods
* `generate_candidate_pool(grid_rows=5, grid_cols=5, margin=0.10) -> list[tuple[int, int]]`:  
  Generates pool of candidate target coordinates.
* `select_next_target(existing_targets, existing_features, current_regressor, candidate_pool=None) -> tuple[int, int]`:  
  Returns the optimal next target screen point.
* `should_stop(loocv_errors, sample_count, max_budget=12, target_mae_px=60.0) -> tuple[bool, str]`:  
  Evaluates stopping criteria (budget exhaustion, accuracy reached, or diminishing returns).

---

## 6. Online Model Adaptation (`ocular.OnlineAdaptiveRefiner`)

```python
from ocular import OnlineAdaptiveRefiner
```

### Methods
* `warm_start(X_init: np.ndarray, y_init: np.ndarray) -> None`:  
  Initializes SGD regressors with initial calibration dataset.
* `partial_fit_interaction(feature_vector, true_screen_x, true_screen_y) -> None`:  
  Incrementally updates model weights based on observed user confirmation.
* `predict(feature_vector) -> tuple[float, float]`:  
  Predicts screen coordinates using adapted weights.

---

## 7. Gaze Regression (`ocular.GazeRegressor`)

```python
from ocular import GazeRegressor
```

### Constructor
```python
GazeRegressor(model_type='ridge', screen_w=1920, screen_h=1080)
```
* **Supported Model Types:** `'ridge'`, `'svr'`, `'rf'`, `'mlp'`.

### Methods
* `fit(X: np.ndarray, y: np.ndarray) -> None`: Fits coordinate regressors.
* `predict(X: np.ndarray) -> np.ndarray`: Predicts screen coordinates $(X, Y)$, bounded by display dimensions.
* `evaluate_loocv(X: np.ndarray, y: np.ndarray) -> dict`:  
  Runs Leave-One-Out Cross-Validation. Returns dictionary with `mean_error_px`, `median_error_px`, `p95_error_px`, `mean_error_deg`.
* `pixels_to_degrees(error_px, screen_w_px=1920, screen_w_cm=53.0, distance_cm=60.0) -> float`:  
  Converts pixel distance to degrees of visual angle.
* `save(directory='models') -> None` / `load(directory='models') -> None`:  
  Serializes / deserializes trained pipeline with Joblib.

---

## 8. Signal Filters (`ocular.OneEuroFilter`, `ocular.PointFilter2D`)

```python
from ocular import OneEuroFilter, PointFilter2D
```

### Constructor (`OneEuroFilter`)
```python
OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.1, dcutoff=1.0)
```
* `filter(x: float, t: float = None) -> float`: Filters single scalar value.
* `reset() -> None`: Clears filter state.

### Constructor (`PointFilter2D`)
```python
PointFilter2D(freq=30.0, mincutoff=1.0, beta=0.1, dcutoff=1.0)
```
* `filter(x: float, y: float, t: float = None) -> tuple[float, float]`: Filters 2D coordinate point $(x, y)$.

---

## 9. Blink Detection (`ocular.BlinkDetector`)

```python
from ocular import BlinkDetector
```

### Constructor
```python
BlinkDetector(ear_threshold=0.20, natural_min_frames=2, deliberate_min_frames=8, wink_discrepancy=0.08)
```

### Methods
* `update(left_ear: float, right_ear: float) -> str`:  
  Processes current frame EAR. Returns:
  - `"NONE"`
  - `"NATURAL_BLINK"` (involuntary blink: 2 to 7 frames)
  - `"DELIBERATE_BLINK"` (held blink: 8 or more frames)
  - `"LEFT_WINK"` / `"RIGHT_WINK"`
* `is_blinking(left_ear, right_ear) -> bool`: Returns `True` while eyelids are closed.

---

## 10. Interaction Controller (`ocular.InteractionController`)

```python
from ocular import InteractionController
```

### Constructor
```python
InteractionController(screen_w=1920, screen_h=1080, enable_os_cursor=False)
```

### Methods
* `process_frame(raw_pred_x, raw_pred_y, blink_event='NONE') -> dict`:  
  Smooths coordinates via 1€ filter, evaluates dwell progress, checks scrolling zones, and dispatches actions. Returns:
  ```python
  {
      'cursor_x': float,
      'cursor_y': float,
      'dwell_triggered': bool,
      'dwell_progress': float, # 0.0 to 1.0
      'scroll_speed': int,
      'blink_event': str
  }
  ```

---

## 11. Evaluation Framework (`ocular.EvaluationFramework`)

```python
from ocular import EvaluationFramework
```

### Methods
* `benchmark_models(X, y, models=None) -> dict`:  
  Runs LOOCV benchmark across candidate models and returns comparative error dictionary.
* `compare_conventional_vs_adaptive(X_conv, y_conv, conv_time, X_adapt, y_adapt, adapt_time) -> dict`:  
  Computes sample reduction ratio, time savings, and accuracy trade-offs.
* `save_experiment_results(filepath: str, report_data: dict) -> None`:  
  Saves evaluation metrics to JSON.

---

## 12. Performance Profiler (`ocular.PerformanceProfiler`)

```python
from ocular import PerformanceProfiler
```

### Constructor
```python
PerformanceProfiler(window_size=100)
```

### Methods
* `start(stage_name: str) -> None`: Begins timing for the specified pipeline stage.
* `stop(stage_name: str) -> float`: Ends timing and records duration in milliseconds.
* `measure(stage_name: str)`: Returns a context manager (`with profiler.measure("stage"):`) that times the enclosed block.
* `get_stats(stage_name: str) -> dict`: Returns `mean_ms`, `min_ms`, `max_ms`, and sample count for a stage.
* `get_all_stats() -> dict`: Returns summary statistics across all monitored stages.
* `get_total_ms() -> float`: Returns cumulative latency across stages.
* `get_fps() -> float`: Returns estimated throughput (frames per second) based on total latency.
* `report() -> None`: Prints a formatted breakdown of stage durations and overall FPS.
* `reset() -> None`: Clears all recorded timing samples.
