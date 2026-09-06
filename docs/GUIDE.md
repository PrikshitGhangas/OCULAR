# OCULAR: Comprehensive Implementation Guide

> **Purpose:** A step-by-step technical guide for building the OCULAR webcam-based
> gaze tracking and interaction system, from the current M1 state through to the
> final evaluation and deployment.

---

# TABLE OF CONTENTS

1. [Prerequisites & Dependencies](#1-prerequisites--dependencies)
2. [M1: Webcam Pipeline (COMPLETED)](#2-m1--webcam-pipeline-completed)
3. [M2: Face Detection & Eye/Iris Tracking](#3-m2--face-detection--eyeiris-tracking)
4. [M3: Feature Extraction](#4-m3--feature-extraction)
5. [M4: Calibration System](#5-m4--calibration-system)
6. [M5: Gaze Regression](#6-m5--gaze-regression)
7. [M6: Adaptive Calibration](#7-m6--adaptive-calibration)
8. [M7: Robustness Testing](#8-m7--robustness-testing)
9. [M8: Interaction Mechanisms](#9-m8--interaction-mechanisms)
10. [M9: Evaluation Framework](#10-m9--evaluation-framework)
11. [M10: Windows Deployment](#11-m10--windows-deployment)
12. [Appendix A: Face Detection Method Comparison](#appendix-a--face-detection-method-comparison)
13. [Appendix B: Complete Landmark Reference](#appendix-b--complete-landmark-reference)
14. [Appendix C: Mathematical Formulas](#appendix-c--mathematical-formulas)
15. [Appendix D: Recommended File Structure](#appendix-d--recommended-file-structure)

---

# 1. Prerequisites & Dependencies

## 1.1 Python Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux
# .venv\Scripts\activate    # Windows
```

## 1.2 Core Dependencies

```text
opencv-python>=4.8.0
mediapipe>=0.10.9
numpy>=1.24.0
scikit-learn>=1.3.0
```

## 1.3 Interaction & Utility Dependencies (install later, at M8)

```text
pyautogui>=0.9.54
pynput>=1.7.6
screeninfo>=0.8.1
```

## 1.4 Optional / Experimental

```text
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
```

## 1.5 Install Command

```bash
pip install opencv-python mediapipe numpy scikit-learn
```

Save to `requirements.txt` as you go. Start with the core dependencies and add
others when their milestone arrives.

---

# 2. M1: Webcam Pipeline (COMPLETED)

This milestone is already complete. The current implementation includes:

- `Camera` class in `src/ocular/camera.py` with `open()`, `read()`, `release()`
- V4L2 configuration for Linux (exposure, MJPG)
- 1280×720 at ~28 FPS measured throughput
- Basic image processing experiments (grayscale, thresholding, cropping)

### What to verify before moving on

```text
[x] Camera opens successfully
[x] Frames are read at expected resolution
[x] Live display works via cv2.imshow
[x] Camera releases cleanly on exit
[x] FPS is approximately 28 at 1280×720
```

### Known issue to keep in mind

The `Camera.open()` method uses `v4l2-ctl` which is Linux-specific. When
targeting Windows (M10), this call must be skipped or wrapped in a platform
check. The rest of the camera abstraction (`cv2.VideoCapture`) is
cross-platform.

---

# 3. M2: Face Detection & Eye/Iris Tracking

This is the **next major milestone** and the most critical decision point.

## 3.1 Choosing a Tracking Framework

Three options were researched. Here is the recommendation:

| Feature | YuNet | MediaPipe Face Mesh | dlib (68-point) |
|---------|-------|---------------------|-----------------|
| **Landmarks** | 5 | 478 (with iris) | 68 |
| **Iris Detection** | No | Direct | No |
| **Speed (CPU)** | ~2-5ms  | ~10-30ms | ~35-60ms |
| **Eye Contour** | Center only | Full contour | 6 points/eye |
| **3D Coordinates** | No | Yes | No |
| **Install** | Built into OpenCV | `pip install mediapipe` | Needs C++ compiler |
| **Windows Support** | Yes | Yes | Yes |

### Recommendation: MediaPipe Face Mesh

**Why:**

1. **Iris landmarks are essential.** MediaPipe provides direct iris center
   coordinates (landmarks 468, 473): no custom CV needed.
2. **478 landmarks** give rich eye contour, face geometry, and head pose data.
3. **3D coordinates** are included for every landmark.
4. **Cross-platform**: works on Linux and Windows without code changes.
5. **~10-30ms** is fast enough for real-time at 30 FPS.

**Keep YuNet** as a fallback and for performance comparison experiments.

## 3.2 Step-by-Step: Integrating MediaPipe

### Step 1: Install MediaPipe

```bash
pip install mediapipe
```

### Step 2: Create the Tracker Module

Create `src/ocular/tracker.py`:

```python
import cv2
import mediapipe as mp
import numpy as np


class FaceTracker:
    """
    Wraps MediaPipe Face Mesh to provide face, eye, and iris
    landmark tracking from webcam frames.
    """

    # --- Landmark Indices ---
    # Iris centers
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Iris boundary points
    LEFT_IRIS_RING = [469, 470, 471, 472]
    RIGHT_IRIS_RING = [474, 475, 476, 477]

    # Eye corners
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Eyelids (for vertical ratio and EAR)
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # EAR landmarks (6 points per eye)
    LEFT_EYE_EAR = {
        'p1': 33,   # outer corner
        'p2': 160,  # upper outer
        'p3': 158,  # upper inner
        'p4': 133,  # inner corner
        'p5': 153,  # lower inner
        'p6': 144,  # lower outer
    }
    RIGHT_EYE_EAR = {
        'p1': 362,  # inner corner
        'p2': 385,  # upper inner
        'p3': 387,  # upper outer
        'p4': 263,  # outer corner
        'p5': 380,  # lower outer
        'p6': 373,  # lower inner
    }

    # Head pose reference landmarks
    POSE_LANDMARKS = {
        'nose_tip': 1,
        'chin': 152,
        'left_eye_outer': 33,
        'right_eye_outer': 263,
        'left_mouth': 61,
        'right_mouth': 291,
    }

    def __init__(self, max_faces=1, detection_confidence=0.5,
                 tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,     # CRITICAL: enables iris landmarks
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarks = None

    def process(self, frame):
        """
        Process a BGR frame.
        Returns True if a face was detected, False otherwise.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            self.landmarks = results.multi_face_landmarks[0]
            return True

        self.landmarks = None
        return False

    def get_landmark_px(self, index, frame_shape):
        """Get a landmark as pixel coordinates (x, y)."""
        if self.landmarks is None:
            return None
        lm = self.landmarks.landmark[index]
        h, w = frame_shape[:2]
        return (int(lm.x * w), int(lm.y * h))

    def get_landmark_normalized(self, index):
        """Get a landmark as normalized coordinates (0-1)."""
        if self.landmarks is None:
            return None
        lm = self.landmarks.landmark[index]
        return (lm.x, lm.y, lm.z)

    def get_iris_centers(self, frame_shape):
        """
        Returns pixel coordinates of both iris centers.
        Returns (left_iris, right_iris) or (None, None).
        """
        if self.landmarks is None:
            return None, None
        left = self.get_landmark_px(self.LEFT_IRIS_CENTER, frame_shape)
        right = self.get_landmark_px(self.RIGHT_IRIS_CENTER, frame_shape)
        return left, right

    def release(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()
```

### Step 3: Create a Test Script

Create `src/ocular/trackerTest.py`:

```python
import cv2
import time
from camera import Camera
from tracker import FaceTracker

camera = Camera("/dev/video0")

if not camera.open():
    print("Failed to open camera")
    exit()

tracker = FaceTracker()

frames = 0
start = time.perf_counter()

while True:
    success, frame = camera.read()
    if not success:
        break

    found = tracker.process(frame)

    if found:
        h, w = frame.shape[:2]

        # Draw iris centers
        left_iris, right_iris = tracker.get_iris_centers(frame.shape)
        if left_iris:
            cv2.circle(frame, left_iris, 3, (0, 255, 0), -1)
        if right_iris:
            cv2.circle(frame, right_iris, 3, (0, 255, 0), -1)

        # Draw eye corners
        for idx in [tracker.LEFT_EYE_INNER, tracker.LEFT_EYE_OUTER,
                    tracker.RIGHT_EYE_INNER, tracker.RIGHT_EYE_OUTER]:
            pt = tracker.get_landmark_px(idx, frame.shape)
            if pt:
                cv2.circle(frame, pt, 2, (255, 255, 0), -1)

    frames += 1

    cv2.imshow("OCULAR - Tracker Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end = time.perf_counter()

camera.release()
tracker.release()
cv2.destroyAllWindows()

elapsed = end - start
print()
print(f"Frames: {frames}")
print(f"Total time: {elapsed:.3f} s")
print(f"FPS: {frames / elapsed:.1f}")
```

### Step 4: Verify the Tracker

Run the test and confirm:

```text
[x] Green dots appear on both iris centers
[x] Yellow dots appear on eye corners
[x] Tracking is stable and follows eye movement
[x] FPS remains above 20
[x] Landmarks track correctly when head moves
```

### Step 5: Investigate the Existing "Red Circles" Issue

The README mentions "two red circles appearing and disappearing." Now that you
have the tracker, check if this comes from the existing `eyeTest.py` contour-
based pupil detection (`pupil.py`). Compare MediaPipe's iris tracking stability
against the contour approach: this will help justify the framework choice in
your report.

---

# 4. M3: Feature Extraction

Once the tracker is stable, build a feature extraction pipeline that converts
raw landmarks into numerical features suitable for regression.

## 4.1 The Feature Vector

The recommended feature vector contains **11 features**:

| # | Feature | Range | Description |
|---|---------|-------|-------------|
| 1 | `left_iris_h_ratio` | 0.0 - 1.0 | Left iris horizontal position between eye corners |
| 2 | `left_iris_v_ratio` | 0.0 - 1.0 | Left iris vertical position between eyelids |
| 3 | `right_iris_h_ratio` | 0.0 - 1.0 | Right iris horizontal position |
| 4 | `right_iris_v_ratio` | 0.0 - 1.0 | Right iris vertical position |
| 5 | `left_ear` | 0.0 - 0.5 | Left Eye Aspect Ratio (openness) |
| 6 | `right_ear` | 0.0 - 0.5 | Right Eye Aspect Ratio (openness) |
| 7 | `left_eye_aspect` | ~1.0 - 5.0 | Left eye width / height |
| 8 | `right_eye_aspect` | ~1.0 - 5.0 | Right eye width / height |
| 9 | `head_pitch` | degrees | Head tilt up/down |
| 10 | `head_yaw` | degrees | Head turn left/right |
| 11 | `head_roll` | degrees | Head tilt sideways |

## 4.2 Iris Position Ratios

These are the **most important features** for gaze estimation.

### Horizontal Ratio

```
ratio_h = (iris_x - outer_corner_x) / (inner_corner_x - outer_corner_x)
```

- **≈ 0.5:** Looking straight ahead
- **< 0.5:** Looking toward the outer corner (away from nose)
- **> 0.5:** Looking toward the inner corner (toward nose)

### Vertical Ratio

```
ratio_v = (iris_y - top_eyelid_y) / (bottom_eyelid_y - top_eyelid_y)
```

- **≈ 0.5:** Looking straight ahead vertically
- **< 0.5:** Looking up
- **> 0.5:** Looking down

## 4.3 Eye Aspect Ratio (EAR)

Used for blink detection and as a feature for the gaze model.

```
        p2    p3
  p1                p4
        p6    p5

EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

- **Open eye:** EAR ≈ 0.20 - 0.35
- **Closed/blink:** EAR < 0.20

## 4.4 Head Pose Estimation (solvePnP)

Head pose is **critical** because head rotation changes the apparent iris
position even when the user is looking at the same screen point.

### 3D Face Model Points (Generic)

```python
model_points = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye outer corner
    (225.0, 170.0, -135.0),     # Right eye outer corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0),    # Right mouth corner
], dtype=np.float64)
```

### Corresponding MediaPipe Landmark Indices

```python
POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
```

### Camera Matrix (Approximate, Without Calibration)

```python
def get_camera_matrix(frame_shape):
    h, w = frame_shape[:2]
    focal_length = w  # Approximation
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs
```

### Computing Pitch, Yaw, Roll

```python
def estimate_head_pose(landmarks, frame_shape):
    h, w = frame_shape[:2]

    image_points = np.array([
        (landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h)
        for idx in [1, 152, 33, 263, 61, 291]
    ], dtype=np.float64)

    camera_matrix, dist_coeffs = get_camera_matrix(frame_shape)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_points, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    proj_matrix = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
        np.vstack((proj_matrix, [0, 0, 0, 1]))[:3]
    )

    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]
    return pitch, yaw, roll
```

## 4.5 Step-by-Step: Building the Feature Extractor

### Step 1: Create `src/ocular/features.py`

```python
import cv2
import numpy as np


class FeatureExtractor:
    """
    Extracts a numerical feature vector from MediaPipe
    face landmarks for gaze regression.
    """

    # 3D model points for head pose
    MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)

    POSE_INDICES = [1, 152, 33, 263, 61, 291]

    # Feature names (for logging/debugging)
    FEATURE_NAMES = [
        'left_iris_h_ratio',
        'left_iris_v_ratio',
        'right_iris_h_ratio',
        'right_iris_v_ratio',
        'left_ear',
        'right_ear',
        'left_eye_aspect',
        'right_eye_aspect',
        'head_pitch',
        'head_yaw',
        'head_roll',
    ]

    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs = None

    def extract(self, landmarks, frame_shape):
        """
        Extract an 11-element feature vector.

        Args:
            landmarks: MediaPipe face_landmarks object
            frame_shape: (height, width, channels)

        Returns:
            numpy array of shape (11,) or None if extraction fails
        """
        h, w = frame_shape[:2]

        try:
            # Iris ratios
            l_h, l_v = self._iris_ratios(landmarks, w, h, 'left')
            r_h, r_v = self._iris_ratios(landmarks, w, h, 'right')

            # Eye aspect ratios
            l_ear = self._ear(landmarks, w, h, 'left')
            r_ear = self._ear(landmarks, w, h, 'right')

            # Eye geometry
            l_aspect = self._eye_aspect(landmarks, w, h, 'left')
            r_aspect = self._eye_aspect(landmarks, w, h, 'right')

            # Head pose
            pitch, yaw, roll = self._head_pose(landmarks, frame_shape)

            return np.array([
                l_h, l_v, r_h, r_v,
                l_ear, r_ear,
                l_aspect, r_aspect,
                pitch, yaw, roll,
            ])

        except Exception:
            return None

    def _pt(self, landmarks, idx, w, h):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])

    def _iris_ratios(self, landmarks, w, h, side):
        if side == 'left':
            iris = self._pt(landmarks, 468, w, h)
            inner = self._pt(landmarks, 133, w, h)
            outer = self._pt(landmarks, 33, w, h)
            top = self._pt(landmarks, 159, w, h)
            bottom = self._pt(landmarks, 145, w, h)
        else:
            iris = self._pt(landmarks, 473, w, h)
            inner = self._pt(landmarks, 362, w, h)
            outer = self._pt(landmarks, 263, w, h)
            top = self._pt(landmarks, 386, w, h)
            bottom = self._pt(landmarks, 374, w, h)

        eye_w = np.linalg.norm(inner - outer)
        eye_h = np.linalg.norm(bottom - top)

        h_ratio = ((iris[0] - outer[0]) / eye_w) if eye_w > 1 else 0.5
        v_ratio = ((iris[1] - top[1]) / eye_h) if eye_h > 1 else 0.5

        return (
            np.clip(h_ratio, 0.0, 1.0),
            np.clip(v_ratio, 0.0, 1.0),
        )

    def _ear(self, landmarks, w, h, side):
        if side == 'left':
            indices = {'p1': 33, 'p2': 160, 'p3': 158,
                       'p4': 133, 'p5': 153, 'p6': 144}
        else:
            indices = {'p1': 362, 'p2': 385, 'p3': 387,
                       'p4': 263, 'p5': 380, 'p6': 373}

        pts = {k: self._pt(landmarks, v, w, h) for k, v in indices.items()}

        v1 = np.linalg.norm(pts['p2'] - pts['p6'])
        v2 = np.linalg.norm(pts['p3'] - pts['p5'])
        horiz = np.linalg.norm(pts['p1'] - pts['p4'])

        if horiz < 1:
            return 0.0

        return (v1 + v2) / (2.0 * horiz)

    def _eye_aspect(self, landmarks, w, h, side):
        if side == 'left':
            inner = self._pt(landmarks, 133, w, h)
            outer = self._pt(landmarks, 33, w, h)
            top = self._pt(landmarks, 159, w, h)
            bottom = self._pt(landmarks, 145, w, h)
        else:
            inner = self._pt(landmarks, 362, w, h)
            outer = self._pt(landmarks, 263, w, h)
            top = self._pt(landmarks, 386, w, h)
            bottom = self._pt(landmarks, 374, w, h)

        eye_w = np.linalg.norm(inner - outer)
        eye_h = np.linalg.norm(bottom - top)

        if eye_h < 1:
            return 1.0

        return eye_w / eye_h

    def _head_pose(self, landmarks, frame_shape):
        h, w = frame_shape[:2]

        if self._camera_matrix is None:
            focal = w
            center = (w / 2.0, h / 2.0)
            self._camera_matrix = np.array([
                [focal, 0, center[0]],
                [0, focal, center[1]],
                [0, 0, 1],
            ], dtype=np.float64)
            self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = np.array([
            (landmarks.landmark[idx].x * w,
             landmarks.landmark[idx].y * h)
            for idx in self.POSE_INDICES
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS, image_points,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        proj = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(
            np.vstack((proj, [0, 0, 0, 1]))[:3]
        )

        return euler[0][0], euler[1][0], euler[2][0]
```

### Step 2: Test Feature Extraction

Create a quick test that prints feature values in real time to verify they
change sensibly when you move your eyes and head.

### Step 3: Log Features to CSV

Before moving to calibration, log features to a CSV file so you can inspect
and visualize the data:

```python
import csv

with open('data/feature_log.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(FeatureExtractor.FEATURE_NAMES)

    # In your main loop:
    features = extractor.extract(tracker.landmarks, frame.shape)
    if features is not None:
        writer.writerow(features.tolist())
```

### Step 4: Verify Feature Behavior

Check that:

```text
[x] Iris h_ratio changes when you look left/right
[x] Iris v_ratio changes when you look up/down
[x] Both eyes' ratios move roughly together
[x] EAR drops dramatically during blinks
[x] Head pose angles change correctly with head movement
[x] Features are stable (not wildly jittery) when holding still
```

---

# 5. M4: Calibration System

Calibration collects paired data: **(features, known_screen_position)**.

## 5.1 Calibration Grid Design

### Standard Grids

| Points | Layout | Notes |
|--------|--------|-------|
| 5 | Cross (center + 4 edges) | Minimal. Likely insufficient. |
| 9 | 3×3 grid | **Recommended starting point.** |
| 13 | 3×3 + 4 edge midpoints | Better coverage. |
| 16 | 4×4 grid | Good for experiments. |
| 25 | 5×5 grid | Thorough for research. |

### Generating Grid Points

```python
def generate_calibration_grid(screen_w, screen_h, rows=3, cols=3,
                               margin=0.1):
    """
    Generate calibration target positions with margin from edges.
    """
    points = []
    x_min = int(screen_w * margin)
    x_max = int(screen_w * (1 - margin))
    y_min = int(screen_h * margin)
    y_max = int(screen_h * (1 - margin))

    for r in range(rows):
        for c in range(cols):
            x = x_min + (x_max - x_min) * c // max(cols - 1, 1)
            y = y_min + (y_max - y_min) * r // max(rows - 1, 1)
            points.append((x, y))

    return points
```

## 5.2 Calibration Procedure

```text
For each calibration point:
    1. Display a target dot/circle at the known screen position.
    2. Animate it (e.g., shrinking circle) to cue fixation.
    3. Wait 1-2 seconds for the user to fixate.
    4. Capture 15-30 frames of features during the latter half
       of the dwell (to avoid saccade noise).
    5. Filter out blink frames (EAR < 0.2).
    6. Average the remaining feature vectors.
    7. Store: (averaged_features, target_x, target_y).
    8. Move to the next point.
```

## 5.3 Calibration Data Format

```python
# Each calibration sample:
{
    'features': np.array([...]),   # shape (11,)
    'target_x': int,               # screen X coordinate
    'target_y': int,               # screen Y coordinate
    'timestamp': float,            # when collected
    'n_frames': int,               # how many frames averaged
}
```

Save calibration data to a JSON or NPZ file so it can be reloaded:

```python
np.savez('calibration/user_session.npz',
         features=features_array,    # shape (N, 11)
         targets=targets_array)      # shape (N, 2)
```

## 5.4 Implementing the Calibration UI

Use a **fullscreen OpenCV window** for the simplest cross-platform approach:

```python
import cv2
import numpy as np

def run_calibration(screen_w, screen_h, points, camera, tracker,
                    extractor, dwell_seconds=2.0, sample_frames=20):
    """
    Run a calibration procedure.

    Returns:
        features_list: list of averaged feature vectors
        targets_list: list of (x, y) screen positions
    """
    features_list = []
    targets_list = []

    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    for (tx, ty) in points:
        # Draw target
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(canvas, (tx, ty), 30, (0, 255, 0), 2)
        cv2.circle(canvas, (tx, ty), 5, (0, 255, 0), -1)
        cv2.imshow("Calibration", canvas)
        cv2.waitKey(500)  # Brief pause before collection

        # Collect features
        collected = []
        for _ in range(sample_frames):
            success, frame = camera.read()
            if not success:
                continue

            found = tracker.process(frame)
            if not found:
                continue

            features = extractor.extract(tracker.landmarks, frame.shape)
            if features is None:
                continue

            # Skip blink frames
            left_ear = features[4]
            right_ear = features[5]
            if left_ear < 0.2 or right_ear < 0.2:
                continue

            collected.append(features)
            cv2.waitKey(1)

        if len(collected) >= 5:  # Need at least 5 valid frames
            avg_features = np.mean(collected, axis=0)
            features_list.append(avg_features)
            targets_list.append((tx, ty))

    cv2.destroyWindow("Calibration")
    return features_list, targets_list
```

## 5.5 Handling Outliers

- **Blink filter:** Already handled above (EAR < 0.2).
- **Temporal filter:** Discard feature vectors that differ wildly from the
  median of the collection window.
- **Robust regression:** Use Ridge or Huber regression which are less
  sensitive to outliers.

---

# 6. M5: Gaze Regression

## 6.1 The Regression Task

```text
Input:  Feature vector (11 features)
Output: Screen coordinates (X, Y)
```

Train **separate models** for X and Y predictions: this often works better
because horizontal and vertical gaze depend on different feature subsets.

## 6.2 Model Progression (Simple → Complex)

Follow this order. Only move to the next model if experiments show the
previous one is insufficient.

### Level 1: Ridge Regression with Polynomial Features (START HERE)

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

model_x = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0)),
])

model_y = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('ridge', Ridge(alpha=1.0)),
])

# X_train: shape (N, 11) : feature vectors
# y_train_x: shape (N,)  : screen X coordinates
# y_train_y: shape (N,)  : screen Y coordinates

model_x.fit(X_train, y_train_x)
model_y.fit(X_train, y_train_y)

# Prediction
pred_x = model_x.predict(X_new)
pred_y = model_y.predict(X_new)
```

**Why start here:** Ridge with polynomial features handles non-linearity while
preventing overfitting with limited calibration data (9-16 points). This is
likely good enough for a usable system.

### Level 2: SVR (Support Vector Regression)

```python
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model_x = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)),
])
```

**When to try:** If Ridge shows systematic errors (e.g., consistently wrong at
screen edges).

### Level 3: Random Forest

```python
from sklearn.ensemble import RandomForestRegressor

model_x = RandomForestRegressor(
    n_estimators=100, max_depth=5, min_samples_leaf=2,
    random_state=42
)
```

**Bonus:** Provides `feature_importances_`: useful for understanding which
features matter most.

### Level 4: Small MLP (Only If Justified)

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model_x = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        learning_rate='adaptive',
        random_state=42,
    )),
])
```

**Only use if:** You have significantly more calibration data (25+ points with
multiple samples each) and the simpler models plateau.

## 6.3 Model Evaluation

### Leave-One-Out Cross-Validation (During Calibration)

With only 9-16 calibration points, use LOOCV:

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import numpy as np

loo = LeaveOneOut()
errors = []

for train_idx, test_idx in loo.split(X):
    model_x.fit(X[train_idx], y_x[train_idx])
    model_y.fit(X[train_idx], y_y[train_idx])

    pred_x = model_x.predict(X[test_idx])
    pred_y = model_y.predict(X[test_idx])

    error = np.sqrt((pred_x - y_x[test_idx])**2 +
                    (pred_y - y_y[test_idx])**2)
    errors.append(error[0])

print(f"Mean error: {np.mean(errors):.1f} px")
print(f"Median error: {np.median(errors):.1f} px")
print(f"95th percentile: {np.percentile(errors, 95):.1f} px")
```

### Error Metrics

| Metric | What It Tells You |
|--------|-------------------|
| Mean error (px) | Overall average accuracy |
| Median error (px) | Typical accuracy (ignores outliers) |
| 95th percentile (px) | Worst-case behavior |
| Error in degrees | Physical angle: gold standard for research |

### Converting Pixels to Degrees

```python
def pixels_to_degrees(error_px, screen_w_px, screen_w_cm, distance_cm):
    """
    Convert pixel error to degrees of visual angle.
    """
    cm_per_px = screen_w_cm / screen_w_px
    error_cm = error_px * cm_per_px
    degrees = np.degrees(np.arctan(error_cm / distance_cm))
    return degrees
```

For reference: **< 2 degrees** is considered good for webcam gaze tracking.

## 6.4 Saving and Loading Models

```python
import joblib

# Save
joblib.dump(model_x, 'models/gaze_model_x.pkl')
joblib.dump(model_y, 'models/gaze_model_y.pkl')

# Load
model_x = joblib.load('models/gaze_model_x.pkl')
model_y = joblib.load('models/gaze_model_y.pkl')
```

---

# 7. M6: Adaptive Calibration

This is the **primary research contribution** of OCULAR.

## 7.1 The Core Idea

Instead of collecting the same number of calibration samples everywhere,
**intelligently select where to collect additional samples** based on model
performance.

```text
1. Perform initial calibration (e.g., 5 points)
         ↓
2. Train baseline model
         ↓
3. Predict gaze on validation points
         ↓
4. Identify weak regions (high error / high uncertainty)
         ↓
5. Request additional sample in the weak region
         ↓
6. Update model
         ↓
7. Repeat until budget is spent or accuracy is sufficient
```

## 7.2 Adaptive Strategies

### Strategy 1: Error-Driven Sampling

```python
def select_next_point_by_error(model_x, model_y, candidate_points,
                                X_calib, y_x, y_y):
    """
    Select the candidate calibration point where prediction
    error is expected to be highest.
    """
    # Use LOO to estimate error at each existing calibration point
    errors_by_region = {}

    # Divide screen into grid regions
    for point in candidate_points:
        # Find nearest existing calibration point
        # Estimate error from LOO residuals in that region
        # Select the point in the highest-error region
        pass

    return worst_region_point
```

### Strategy 2: Uncertainty-Based Sampling

Use a model that provides prediction variance:

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = RBF() + WhiteKernel()
gp_x = GaussianProcessRegressor(kernel=kernel)
gp_x.fit(X_calib, y_x)

# Predict with uncertainty
pred_x, std_x = gp_x.predict(X_candidate, return_std=True)

# Select the candidate with highest uncertainty
next_point_idx = np.argmax(std_x)
```

### Strategy 3: Coverage + Error

```python
def select_next_point_coverage_error(screen_w, screen_h,
                                      existing_points, errors,
                                      grid_rows=4, grid_cols=4):
    """
    Combine coverage gaps and prediction errors.
    Divide screen into zones. Score each zone by:
      score = coverage_weight * (1 / sample_count) +
              error_weight * regional_error
    Select the zone with the highest score.
    """
    zone_w = screen_w // grid_cols
    zone_h = screen_h // grid_rows

    best_score = -1
    best_zone = None

    for r in range(grid_rows):
        for c in range(grid_cols):
            zone_x = c * zone_w + zone_w // 2
            zone_y = r * zone_h + zone_h // 2

            # Count existing samples in this zone
            count = sum(1 for p in existing_points
                       if abs(p[0] - zone_x) < zone_w // 2
                       and abs(p[1] - zone_y) < zone_h // 2)

            # Average error in this zone
            zone_errors = [e for p, e in zip(existing_points, errors)
                          if abs(p[0] - zone_x) < zone_w
                          and abs(p[1] - zone_y) < zone_h]
            avg_error = np.mean(zone_errors) if zone_errors else 100

            # Score: penalize low coverage and high error
            coverage_score = 1.0 / (count + 1)
            score = 0.5 * coverage_score + 0.5 * (avg_error / 100)

            if score > best_score:
                best_score = score
                best_zone = (zone_x, zone_y)

    return best_zone
```

### Strategy 4: Budget / Confidence Stopping

```python
def should_stop_calibration(errors, max_samples, target_error=50):
    """
    Decide whether to stop calibration.
    """
    if len(errors) >= max_samples:
        return True  # Budget exhausted

    if np.mean(errors) < target_error:
        return True  # Accuracy sufficient

    # Check diminishing returns
    if len(errors) > 5:
        recent = np.mean(errors[-3:])
        earlier = np.mean(errors[-6:-3])
        if earlier - recent < 5:  # Less than 5px improvement
            return True

    return False
```

## 7.3 Online Model Updating

For incremental updates without full retraining:

```python
from sklearn.linear_model import SGDRegressor

model_x = SGDRegressor(
    learning_rate='constant',
    eta0=0.01,
    penalty='l2',
    alpha=0.001,
)

# Initial training
model_x.partial_fit(X_initial, y_x_initial)

# Incremental update with new sample
model_x.partial_fit(X_new_sample, y_x_new_sample)
```

## 7.4 The Experiment: Conventional vs Adaptive

```text
             Same User / Same Session
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
    Conventional             Adaptive
    (e.g., 9 points)        (5 initial + up to 4 adaptive)
            │                   │
            ▼                   ▼
       Train Model          Train + Update Model
            │                   │
            └─────────┬─────────┘
                      ▼
                 Evaluate Both
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        Accuracy    Time     Reliability
```

### What to Measure

| Metric | Conventional | Adaptive |
|--------|-------------|----------|
| Mean gaze error (px) | ? | ? |
| Median gaze error (px) | ? | ? |
| 95th percentile error (px) | ? | ? |
| Calibration time (s) | ? | ? |
| Number of samples | Fixed (9) | Variable |
| User burden (subjective) | ? | ? |

---

# 8. M7: Robustness Testing

## 8.1 Test Conditions

Test the system under varying conditions to measure degradation:

| Condition | Variations |
|-----------|-----------|
| **Lighting** | Bright overhead, dim room, backlit (window behind user), side-lit |
| **Head position** | Centered, leaned left, leaned right, leaned forward, leaned back |
| **Distance** | 40cm, 60cm (normal), 80cm from camera |
| **Users** | At least 3-5 different people |
| **Glasses** | With and without glasses/contacts |

## 8.2 Test Protocol

For each condition:

1. Calibrate the system under that condition.
2. Display 9-16 test points (different from calibration points).
3. Record predicted vs actual positions.
4. Compute error metrics.
5. Compare against the baseline (normal lighting, normal distance, centered).

## 8.3 Recording Results

```python
# Store experiment results
experiment = {
    'condition': 'dim_lighting',
    'user': 'user_01',
    'distance_cm': 60,
    'calibration_points': 9,
    'test_points': 16,
    'mean_error_px': 0.0,
    'median_error_px': 0.0,
    'p95_error_px': 0.0,
    'fps': 0.0,
}
```

---

# 9. M8: Interaction Mechanisms

## 9.1 Temporal Filtering (REQUIRED Before Any Interaction)

Raw gaze estimates are noisy. You **must** filter before using gaze for
interaction.

### Recommendation: One Euro Filter (1€ Filter)

Specifically designed for noisy HCI input. Adapts its cutoff frequency based on
movement speed: low jitter when still, low lag when moving.

```python
import math
import time


class OneEuroFilter:
    """
    One Euro Filter for smoothing noisy gaze signals.

    Parameters:
        freq:       Expected signal frequency (e.g., 30 for 30 FPS)
        mincutoff:  Minimum cutoff frequency (lower = less jitter)
        beta:       Speed coefficient (higher = less lag during movement)
        dcutoff:    Cutoff for derivative computation
    """

    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.1, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t=None):
        if t is None:
            t = time.time()

        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = t - self.t_prev
        if dt > 0:
            self.freq = 1.0 / dt

        # Derivative
        dx = (x - self.x_prev) * self.freq
        a_d = self._smoothing_factor(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # Adaptive cutoff
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(cutoff)

        # Filtered value
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


# Usage:
filter_x = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.1)
filter_y = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.1)

# In main loop:
smoothed_x = filter_x(raw_predicted_x)
smoothed_y = filter_y(raw_predicted_y)
```

### Parameter Tuning

| Parameter | Effect | Starting Value |
|-----------|--------|---------------|
| `mincutoff` | Lower = less jitter, more lag | 1.0 |
| `beta` | Higher = less lag during fast movement | 0.1 |
| `dcutoff` | Usually leave at default | 1.0 |

### Alternative Filters

| Filter | Pros | Cons |
|--------|------|------|
| Moving Average | Simple | Fixed lag |
| EMA | Responsive | Jitter vs lag tradeoff |
| Kalman | Predictive | Complex to tune |
| **1€ Filter** | **Adaptive, designed for HCI** | **Recommended** |

## 9.2 Gaze Cursor

```python
import pyautogui

pyautogui.PAUSE = 0         # Remove delay between calls
pyautogui.FAILSAFE = True   # Move mouse to corner to abort

# In main loop (after filtering):
pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
```

### Magnetism / Snapping (Optional)

```python
def apply_magnetism(gaze_x, gaze_y, ui_targets, snap_radius=50):
    """
    Snap gaze cursor to nearby UI targets.
    """
    for (tx, ty) in ui_targets:
        dist = math.sqrt((gaze_x - tx)**2 + (gaze_y - ty)**2)
        if dist < snap_radius:
            return tx, ty
    return gaze_x, gaze_y
```

## 9.3 Dwell Selection

```python
class DwellDetector:
    """
    Detect when gaze dwells on a target for a specified duration.
    """

    def __init__(self, threshold_ms=500, radius_px=50, cooldown_ms=1000):
        self.threshold_ms = threshold_ms
        self.radius_px = radius_px
        self.cooldown_ms = cooldown_ms
        self.dwell_start = None
        self.dwell_center = None
        self.last_activation = 0

    def update(self, gaze_x, gaze_y, current_time_ms):
        """
        Returns (activated, progress) where:
          activated: True if dwell threshold was reached
          progress:  0.0 to 1.0 indicating dwell progress
        """
        # Cooldown check
        if current_time_ms - self.last_activation < self.cooldown_ms:
            return False, 0.0

        if self.dwell_center is None:
            self.dwell_center = (gaze_x, gaze_y)
            self.dwell_start = current_time_ms
            return False, 0.0

        # Check if gaze is still within radius
        dist = math.sqrt((gaze_x - self.dwell_center[0])**2 +
                         (gaze_y - self.dwell_center[1])**2)

        if dist > self.radius_px:
            # Gaze moved: reset
            self.dwell_center = (gaze_x, gaze_y)
            self.dwell_start = current_time_ms
            return False, 0.0

        elapsed = current_time_ms - self.dwell_start
        progress = min(elapsed / self.threshold_ms, 1.0)

        if elapsed >= self.threshold_ms:
            self.last_activation = current_time_ms
            self.dwell_center = None
            self.dwell_start = None
            return True, 1.0

        return False, progress
```

## 9.4 Gaze Scrolling

```python
def gaze_scroll(gaze_y, screen_h, scroll_zone=0.2, max_speed=5):
    """
    Scroll based on vertical gaze position.

    Args:
        gaze_y: Current filtered gaze Y position
        screen_h: Screen height in pixels
        scroll_zone: Fraction of screen that triggers scrolling
        max_speed: Maximum scroll speed (lines per call)
    """
    top_threshold = screen_h * scroll_zone
    bottom_threshold = screen_h * (1 - scroll_zone)

    if gaze_y < top_threshold:
        # Looking near top: scroll up
        speed = int(max_speed * (1 - gaze_y / top_threshold))
        pyautogui.scroll(speed)

    elif gaze_y > bottom_threshold:
        # Looking near bottom: scroll down
        speed = int(max_speed *
                    (gaze_y - bottom_threshold) /
                    (screen_h - bottom_threshold))
        pyautogui.scroll(-speed)
```

## 9.5 Blink Detection (for Intentional Gestures)

```python
class BlinkDetector:
    """
    Detect deliberate blinks vs natural blinks using EAR.
    """

    def __init__(self, ear_threshold=0.2, natural_frames=4,
                 deliberate_frames=10):
        self.ear_threshold = ear_threshold
        self.natural_frames = natural_frames   # ~133ms at 30 FPS
        self.deliberate_frames = deliberate_frames  # ~333ms at 30 FPS
        self.closed_count = 0

    def update(self, avg_ear):
        """
        Returns:
            'none', 'natural_blink', or 'deliberate_blink'
        """
        if avg_ear < self.ear_threshold:
            self.closed_count += 1
            return 'none'  # Still closed: wait

        # Eye just opened
        result = 'none'
        if self.closed_count >= self.deliberate_frames:
            result = 'deliberate_blink'
        elif self.closed_count >= self.natural_frames:
            result = 'natural_blink'

        self.closed_count = 0
        return result
```

## 9.6 False Activation Prevention

The **Midas Touch problem**: everything you look at gets activated.

### Mitigation Strategies

1. **Dwell threshold:** Don't activate instantly: require sustained gaze
   (500-1000ms).
2. **Spatial stability:** Require low variance in gaze position before
   starting dwell timer.
3. **Cooldown:** After activation, ignore input for 500-1000ms.
4. **Confidence threshold:** Only process frames where face detection
   confidence > 0.8.
5. **Two-step activation:** Dwell to select, then deliberate blink to
   confirm.
6. **Dynamic dwell:** Longer dwell for destructive actions, shorter for safe
   ones.

---

# 10. M9: Evaluation Framework

## 10.1 Evaluation Categories

### A. Gaze Accuracy

| Metric | Description |
|--------|-------------|
| Mean error (px) | Average Euclidean distance between predicted and actual |
| Median error (px) | Middle-value error (robust to outliers) |
| 95th percentile (px) | Worst-case error |
| Mean error (degrees) | Physical angle: research standard |

### B. Calibration

| Metric | Description |
|--------|-------------|
| Calibration time (s) | Total time for calibration procedure |
| Number of samples | How many calibration points collected |
| Sample efficiency | Accuracy improvement per additional sample |

### C. Interaction

| Metric | Description |
|--------|-------------|
| Selection success rate (%) | Correct target selections / total attempts |
| False activation rate | Unintended activations per minute |
| Selection latency (ms) | Time from looking at target to selection |

### D. Robustness

| Metric | Description |
|--------|-------------|
| Cross-user error | Error when testing on different users |
| Lighting degradation | Error increase under different lighting |
| Movement tolerance | Error vs head movement range |

### E. System Performance

| Metric | Description |
|--------|-------------|
| FPS | Frames per second of full pipeline |
| CPU usage (%) | Processor utilization |
| Memory usage (MB) | RAM consumption |
| End-to-end latency (ms) | Camera capture to interaction |

## 10.2 Experimental Protocol

### Gaze Accuracy Test

```text
1. Calibrate the system.
2. Display 16 test points (NOT the calibration points).
3. For each test point:
   a. Display the target.
   b. Wait 2 seconds for fixation.
   c. Record 30 frames of predicted gaze.
   d. Average predictions.
   e. Compute error vs known target.
4. Calculate mean, median, 95th percentile error.
```

### Interaction Test (Dwell Selection)

```text
1. Calibrate the system.
2. Display a grid of 9 large buttons (200×200 px).
3. Highlight one button as the target.
4. User must select it via dwell.
5. Record: success/failure, time to select, false activations.
6. Repeat for 20-30 trials.
```

### Conventional vs Adaptive Calibration Comparison

```text
Within-subject design (same user does both):

Session A: Conventional:
  - 9-point fixed grid calibration
  - Gaze accuracy test
  - Interaction test

Session B: Adaptive:
  - 5-point initial + up to 4 adaptive points
  - Same gaze accuracy test
  - Same interaction test

Compare: accuracy, time, reliability
```

## 10.3 Recording Experiment Results

Store results in structured format:

```python
experiment_results = {
    'session_id': 'exp_001',
    'user_id': 'user_01',
    'calibration_type': 'adaptive',  # or 'conventional'
    'calibration_points': 9,
    'calibration_time_s': 45.2,
    'gaze_accuracy': {
        'mean_error_px': 82.3,
        'median_error_px': 65.1,
        'p95_error_px': 198.4,
        'mean_error_deg': 1.8,
    },
    'interaction': {
        'selection_success_rate': 0.87,
        'false_activations_per_min': 2.1,
        'mean_selection_latency_ms': 620,
    },
    'performance': {
        'fps': 27.4,
        'cpu_percent': 45,
        'memory_mb': 312,
    },
    'conditions': {
        'lighting': 'normal',
        'distance_cm': 60,
        'glasses': False,
    },
}
```

---

# 11. M10: Windows Deployment

## 11.1 Platform Abstraction

The `Camera` class already provides a useful abstraction. For Windows:

1. **Remove V4L2 dependency:** The `subprocess.run(["v4l2-ctl", ...])` call in
   `camera.py` must be skipped on Windows.

```python
import platform

class Camera:
    def open(self):
        if platform.system() == 'Linux':
            subprocess.run([
                "v4l2-ctl", "-d", self.source,
                "--set-ctrl=exposure_dynamic_framerate=0"
            ])

        # Windows: use DirectShow backend
        backend = cv2.CAP_V4L2 if platform.system() == 'Linux' \
                  else cv2.CAP_DSHOW

        self.capture = cv2.VideoCapture(self.source, backend)
        # ... rest of configuration
```

2. **Camera source:** On Windows, use integer indices (0, 1, 2) instead of
   `/dev/videoN`.

## 11.2 Dependencies

All core dependencies (`opencv-python`, `mediapipe`, `numpy`, `scikit-learn`)
work on Windows via pip. No special Windows-only packages are needed for the
core pipeline.

## 11.3 Cursor Control

`pyautogui` and `pynput` both work on Windows for cursor control and
interaction.

## 11.4 Packaging (Future)

Consider `PyInstaller` or `cx_Freeze` for creating a standalone Windows
executable:

```bash
pip install pyinstaller
pyinstaller --onefile --windowed src/ocular/main.py
```

---

# Appendix A: Face Detection Method Comparison

| Feature | YuNet | MediaPipe Face Mesh | dlib (68-point) |
|---------|-------|---------------------|-----------------|
| **Total Landmarks** | 5 | 478 (with iris) | 68 |
| **Iris Detection** | No | Yes (10 points) | No |
| **Eye Contour** | 1 center point | Full detailed contour | 6 points per eye |
| **3D Coordinates** | 2D only | x, y, z | 2D only |
| **Speed (CPU)** | ~2-5ms  | ~10-30ms | ~35-60ms |
| **Model Size** | ~230KB | ~2-4MB | ~100MB |
| **Install** | Built into OpenCV | `pip install mediapipe` | Needs CMake + C++ |
| **Head Pose Data** | 5 points (basic PnP) | Rich 3D mesh | 68 points (good PnP) |
| **Blink Detection** | No | Yes (via EAR) | Yes (via EAR) |
| **Best For** | Fast face detection | **Full gaze pipeline** | EAR / eye shape analysis |

---

# Appendix B: Complete Landmark Reference

## MediaPipe Iris Landmarks (refine_landmarks=True)

| Index | Description |
|-------|-------------|
| 468 | Left iris center |
| 469 | Left iris top |
| 470 | Left iris right |
| 471 | Left iris bottom |
| 472 | Left iris left |
| 473 | Right iris center |
| 474 | Right iris top |
| 475 | Right iris right |
| 476 | Right iris bottom |
| 477 | Right iris left |

## Key Eye Landmarks

| Landmark | Left Eye | Right Eye |
|----------|----------|-----------|
| Outer corner | 33 | 263 |
| Inner corner | 133 | 362 |
| Upper eyelid | 159 | 386 |
| Lower eyelid | 145 | 374 |

## EAR Landmarks (6-Point Eye Contour)

| Point | Left Eye | Right Eye |
|-------|----------|-----------|
| p1 (outer) | 33 | 362 |
| p2 (upper-outer) | 160 | 385 |
| p3 (upper-inner) | 158 | 387 |
| p4 (inner) | 133 | 263 |
| p5 (lower-inner) | 153 | 380 |
| p6 (lower-outer) | 144 | 373 |

## Head Pose Landmarks (for solvePnP)

| Point | MediaPipe Index |
|-------|----------------|
| Nose tip | 1 |
| Chin | 152 |
| Left eye outer | 33 |
| Right eye outer | 263 |
| Left mouth corner | 61 |
| Right mouth corner | 291 |

---

# Appendix C: Mathematical Formulas

## Iris Horizontal Ratio

```
ratio_h = (iris_x - outer_corner_x) / (inner_corner_x - outer_corner_x)
```

- ≈ 0.5 → Looking straight
- < 0.5 → Looking toward outer side
- > 0.5 → Looking toward nose

## Iris Vertical Ratio

```
ratio_v = (iris_y - top_eyelid_y) / (bottom_eyelid_y - top_eyelid_y)
```

- ≈ 0.5 → Looking straight
- < 0.5 → Looking up
- > 0.5 → Looking down

## Eye Aspect Ratio (EAR)

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

- Open eye: 0.20 - 0.35
- Closed: < 0.20

## Euclidean Distance

```
d = sqrt((x1 - x2)² + (y1 - y2)²)
```

## Pixel to Degree Conversion

```
degrees = arctan(error_cm / distance_cm) × (180 / π)
error_cm = error_px × (screen_width_cm / screen_width_px)
```

---

# Appendix D: Recommended File Structure

```text
OCULAR/
├── README.md                    # Project overview (existing)
├── requirements.txt             # Python dependencies
├── pyproject.toml               # Package setup
│
├── archive/                     # Legacy and prototype scripts
│   ├── pupil.py
│   ├── cameraTest.py
│   ├── eyeTest.py
│   └── trackerTest.py
│
├── models/
│   ├── face_detection_yunet_2026may.onnx   # YuNet model (existing)
│   └── face_landmarker.task                # MediaPipe model
│
├── calibration/
│   ├── user_session_001.npz     # Calibration data
│   └── user_session_002.npz
│
├── data/
│   ├── feature_log.csv          # Logged feature vectors
│   └── experiments/
│       ├── exp_001.json         # Experiment results
│       └── exp_002.json
│
├── docs/
│   ├── GUIDE.md                 # This guide
│   ├── COMPREHENSIVE_SYSTEM_DOCUMENTATION.md
│   ├── EVALUATION_REPORT.md     # Final evaluation writeup
│   └── API_REFERENCE.md
│
├── experiments/
│   ├── compare_models.py        # Model comparison script
│   └── calibration_experiment.py
│
├── src/
│   └── ocular/
│       ├── __init__.py
│       ├── camera.py            # Camera abstraction (existing)
│       ├── tracker.py           # MediaPipe face/iris tracker
│       ├── features.py          # Feature extraction
│       ├── calibration.py       # Calibration system
│       ├── gaze.py              # Gaze regression models
│       ├── adaptive.py          # Adaptive calibration
│       ├── interaction.py       # Cursor, dwell, scroll
│       ├── filters.py           # One Euro Filter, etc.
│       ├── blink.py             # Blink detection
│       ├── evaluation.py        # Evaluation module
│       └── main.py              # Main application entry
│
└── tests/
    ├── test_tracker.py
    ├── test_features.py
    └── test_calibration.py
```

---

# Quick Reference: Development Order

```text
 1. M1: Camera pipeline                          [DONE]
 2. M2: Face and iris tracking (MediaPipe)        [DONE]
 3. M3: 11-D feature vector and solvePnP head pose [DONE]
 4. M4: Fullscreen visual calibration             [DONE]
 5. M5: Multi-model gaze regression (Ridge/RF)    [DONE]
 6. M6: Adaptive active-learning calibration      [DONE]
 7. M7: Robustness and metric conversions         [DONE]
 8. M8: One Euro filter, cursor, dwell, scroller  [DONE]
 9. M9: Benchmark and evaluation framework        [DONE]
10. M10: Unified CLI and cross-platform packaging [DONE]
11. M11: Performance profiler and study runner   [DONE]
```

> **Remember:** Start simple. Increase complexity only when experimental
> evidence demonstrates the simpler approach is insufficient.
