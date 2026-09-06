import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np


class LandmarksWrapper:
    """
    Uniform wrapper around landmark lists providing both sequence indexing
    and legacy `.landmark` attribute access for maximum cross-API compatibility.
    """

    def __init__(self, raw_landmarks):
        self._lms = raw_landmarks
        self.landmark = raw_landmarks

    def __getitem__(self, idx):
        return self._lms[idx]

    def __len__(self):
        return len(self._lms)

    def __iter__(self):
        return iter(self._lms)


class FaceTracker:
    """
    High-fidelity face, eye, and iris tracking module powered by MediaPipe.

    Supports both modern MediaPipe Tasks API (>= 1.0) and legacy solutions API (< 1.0),
    extracting 478 3D facial landmarks including sub-pixel iris centers,
    dense ocular contours, and head-pose anchor points.
    """

    # --- Landmark Indices (MediaPipe Face Mesh / FaceLandmarker with iris) ---
    # Iris centers
    LEFT_IRIS_CENTER = 468
    RIGHT_IRIS_CENTER = 473

    # Iris ring boundary points
    LEFT_IRIS_RING = [469, 470, 471, 472]
    RIGHT_IRIS_RING = [474, 475, 476, 477]

    # Eye corners (inner / outer canthi)
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263

    # Eyelid extremes (superior / inferior palpebral margins)
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # 6-Point Eye Contour for Eye Aspect Ratio (EAR) computation
    LEFT_EYE_EAR = {
        "p1": 33,   # Outer corner
        "p2": 160,  # Upper-outer
        "p3": 158,  # Upper-inner
        "p4": 133,  # Inner corner
        "p5": 153,  # Lower-inner
        "p6": 144,  # Lower-outer
    }

    RIGHT_EYE_EAR = {
        "p1": 362,  # Inner corner
        "p2": 385,  # Upper-inner
        "p3": 387,  # Upper-outer
        "p4": 263,  # Outer corner
        "p5": 380,  # Lower-outer
        "p6": 373,  # Lower-inner
    }

    # Head pose anatomical anchors for Perspective-n-Point (solvePnP)
    POSE_LANDMARKS = {
        "nose_tip": 1,
        "chin": 152,
        "left_eye_outer": 33,
        "right_eye_outer": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    def __init__(
        self,
        max_faces=1,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        model_path=None,
    ):
        """
        Initialize FaceTracker.

        Args:
            max_faces: Maximum number of faces to detect and track (default: 1)
            detection_confidence: Minimum face detection confidence threshold [0.0, 1.0]
            tracking_confidence: Minimum landmark tracking confidence threshold [0.0, 1.0]
            model_path: Path to face_landmarker.task model (auto-downloaded if absent)
        """
        self.landmarks = None
        self.face_present = False
        self._use_tasks_api = not hasattr(mp, "solutions")

        if self._use_tasks_api:
            # Modern MediaPipe Tasks API (>= 1.0)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            if model_path is None:
                # Resolve default path relative to project structure
                base_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "..", "..", "models")
                )
                os.makedirs(base_dir, exist_ok=True)
                model_path = os.path.join(base_dir, "face_landmarker.task")

            if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                print(f"[+] Downloading MediaPipe FaceLandmarker task model to {model_path}...")
                try:
                    urllib.request.urlretrieve(self.MODEL_URL, model_path)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to download FaceLandmarker model from {self.MODEL_URL}: {e}"
                    )

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=max_faces,
                min_face_detection_confidence=detection_confidence,
                min_face_presence_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.face_mesh = None
        else:
            # Legacy MediaPipe Solutions API (< 1.0)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.detector = None

    def process(self, frame):
        """
        Process a BGR video frame and extract facial/ocular landmarks.

        Args:
            frame: Input image in OpenCV BGR format (numpy.ndarray)

        Returns:
            bool: True if a face was detected and landmarks extracted, False otherwise.
        """
        if frame is None or frame.size == 0:
            self.landmarks = None
            self.face_present = False
            return False

        # MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._use_tasks_api:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results and results.face_landmarks:
                self.landmarks = LandmarksWrapper(results.face_landmarks[0])
                self.face_present = True
                return True
        else:
            results = self.face_mesh.process(rgb_frame)
            if results and results.multi_face_landmarks:
                self.landmarks = LandmarksWrapper(results.multi_face_landmarks[0].landmark)
                self.face_present = True
                return True

        self.landmarks = None
        self.face_present = False
        return False

    def get_landmark_px(self, index, frame_shape):
        """
        Retrieve pixel coordinates (x, y) for a specific landmark index.

        Args:
            index: Landmark index (0 to 477)
            frame_shape: Frame shape tuple (height, width, ...)

        Returns:
            tuple (int, int) or None: (x, y) in image coordinates.
        """
        if self.landmarks is None or index >= len(self.landmarks):
            return None
        lm = self.landmarks[index]
        h, w = frame_shape[:2]
        return (int(lm.x * w), int(lm.y * h))

    def get_landmark_normalized(self, index):
        """
        Retrieve normalized 3D coordinates (x, y, z) for a landmark index.

        Args:
            index: Landmark index (0 to 477)

        Returns:
            tuple (float, float, float) or None: (x, y, z) normalized coordinates.
        """
        if self.landmarks is None or index >= len(self.landmarks):
            return None
        lm = self.landmarks[index]
        return (lm.x, lm.y, lm.z)

    def get_iris_centers(self, frame_shape):
        """
        Retrieve pixel coordinates for both iris centers.

        Args:
            frame_shape: Frame shape tuple (height, width, ...)

        Returns:
            tuple: (left_iris_center, right_iris_center), each (x, y) or (None, None).
        """
        if self.landmarks is None:
            return None, None
        left = self.get_landmark_px(self.LEFT_IRIS_CENTER, frame_shape)
        right = self.get_landmark_px(self.RIGHT_IRIS_CENTER, frame_shape)
        return left, right

    def get_eye_bounding_box(self, side, frame_shape, padding=0.2):
        """
        Compute pixel bounding box for left or right eye region.

        Args:
            side: 'left' or 'right'
            frame_shape: (height, width, ...)
            padding: fractional padding to expand box bounds

        Returns:
            tuple: (x, y, w, h) bounding box or None
        """
        if self.landmarks is None:
            return None

        h, w = frame_shape[:2]
        indices = self.LEFT_EYE_EAR.values() if side == "left" else self.RIGHT_EYE_EAR.values()

        xs = [int(self.landmarks[i].x * w) for i in indices]
        ys = [int(self.landmarks[i].y * h) for i in indices]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        box_w = max_x - min_x
        box_h = max_y - min_y

        pad_x = int(box_w * padding)
        pad_y = int(box_h * padding)

        x = max(0, min_x - pad_x)
        y = max(0, min_y - pad_y)
        bw = min(w - x, box_w + 2 * pad_x)
        bh = min(h - y, box_h + 2 * pad_y)

        return (x, y, bw, bh)

    def draw_debug_overlay(self, frame):
        """
        Render diagnostic visual overlays onto frame for tracking verification.

        Draws iris centers (cyan), eye contours (green), and pose anchors (magenta).
        """
        if self.landmarks is None:
            return frame

        # Draw Iris Centers
        left_iris, right_iris = self.get_iris_centers(frame.shape)
        if left_iris:
            cv2.circle(frame, left_iris, 3, (255, 255, 0), -1)  # Cyan
        if right_iris:
            cv2.circle(frame, right_iris, 3, (255, 255, 0), -1)

        # Draw Iris Boundary Rings
        for idx in self.LEFT_IRIS_RING + self.RIGHT_IRIS_RING:
            pt = self.get_landmark_px(idx, frame.shape)
            if pt:
                cv2.circle(frame, pt, 1, (0, 255, 255), -1)  # Yellow

        # Draw Eye Corner & Eyelid Landmarks
        eye_points = [
            self.LEFT_EYE_INNER,
            self.LEFT_EYE_OUTER,
            self.LEFT_EYE_TOP,
            self.LEFT_EYE_BOTTOM,
            self.RIGHT_EYE_INNER,
            self.RIGHT_EYE_OUTER,
            self.RIGHT_EYE_TOP,
            self.RIGHT_EYE_BOTTOM,
        ]
        for idx in eye_points:
            pt = self.get_landmark_px(idx, frame.shape)
            if pt:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)  # Green

        # Draw Head Pose Anchors
        for name, idx in self.POSE_LANDMARKS.items():
            pt = self.get_landmark_px(idx, frame.shape)
            if pt:
                cv2.circle(frame, pt, 3, (255, 0, 255), -1)  # Magenta

        return frame

    def release(self):
        """Release MediaPipe resources."""
        if self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

        if self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except Exception:
                pass
            self.face_mesh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
