import cv2
import numpy as np


class FeatureExtractor:
    """
    Translates raw facial and ocular landmarks into a structured, numerical
    feature vector optimized for gaze estimation and blink classification.
    """

    # Anatomical 3D model reference points (in mm, generic anthropometric standard)
    MODEL_POINTS_3D = np.array(
        [
            (0.0, 0.0, 0.0),          # Nose tip (idx 1)
            (0.0, -330.0, -65.0),      # Chin (idx 152)
            (-225.0, 170.0, -135.0),   # Left eye outer corner (idx 33)
            (225.0, 170.0, -135.0),    # Right eye outer corner (idx 263)
            (-150.0, -150.0, -125.0),  # Left mouth corner (idx 61)
            (150.0, -150.0, -125.0),   # Right mouth corner (idx 291)
        ],
        dtype=np.float64,
    )

    POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]

    FEATURE_NAMES = [
        "left_iris_h_ratio",
        "left_iris_v_ratio",
        "right_iris_h_ratio",
        "right_iris_v_ratio",
        "left_ear",
        "right_ear",
        "left_eye_aspect",
        "right_eye_aspect",
        "head_pitch",
        "head_yaw",
        "head_roll",
    ]

    def __init__(self):
        self._camera_matrix = None
        self._dist_coeffs = None

    def extract(self, landmarks, frame_shape):
        """
        Extract the standard 11-element feature vector from landmarks.

        Args:
            landmarks: MediaPipe NormalizedLandmarkList
            frame_shape: (height, width, channels)

        Returns:
            numpy.ndarray: 1D array of shape (11,) or None if extraction fails.
        """
        if landmarks is None or len(landmarks.landmark) < 478:
            return None

        h, w = frame_shape[:2]

        try:
            # 1. Iris Ratios
            l_h, l_v = self._compute_iris_ratios(landmarks, w, h, "left")
            r_h, r_v = self._compute_iris_ratios(landmarks, w, h, "right")

            # 2. Eye Aspect Ratios (EAR)
            l_ear = self._compute_ear(landmarks, w, h, "left")
            r_ear = self._compute_ear(landmarks, w, h, "right")

            # 3. Eye Geometry Aspects
            l_aspect = self._compute_eye_aspect(landmarks, w, h, "left")
            r_aspect = self._compute_eye_aspect(landmarks, w, h, "right")

            # 4. Head Pose (Euler Angles)
            pitch, yaw, roll, _, _ = self.estimate_head_pose(landmarks, frame_shape)

            features = np.array(
                [
                    l_h,
                    l_v,
                    r_h,
                    r_v,
                    l_ear,
                    r_ear,
                    l_aspect,
                    r_aspect,
                    pitch,
                    yaw,
                    roll,
                ],
                dtype=np.float64,
            )

            # Sanity check for NaN/Inf
            if np.isnan(features).any() or np.isinf(features).any():
                return None

            return features

        except Exception:
            return None

    def _get_pt(self, landmarks, idx, w, h):
        lm = landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h], dtype=np.float64)

    def _compute_iris_ratios(self, landmarks, w, h, side):
        """
        Calculate relative position of iris center within ocular bounding margins.
        """
        if side == "left":
            iris = self._get_pt(landmarks, 468, w, h)
            inner = self._get_pt(landmarks, 133, w, h)
            outer = self._get_pt(landmarks, 33, w, h)
            top = self._get_pt(landmarks, 159, w, h)
            bottom = self._get_pt(landmarks, 145, w, h)
        else:
            iris = self._get_pt(landmarks, 473, w, h)
            inner = self._get_pt(landmarks, 362, w, h)
            outer = self._get_pt(landmarks, 263, w, h)
            top = self._get_pt(landmarks, 386, w, h)
            bottom = self._get_pt(landmarks, 374, w, h)

        eye_w = np.linalg.norm(inner - outer)
        eye_h = np.linalg.norm(bottom - top)

        if eye_w > 1.0:
            h_ratio = float((iris[0] - outer[0]) / (inner[0] - outer[0]))
        else:
            h_ratio = 0.5

        if eye_h > 1.0:
            v_ratio = float((iris[1] - top[1]) / (bottom[1] - top[1]))
        else:
            v_ratio = 0.5

        # Bound to reasonable limits
        h_ratio = np.clip(h_ratio, 0.0, 1.0)
        v_ratio = np.clip(v_ratio, 0.0, 1.0)

        return h_ratio, v_ratio

    def _compute_ear(self, landmarks, w, h, side):
        """
        Compute Eye Aspect Ratio (Soukupová & Čech formula).
        """
        if side == "left":
            indices = {"p1": 33, "p2": 160, "p3": 158, "p4": 133, "p5": 153, "p6": 144}
        else:
            indices = {"p1": 362, "p2": 385, "p3": 387, "p4": 263, "p5": 380, "p6": 373}

        pts = {k: self._get_pt(landmarks, v, w, h) for k, v in indices.items()}

        vertical_1 = np.linalg.norm(pts["p2"] - pts["p6"])
        vertical_2 = np.linalg.norm(pts["p3"] - pts["p5"])
        horizontal = np.linalg.norm(pts["p1"] - pts["p4"])

        if horizontal < 1e-3:
            return 0.0

        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return float(np.clip(ear, 0.0, 1.0))

    def _compute_eye_aspect(self, landmarks, w, h, side):
        """
        Calculate eye width to height aspect ratio.
        """
        if side == "left":
            inner = self._get_pt(landmarks, 133, w, h)
            outer = self._get_pt(landmarks, 33, w, h)
            top = self._get_pt(landmarks, 159, w, h)
            bottom = self._get_pt(landmarks, 145, w, h)
        else:
            inner = self._get_pt(landmarks, 362, w, h)
            outer = self._get_pt(landmarks, 263, w, h)
            top = self._get_pt(landmarks, 386, w, h)
            bottom = self._get_pt(landmarks, 374, w, h)

        eye_w = np.linalg.norm(inner - outer)
        eye_h = np.linalg.norm(bottom - top)

        if eye_h < 1e-3:
            return 1.0

        return float(np.clip(eye_w / eye_h, 0.5, 10.0))

    def estimate_head_pose(self, landmarks, frame_shape):
        """
        Estimate 3D head pose (pitch, yaw, roll) using solvePnP.

        Returns:
            tuple: (pitch, yaw, roll, rvec, tvec) in degrees and vectors
        """
        h, w = frame_shape[:2]

        if self._camera_matrix is None:
            focal_length = float(w)
            center = (w / 2.0, h / 2.0)
            self._camera_matrix = np.array(
                [
                    [focal_length, 0.0, center[0]],
                    [0.0, focal_length, center[1]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        image_points = np.array(
            [
                (landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h)
                for idx in self.POSE_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS_3D,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return 0.0, 0.0, 0.0, None, None

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
            np.vstack((proj_matrix, [0.0, 0.0, 0.0, 1.0]))[:3]
        )

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        return pitch, yaw, roll, rvec, tvec

    def draw_head_pose_axes(self, frame, rvec, tvec, length=100):
        """
        Draw 3D coordinate axes projecting from nose tip onto the frame.
        """
        if rvec is None or tvec is None or self._camera_matrix is None:
            return frame

        axis_3d = np.array(
            [
                (0.0, 0.0, 0.0),
                (length, 0.0, 0.0),    # X axis (Red)
                (0.0, -length, 0.0),   # Y axis (Green)
                (0.0, 0.0, -length),   # Z axis (Blue)
            ],
            dtype=np.float64,
        )

        axis_2d, _ = cv2.projectPoints(
            axis_3d, rvec, tvec, self._camera_matrix, self._dist_coeffs
        )

        origin = tuple(map(int, axis_2d[0].ravel()))
        pt_x = tuple(map(int, axis_2d[1].ravel()))
        pt_y = tuple(map(int, axis_2d[2].ravel()))
        pt_z = tuple(map(int, axis_2d[3].ravel()))

        cv2.line(frame, origin, pt_x, (0, 0, 255), 2)  # Red X
        cv2.line(frame, origin, pt_y, (0, 255, 0), 2)  # Green Y
        cv2.line(frame, origin, pt_z, (255, 0, 0), 2)  # Blue Z

        return frame
