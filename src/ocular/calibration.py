import json
import os
import time
import cv2
import numpy as np


class CalibrationSession:
    """
    Manages conventional N-point visual calibration sessions.

    Generates on-screen visual fixation stimuli, collects synchronous ocular
    features, filters blinks/saccades, and serializes calibration datasets.
    """

    def __init__(self, screen_w=1920, screen_h=1080):
        self.screen_w = screen_w
        self.screen_h = screen_h

    @staticmethod
    def get_screen_resolution():
        """
        Attempt to detect primary monitor resolution, with safe fallback.
        """
        try:
            import screeninfo

            monitors = screeninfo.get_monitors()
            if monitors:
                return monitors[0].width, monitors[0].height
        except Exception:
            pass
        return 1920, 1080

    @classmethod
    def generate_grid(cls, screen_w, screen_h, pattern="9-point", margin=0.10):
        """
        Generate target screen coordinates for calibration.

        Args:
            screen_w: Screen width in pixels
            screen_h: Screen height in pixels
            pattern: One of '5-point', '9-point', '13-point', '16-point'
            margin: Fractional border margin [0.05 - 0.20]

        Returns:
            list of tuple (x, y): Screen pixel targets
        """
        x_min = int(screen_w * margin)
        x_max = int(screen_w * (1.0 - margin))
        y_min = int(screen_h * margin)
        y_max = int(screen_h * (1.0 - margin))

        x_mid = (x_min + x_max) // 2
        y_mid = (y_min + y_max) // 2

        if pattern == "5-point":
            return [
                (x_mid, y_mid),  # Center
                (x_min, y_min),  # Top-left
                (x_max, y_min),  # Top-right
                (x_min, y_max),  # Bottom-left
                (x_max, y_max),  # Bottom-right
            ]

        elif pattern == "9-point":
            points = []
            xs = [x_min, x_mid, x_max]
            ys = [y_min, y_mid, y_max]
            for y in ys:
                for x in xs:
                    points.append((x, y))
            return points

        elif pattern == "13-point":
            # 9-point grid + 4 edge intermediate points
            pts = cls.generate_grid(screen_w, screen_h, "9-point", margin)
            extra = [
                ((x_min + x_mid) // 2, y_mid),
                ((x_mid + x_max) // 2, y_mid),
                (x_mid, (y_min + y_mid) // 2),
                (x_mid, (y_mid + y_max) // 2),
            ]
            return pts + extra

        elif pattern == "16-point":
            points = []
            xs = np.linspace(x_min, x_max, 4, dtype=int)
            ys = np.linspace(y_min, y_max, 4, dtype=int)
            for y in ys:
                for x in xs:
                    points.append((int(x), int(y)))
            return points

        else:
            raise ValueError(f"Unknown calibration pattern: {pattern}")

    def run_interactive(
        self,
        camera,
        tracker,
        extractor,
        points=None,
        dwell_seconds=1.8,
        prep_seconds=0.6,
        window_name="OCULAR - Calibration",
    ):
        """
        Execute interactive full-screen visual calibration routine.

        Returns:
            tuple (numpy.ndarray, numpy.ndarray): (X_features, y_targets)
        """
        if points is None:
            points = self.generate_grid(self.screen_w, self.screen_h, "9-point")

        features_dataset = []
        targets_dataset = []

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        total_pts = len(points)

        for pt_idx, (tx, ty) in enumerate(points):
            # Phase 1: Saccade & fixation orienting (prep period)
            prep_start = time.perf_counter()
            while time.perf_counter() - prep_start < prep_seconds:
                canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

                # Target marker
                cv2.circle(canvas, (tx, ty), 28, (0, 165, 255), 2)  # Orange target ring
                cv2.circle(canvas, (tx, ty), 4, (0, 165, 255), -1)

                info = f"Point {pt_idx + 1}/{total_pts} - Focus on target"
                cv2.putText(
                    canvas, info, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2
                )
                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow(window_name)
                    return None, None

            # Phase 2: Active sampling with shrinking fixation ring
            sample_start = time.perf_counter()
            collected_samples = []

            while True:
                elapsed = time.perf_counter() - sample_start
                if elapsed >= dwell_seconds:
                    break

                progress = min(1.0, elapsed / dwell_seconds)
                current_radius = max(6, int(30 * (1.0 - progress)))

                canvas = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

                # Shrinking cyan ring cues exact timing
                cv2.circle(canvas, (tx, ty), current_radius, (255, 255, 0), 2)
                cv2.circle(canvas, (tx, ty), 3, (0, 255, 0), -1)

                info = f"Calibrating Point {pt_idx + 1}/{total_pts} [{int(progress * 100)}%]"
                cv2.putText(
                    canvas, info, (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

                # Capture camera frame and extract features
                success, frame = camera.read()
                if success:
                    found = tracker.process(frame)
                    if found:
                        feats = extractor.extract(tracker.landmarks, frame.shape)
                        if feats is not None:
                            # Discard samples with blinks (EAR < 0.20)
                            left_ear, right_ear = feats[4], feats[5]
                            if left_ear >= 0.20 and right_ear >= 0.20:
                                # Collect in latter 70% of dwell window to prevent saccade drag
                                if progress >= 0.30:
                                    collected_samples.append(feats)

                cv2.imshow(window_name, canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyWindow(window_name)
                    return None, None

            # Process collected samples for this point
            if len(collected_samples) >= 5:
                samples_arr = np.array(collected_samples)
                # Compute robust median / trimmed mean
                median_feat = np.median(samples_arr, axis=0)
                dists = np.linalg.norm(samples_arr - median_feat, axis=1)
                inliers = samples_arr[dists <= np.percentile(dists, 80)]
                point_feature = np.mean(inliers, axis=0)

                features_dataset.append(point_feature)
                targets_dataset.append([float(tx), float(ty)])

        cv2.destroyWindow(window_name)

        if not features_dataset:
            return None, None

        X = np.array(features_dataset, dtype=np.float64)
        y = np.array(targets_dataset, dtype=np.float64)
        return X, y

    @staticmethod
    def save_session(filepath, X, y, metadata=None):
        """
        Save calibration dataset to compressed .npz archive.
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        meta_str = json.dumps(metadata or {})
        np.savez_compressed(filepath, X=X, y=y, metadata=meta_str)

    @staticmethod
    def load_session(filepath):
        """
        Load calibration dataset from compressed .npz archive.
        """
        data = np.load(filepath, allow_pickle=False)
        X = data["X"]
        y = data["y"]
        metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
        return X, y, metadata
