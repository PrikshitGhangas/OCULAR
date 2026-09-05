import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from tracker import FaceTracker


class TestTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = FaceTracker()

    def tearDown(self):
        self.tracker.release()

    def test_tracker_initialization(self):
        backend = self.tracker.detector if self.tracker._use_tasks_api else self.tracker.face_mesh
        self.assertIsNotNone(backend)
        self.assertFalse(self.tracker.face_present)
        self.assertIsNone(self.tracker.landmarks)

    def test_tracker_process_empty_frame(self):
        empty_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detected = self.tracker.process(empty_frame)
        self.assertFalse(detected)
        self.assertIsNone(self.tracker.landmarks)

    def test_tracker_none_frame_safety(self):
        self.assertFalse(self.tracker.process(None))
        self.assertIsNone(self.tracker.get_landmark_px(468, (720, 1280, 3)))
        left_iris, right_iris = self.tracker.get_iris_centers((720, 1280, 3))
        self.assertIsNone(left_iris)
        self.assertIsNone(right_iris)


if __name__ == "__main__":
    unittest.main()
