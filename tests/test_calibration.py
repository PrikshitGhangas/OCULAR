import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from calibration import CalibrationSession


class TestCalibration(unittest.TestCase):
    def test_grid_generation_patterns(self):
        w, h = 1920, 1080

        pts_5 = CalibrationSession.generate_grid(w, h, "5-point")
        self.assertEqual(len(pts_5), 5)

        pts_9 = CalibrationSession.generate_grid(w, h, "9-point")
        self.assertEqual(len(pts_9), 9)

        pts_13 = CalibrationSession.generate_grid(w, h, "13-point")
        self.assertEqual(len(pts_13), 13)

        pts_16 = CalibrationSession.generate_grid(w, h, "16-point")
        self.assertEqual(len(pts_16), 16)

    def test_save_and_load_session(self):
        X = np.random.rand(9, 11)
        y = np.array(CalibrationSession.generate_grid(1920, 1080, "9-point"))
        metadata = {"user": "test_user", "notes": "unit test"}

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
            temp_path = tf.name

        try:
            CalibrationSession.save_session(temp_path, X, y, metadata)
            loaded_X, loaded_y, loaded_meta = CalibrationSession.load_session(temp_path)

            np.testing.assert_allclose(X, loaded_X)
            np.testing.assert_allclose(y, loaded_y)
            self.assertEqual(loaded_meta["user"], "test_user")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
