import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from gaze import GazeRegressor


class TestGazeRegressor(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create small synthetic dataset with known linear + polynomial relation
        self.n_samples = 12
        self.X = np.random.uniform(0.3, 0.7, size=(self.n_samples, 11))
        # Target X mapped predominantly from feature 0 (left iris h) and feature 9 (yaw)
        # Target Y mapped predominantly from feature 1 (left iris v) and feature 8 (pitch)
        target_x = self.X[:, 0] * 1920.0 + self.X[:, 9] * 10.0
        target_y = self.X[:, 1] * 1080.0 + self.X[:, 8] * 10.0
        self.y = np.column_stack((target_x, target_y))

    def test_ridge_regression_fit_and_predict(self):
        reg = GazeRegressor("ridge", screen_w=1920, screen_h=1080)
        reg.fit(self.X, self.y)
        self.assertTrue(reg.is_trained)

        preds = reg.predict(self.X)
        self.assertEqual(preds.shape, (self.n_samples, 2))

        # Check bounds
        self.assertTrue((preds[:, 0] >= 0).all() and (preds[:, 0] <= 1920).all())
        self.assertTrue((preds[:, 1] >= 0).all() and (preds[:, 1] <= 1080).all())

    def test_loocv_evaluation(self):
        reg = GazeRegressor("ridge", screen_w=1920, screen_h=1080)
        metrics = reg.evaluate_loocv(self.X, self.y)

        self.assertIn("mean_error_px", metrics)
        self.assertIn("median_error_px", metrics)
        self.assertIn("mean_error_deg", metrics)
        self.assertGreater(metrics["mean_error_px"], 0.0)

    def test_svr_and_rf_models(self):
        for m_type in ["svr", "rf"]:
            reg = GazeRegressor(m_type, screen_w=1920, screen_h=1080)
            reg.fit(self.X, self.y)
            pred = reg.predict(self.X[0])
            self.assertEqual(len(pred), 2)


if __name__ == "__main__":
    unittest.main()
