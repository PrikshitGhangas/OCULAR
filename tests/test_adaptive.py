import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from adaptive import AdaptiveCalibrationEngine


class TestAdaptiveCalibration(unittest.TestCase):
    def setUp(self):
        self.engine = AdaptiveCalibrationEngine(screen_w=1920, screen_h=1080, strategy="hybrid")

    def test_generate_candidate_pool(self):
        pool = self.engine.generate_candidate_pool(grid_rows=4, grid_cols=4)
        self.assertEqual(len(pool), 16)
        for x, y in pool:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 1920)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 1080)

    def test_candidate_pool_respects_margin(self):
        pool = self.engine.generate_candidate_pool(grid_rows=3, grid_cols=3, margin=0.20)
        for x, y in pool:
            self.assertGreaterEqual(x, 1920 * 0.20 - 1)
            self.assertLessEqual(x, 1920 * 0.80 + 1)

    def test_select_next_target_returns_valid_point(self):
        existing_targets = [(960, 540), (192, 108), (1728, 972)]
        X = np.random.uniform(0.3, 0.7, size=(3, 11))

        from gaze import GazeRegressor
        reg = GazeRegressor("ridge", 1920, 1080)
        y = np.array(existing_targets, dtype=float)
        reg.fit(X, y)

        next_pt = self.engine.select_next_target(existing_targets, X, reg)
        self.assertIsInstance(next_pt, tuple)
        self.assertEqual(len(next_pt), 2)

    def test_should_stop_budget_exhausted(self):
        stop, reason = self.engine.should_stop([50.0, 45.0], sample_count=12, max_budget=12)
        self.assertTrue(stop)
        self.assertIn("BUDGET", reason)

    def test_should_stop_target_reached(self):
        stop, reason = self.engine.should_stop([30.0, 25.0, 20.0], sample_count=5, target_mae_px=30.0)
        self.assertTrue(stop)
        self.assertIn("TARGET", reason)

    def test_should_not_stop_early(self):
        stop, reason = self.engine.should_stop([100.0, 90.0], sample_count=3, max_budget=12)
        self.assertFalse(stop)
        self.assertEqual(reason, "CONTINUE")

    def test_should_stop_empty_errors(self):
        stop, reason = self.engine.should_stop([], sample_count=3)
        self.assertFalse(stop)

    def test_uncertainty_strategy(self):
        engine = AdaptiveCalibrationEngine(1920, 1080, strategy="uncertainty")
        existing_targets = [(960, 540), (192, 108)]
        X = np.random.uniform(0.3, 0.7, size=(2, 11))
        y = np.array(existing_targets, dtype=float)

        from gaze import GazeRegressor
        reg = GazeRegressor("ridge", 1920, 1080)

        # Should not crash, even with too-few samples for GP
        next_pt = engine.select_next_target(existing_targets, X, reg)
        self.assertIsInstance(next_pt, tuple)

    def test_error_strategy(self):
        engine = AdaptiveCalibrationEngine(1920, 1080, strategy="error")
        existing_targets = [(960, 540), (192, 108), (1728, 972), (192, 972)]
        X = np.random.uniform(0.3, 0.7, size=(4, 11))
        y = np.array(existing_targets, dtype=float)

        from gaze import GazeRegressor
        reg = GazeRegressor("ridge", 1920, 1080)
        reg.fit(X, y)

        next_pt = engine.select_next_target(existing_targets, X, reg)
        self.assertIsInstance(next_pt, tuple)


if __name__ == "__main__":
    unittest.main()
