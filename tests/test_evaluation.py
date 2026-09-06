import os
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from evaluation import EvaluationFramework
from gaze import GazeRegressor


class TestEvaluation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 12
        self.X = np.random.uniform(0.3, 0.7, size=(self.n, 11))
        target_x = self.X[:, 0] * 1920.0
        target_y = self.X[:, 1] * 1080.0
        self.y = np.column_stack((target_x, target_y))
        self.evaluator = EvaluationFramework(1920, 1080)

    def test_benchmark_models(self):
        results = self.evaluator.benchmark_models(self.X, self.y, models=["ridge", "rf"])
        self.assertIn("ridge", results)
        self.assertIn("rf", results)
        self.assertGreater(results["ridge"]["mean_error_px"], 0)

    def test_evaluate_test_points(self):
        reg = GazeRegressor("ridge", 1920, 1080)
        reg.fit(self.X, self.y)
        metrics = self.evaluator.evaluate_test_points(reg, self.X[:3], self.y[:3])
        self.assertIn("mean_error_px", metrics)
        self.assertEqual(metrics["test_sample_count"], 3)

    def test_save_experiment_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            self.evaluator.save_experiment_results(path, {"test": "data"})
            self.assertTrue(os.path.exists(path))

    def test_compare_conventional_vs_adaptive(self):
        np.random.seed(42)
        X1 = np.random.uniform(0.3, 0.7, size=(9, 11))
        y1 = np.column_stack((X1[:, 0] * 1920, X1[:, 1] * 1080))
        X2 = np.random.uniform(0.3, 0.7, size=(6, 11))
        y2 = np.column_stack((X2[:, 0] * 1920, X2[:, 1] * 1080))
        analysis = self.evaluator.compare_conventional_vs_adaptive(
            X1, y1, 18.0, X2, y2, 12.0
        )
        self.assertIn("conventional", analysis)
        self.assertIn("adaptive", analysis)
        self.assertIn("tradeoff_analysis", analysis)


if __name__ == "__main__":
    unittest.main()
