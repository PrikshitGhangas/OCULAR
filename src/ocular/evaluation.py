import json
import os
import time
import numpy as np

try:
    from .gaze import GazeRegressor
except ImportError:
    from gaze import GazeRegressor


class EvaluationFramework:
    """
    Quantitative evaluation framework for OCULAR.

    Evaluates gaze estimation accuracy, comparative algorithm benchmarks,
    conventional vs adaptive calibration trade-offs, and computational latency.
    """

    def __init__(self, screen_w=1920, screen_h=1080):
        self.screen_w = screen_w
        self.screen_h = screen_h

    def benchmark_models(self, X, y, models=None):
        """
        Benchmark multiple regression models on the same calibration dataset.

        Args:
            X: Feature matrix (N, 11)
            y: Screen target coordinates (N, 2)
            models: List of model names to compare

        Returns:
            dict: Comparative benchmark results for each model
        """
        if models is None:
            models = [
                GazeRegressor.MODEL_RIDGE,
                GazeRegressor.MODEL_SVR,
                GazeRegressor.MODEL_RF,
                GazeRegressor.MODEL_MLP,
            ]

        results = {}
        for m_type in models:
            regressor = GazeRegressor(m_type, self.screen_w, self.screen_h)
            metrics = regressor.evaluate_loocv(X, y)
            results[m_type] = metrics

        return results

    def compare_conventional_vs_adaptive(
        self,
        X_conv,
        y_conv,
        conv_time_s,
        X_adapt,
        y_adapt,
        adapt_time_s,
        model_type=GazeRegressor.MODEL_RIDGE,
    ):
        """
        Compare conventional fixed calibration against adaptive calibration.

        Returns:
            dict: Comparative analysis between both methodologies
        """
        reg_conv = GazeRegressor(model_type, self.screen_w, self.screen_h)
        reg_adapt = GazeRegressor(model_type, self.screen_w, self.screen_h)

        metrics_conv = reg_conv.evaluate_loocv(X_conv, y_conv)
        metrics_adapt = reg_adapt.evaluate_loocv(X_adapt, y_adapt)

        analysis = {
            "conventional": {
                "samples": len(X_conv),
                "duration_seconds": conv_time_s,
                "mean_error_px": metrics_conv["mean_error_px"],
                "median_error_px": metrics_conv["median_error_px"],
                "p95_error_px": metrics_conv["p95_error_px"],
                "mean_error_deg": metrics_conv["mean_error_deg"],
            },
            "adaptive": {
                "samples": len(X_adapt),
                "duration_seconds": adapt_time_s,
                "mean_error_px": metrics_adapt["mean_error_px"],
                "median_error_px": metrics_adapt["median_error_px"],
                "p95_error_px": metrics_adapt["p95_error_px"],
                "mean_error_deg": metrics_adapt["mean_error_deg"],
            },
            "tradeoff_analysis": {
                "sample_reduction_ratio": float(1.0 - len(X_adapt) / max(1, len(X_conv))),
                "time_reduction_ratio": float(1.0 - adapt_time_s / max(0.1, conv_time_s)),
                "accuracy_difference_px": float(
                    metrics_adapt["mean_error_px"] - metrics_conv["mean_error_px"]
                ),
            },
        }

        return analysis

    def evaluate_test_points(self, regressor, test_features, ground_truth_targets):
        """
        Evaluate trained regressor on independent, held-out test points.

        Returns:
            dict: Evaluation metrics on unseen screen regions
        """
        preds = regressor.predict(test_features)
        diffs = preds - ground_truth_targets
        errors = np.linalg.norm(diffs, axis=1)

        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        p95_err = float(np.percentile(errors, 95))
        deg_err = regressor.pixels_to_degrees(mean_err)

        return {
            "test_sample_count": len(test_features),
            "mean_error_px": mean_err,
            "median_error_px": median_err,
            "p95_error_px": p95_err,
            "mean_error_deg": deg_err,
            "errors_per_point": errors.tolist(),
        }

    @staticmethod
    def save_experiment_results(filepath, report_data):
        """Save experiment report to JSON."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)
