import os
import sys
import numpy as np

# Ensure src/ocular is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from adaptive import AdaptiveCalibrationEngine
from calibration import CalibrationSession
from evaluation import EvaluationFramework
from gaze import GazeRegressor


def simulate_features(targets):
    """Generate realistic feature vectors corresponding to screen targets."""
    X = []
    for tx, ty in targets:
        h_ratio = 0.35 + 0.30 * (tx / 1920.0) + np.random.normal(0, 0.008)
        v_ratio = 0.35 + 0.30 * (ty / 1080.0) + np.random.normal(0, 0.008)
        ear_l = 0.28 + np.random.normal(0, 0.008)
        ear_r = 0.28 + np.random.normal(0, 0.008)
        aspect_l = 2.4 + np.random.normal(0, 0.01)
        aspect_r = 2.4 + np.random.normal(0, 0.01)
        pitch = (ty - 540) / 1080.0 * 10.0 + np.random.normal(0, 0.15)
        yaw = (tx - 960) / 1920.0 * 15.0 + np.random.normal(0, 0.15)
        roll = np.random.normal(0, 0.15)
        X.append([h_ratio, v_ratio, h_ratio, v_ratio, ear_l, ear_r, aspect_l, aspect_r, pitch, yaw, roll])
    return np.array(X, dtype=np.float64)


def main():
    print("=" * 70)
    print("OCULAR - Conventional vs. Adaptive Calibration Experiment")
    print("=" * 70)

    np.random.seed(42)
    evaluator = EvaluationFramework(1920, 1080)
    engine = AdaptiveCalibrationEngine(1920, 1080, strategy="hybrid")

    # 1. Conventional Calibration (Fixed 9-Point Grid)
    print("[1] Executing Conventional Calibration (Fixed 9-Point Grid)...")
    conv_targets = CalibrationSession.generate_grid(1920, 1080, "9-point")
    X_conv = simulate_features(conv_targets)
    y_conv = np.array(conv_targets, dtype=np.float64)
    conv_duration_s = 9 * 2.0  # 18.0 seconds

    # 2. Adaptive Calibration (Initial 5 Points + Active Sampling)
    print("[2] Executing Adaptive Calibration (5 Initial Points + Active Discovery)...")
    adapt_targets = list(CalibrationSession.generate_grid(1920, 1080, "5-point"))
    X_adapt = list(simulate_features(adapt_targets))
    y_adapt = [list(pt) for pt in adapt_targets]

    current_regressor = GazeRegressor("ridge", 1920, 1080)
    max_budget = 9
    pool = engine.generate_candidate_pool(grid_rows=6, grid_cols=6)

    while len(y_adapt) < max_budget:
        current_regressor.fit(np.array(X_adapt), np.array(y_adapt))
        loocv = current_regressor.evaluate_loocv(np.array(X_adapt), np.array(y_adapt))

        should_stop, reason = engine.should_stop(loocv["point_errors_px"], len(y_adapt), max_budget, target_mae_px=45.0)
        if should_stop:
            print(f"    -> Adaptive stopping condition triggered: {reason}")
            break

        next_target = engine.select_next_target(y_adapt, np.array(X_adapt), current_regressor, pool)
        adapt_targets.append(next_target)
        new_feature = simulate_features([next_target])[0]

        X_adapt.append(new_feature)
        y_adapt.append(list(next_target))
        print(f"    -> Dynamically sampled target {len(y_adapt)}: {next_target}")

    X_adapt = np.array(X_adapt, dtype=np.float64)
    y_adapt = np.array(y_adapt, dtype=np.float64)
    adapt_duration_s = len(y_adapt) * 2.0

    # 3. Comparative Evaluation
    print("\n[3] Computing Comparative Trade-off Analysis...")
    analysis = evaluator.compare_conventional_vs_adaptive(
        X_conv, y_conv, conv_duration_s,
        X_adapt, y_adapt, adapt_duration_s,
        model_type=GazeRegressor.MODEL_RIDGE
    )

    print("\n" + "=" * 70)
    print(f"{'Metric':<30} | {'Conventional':<16} | {'Adaptive':<16}")
    print("-" * 70)
    print(f"{'Sample Count':<30} | {analysis['conventional']['samples']:<16} | {analysis['adaptive']['samples']:<16}")
    print(f"{'Calibration Time (s)':<30} | {analysis['conventional']['duration_seconds']:<16.1f} | {analysis['adaptive']['duration_seconds']:<16.1f}")
    print(f"{'Mean Error (px)':<30} | {analysis['conventional']['mean_error_px']:<16.2f} | {analysis['adaptive']['mean_error_px']:<16.2f}")
    print(f"{'Median Error (px)':<30} | {analysis['conventional']['median_error_px']:<16.2f} | {analysis['adaptive']['median_error_px']:<16.2f}")
    print(f"{'P95 Error (px)':<30} | {analysis['conventional']['p95_error_px']:<16.2f} | {analysis['adaptive']['p95_error_px']:<16.2f}")
    print(f"{'Visual Angle Error':<30} | {analysis['conventional']['mean_error_deg']:<16.2f}°| {analysis['adaptive']['mean_error_deg']:<16.2f}°")
    print("=" * 70)

    # Save to data/experiments
    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "experiments", "calibration_comparison.json")
    )
    evaluator.save_experiment_results(out_path, analysis)
    print(f"\n[+] Detailed experimental comparison saved to: {out_path}")


if __name__ == "__main__":
    main()
