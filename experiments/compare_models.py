import os
import sys
import numpy as np

# Ensure src/ocular is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from calibration import CalibrationSession
from evaluation import EvaluationFramework
from gaze import GazeRegressor


def main():
    print("=" * 65)
    print("OCULAR - Regression Model Benchmark Experiment")
    print("=" * 65)

    evaluator = EvaluationFramework(1920, 1080)

    # Check for recorded calibration data
    calib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "calibration"))
    files = [f for f in os.listdir(calib_dir) if f.endswith(".npz")] if os.path.exists(calib_dir) else []

    if files:
        sample_file = os.path.join(calib_dir, files[0])
        print(f"[+] Loading calibration dataset: {sample_file}")
        X, y, meta = CalibrationSession.load_session(sample_file)
    else:
        print("[i] No recorded session found. Generating synthetic calibration points...")
        np.random.seed(42)
        # 16-point grid ground truth
        pts = CalibrationSession.generate_grid(1920, 1080, "16-point")
        y = np.array(pts, dtype=np.float64)

        # Synthetic feature generation: iris ratios + head pose correlation + noise
        X = []
        for tx, ty in y:
            h_ratio = 0.35 + 0.30 * (tx / 1920.0) + np.random.normal(0, 0.01)
            v_ratio = 0.35 + 0.30 * (ty / 1080.0) + np.random.normal(0, 0.01)
            ear_l = 0.28 + np.random.normal(0, 0.01)
            ear_r = 0.28 + np.random.normal(0, 0.01)
            aspect_l = 2.4 + np.random.normal(0, 0.02)
            aspect_r = 2.4 + np.random.normal(0, 0.02)
            pitch = (ty - 540) / 1080.0 * 10.0 + np.random.normal(0, 0.2)
            yaw = (tx - 960) / 1920.0 * 15.0 + np.random.normal(0, 0.2)
            roll = np.random.normal(0, 0.2)
            X.append([h_ratio, v_ratio, h_ratio, v_ratio, ear_l, ear_r, aspect_l, aspect_r, pitch, yaw, roll])
        X = np.array(X, dtype=np.float64)

    print(f"[+] Calibration Samples: {len(X)}")
    print("\nRunning Leave-One-Out Cross-Validation across candidate models...")

    models = [
        GazeRegressor.MODEL_RIDGE,
        GazeRegressor.MODEL_SVR,
        GazeRegressor.MODEL_RF,
        GazeRegressor.MODEL_MLP,
    ]

    results = evaluator.benchmark_models(X, y, models)

    print("\n" + "-" * 75)
    print(f"{'Model':<15} | {'Mean Err (px)':<14} | {'Median Err (px)':<16} | {'P95 (px)':<10} | {'Visual Ang (deg)'}")
    print("-" * 75)

    for m_name, res in results.items():
        print(
            f"{m_name.upper():<15} | {res['mean_error_px']:<14.2f} | "
            f"{res['median_error_px']:<16.2f} | {res['p95_error_px']:<10.2f} | "
            f"{res['mean_error_deg']:.2f}°"
        )
    print("-" * 75)

    # Save to data/experiments
    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "experiments", "model_benchmark.json")
    )
    evaluator.save_experiment_results(out_path, results)
    print(f"\n[+] Detailed benchmark report saved to: {out_path}")


if __name__ == "__main__":
    main()
