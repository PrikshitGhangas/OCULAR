"""
OCULAR User Study Runner.

Orchestrates a complete user study session:
1. Conventional calibration (fixed grid)
2. Adaptive calibration (active learning)
3. Test point evaluation for both
4. Model comparison (Ridge, RF)
5. Results logged to JSON

Usage:
    python -m experiments.run_user_study --participant P01 --camera 0
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from calibration import CalibrationSession
from camera import Camera
from features import FeatureExtractor
from gaze import GazeRegressor
from tracker import FaceTracker


def get_hardware_info():
    """Collect hardware and environment metadata."""
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }


def evaluate_on_test_points(cam, tracker, extractor, regressor, test_points, screen_w, screen_h, dwell_seconds=2.0):
    """
    Run test point evaluation: display each test point, collect gaze features,
    predict gaze, and compute per-point error.
    """
    session = CalibrationSession(screen_w, screen_h)
    X_test, y_test = session.run_interactive(
        cam, tracker, extractor, points=test_points, dwell_seconds=dwell_seconds,
    )

    if X_test is None or len(X_test) == 0:
        return None

    errors = []
    predictions = []
    for i in range(len(X_test)):
        pred = regressor.predict(X_test[i])
        actual = y_test[i]
        err_px = float(np.linalg.norm(pred - actual))
        err_h = float(abs(pred[0] - actual[0]))
        err_v = float(abs(pred[1] - actual[1]))
        errors.append({
            "point_idx": i,
            "target": [float(actual[0]), float(actual[1])],
            "predicted": [float(pred[0]), float(pred[1])],
            "error_px": round(err_px, 2),
            "error_h_px": round(err_h, 2),
            "error_v_px": round(err_v, 2),
            "error_deg": round(GazeRegressor.pixels_to_degrees(err_px), 3),
        })
        predictions.append(pred)

    error_values = [e["error_px"] for e in errors]
    summary = {
        "n_test_points": len(errors),
        "mean_error_px": round(float(np.mean(error_values)), 2),
        "median_error_px": round(float(np.median(error_values)), 2),
        "std_error_px": round(float(np.std(error_values)), 2),
        "p95_error_px": round(float(np.percentile(error_values, 95)), 2),
        "min_error_px": round(float(np.min(error_values)), 2),
        "max_error_px": round(float(np.max(error_values)), 2),
        "mean_error_deg": round(float(np.mean([e["error_deg"] for e in errors])), 3),
        "mean_h_error_px": round(float(np.mean([e["error_h_px"] for e in errors])), 2),
        "mean_v_error_px": round(float(np.mean([e["error_v_px"] for e in errors])), 2),
        "per_point": errors,
    }
    return summary


def run_condition(cam, tracker, extractor, condition_name, calib_points, test_points,
                  screen_w, screen_h, model_types, dwell_seconds):
    """Run a single calibration condition and evaluate."""
    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  Calibration points: {len(calib_points)}")
    print(f"{'='*60}")

    session = CalibrationSession(screen_w, screen_h)
    X_calib, y_calib = session.run_interactive(
        cam, tracker, extractor, points=calib_points, dwell_seconds=dwell_seconds,
    )

    if X_calib is None or len(X_calib) < 3:
        print(f"[!] {condition_name}: insufficient calibration data.")
        return None

    results = {
        "condition": condition_name,
        "n_calibration_points": len(X_calib),
        "models": {},
    }

    for model_type in model_types:
        print(f"  Training {model_type} model...")
        reg = GazeRegressor(model_type, screen_w, screen_h)
        reg.fit(X_calib, y_calib)

        # LOOCV on training data
        loocv = reg.evaluate_loocv(X_calib, y_calib)

        # Test point evaluation
        print(f"  Evaluating on {len(test_points)} test points...")
        test_results = evaluate_on_test_points(
            cam, tracker, extractor, reg, test_points, screen_w, screen_h, dwell_seconds
        )

        results["models"][model_type] = {
            "loocv": {
                "mean_error_px": round(loocv["mean_error_px"], 2),
                "median_error_px": round(loocv["median_error_px"], 2),
                "mean_error_deg": round(loocv["mean_error_deg"], 3),
                "p95_error_px": round(loocv["p95_error_px"], 2),
            },
            "test_points": test_results,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="OCULAR User Study Runner")
    parser.add_argument("--participant", "-p", required=True, help="Participant ID (e.g., P01)")
    parser.add_argument("--camera", default="0", help="Camera device")
    parser.add_argument("--dwell", type=float, default=2.0, help="Dwell duration per point")
    parser.add_argument("--output-dir", default="data/experiments", help="Results output directory")
    parser.add_argument("--models", nargs="+", default=["ridge", "rf"], help="Models to compare")
    args = parser.parse_args()

    screen_w, screen_h = CalibrationSession.get_screen_resolution()
    print(f"[+] OCULAR User Study")
    print(f"[+] Participant: {args.participant}")
    print(f"[+] Screen: {screen_w}x{screen_h}")

    cam = Camera(args.camera)
    if not cam.open():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    tracker = FaceTracker()
    extractor = FeatureExtractor()
    session = CalibrationSession(screen_w, screen_h)

    # Generate test points (4x4 grid, different from calibration grids)
    test_points = session.generate_grid(screen_w, screen_h, "16-point")

    study_results = {
        "participant": args.participant,
        "timestamp": time.time(),
        "hardware": get_hardware_info(),
        "screen_w": screen_w,
        "screen_h": screen_h,
        "camera": args.camera,
        "dwell_seconds": args.dwell,
        "conditions": [],
    }

    # Condition 1: 9-point conventional calibration
    calib_9 = session.generate_grid(screen_w, screen_h, "9-point")
    result_conv = run_condition(
        cam, tracker, extractor,
        "conventional_9point", calib_9, test_points,
        screen_w, screen_h, args.models, args.dwell
    )
    if result_conv:
        study_results["conditions"].append(result_conv)

    input("\n[PAUSE] Press Enter when ready for next condition...")

    # Condition 2: 16-point conventional calibration (uses same test points)
    calib_16 = session.generate_grid(screen_w, screen_h, "16-point")
    result_16 = run_condition(
        cam, tracker, extractor,
        "conventional_16point", calib_16, test_points,
        screen_w, screen_h, args.models, args.dwell
    )
    if result_16:
        study_results["conditions"].append(result_16)

    cam.release()
    tracker.release()

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"study_{args.participant}_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(study_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Study Complete!")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")

    # Print summary
    for cond in study_results["conditions"]:
        print(f"\n  {cond['condition']} ({cond['n_calibration_points']} cal points):")
        for model, data in cond["models"].items():
            tp = data.get("test_points")
            if tp:
                print(f"    {model}: {tp['mean_error_px']:.1f}px mean, {tp['median_error_px']:.1f}px median, {tp['mean_error_deg']:.3f}°")


if __name__ == "__main__":
    main()
