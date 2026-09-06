"""
OCULAR Feature Ablation Study Script.

Runs LOOCV evaluation with different feature subsets to determine
which components contribute most to gaze estimation accuracy.

Usage:
    python -m experiments.feature_ablation --input calibration/session.npz
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from gaze import GazeRegressor


# Feature indices and names (11-D feature vector)
FEATURE_NAMES = [
    "left_iris_h",    # 0
    "left_iris_v",    # 1
    "right_iris_h",   # 2
    "right_iris_v",   # 3
    "left_ear",       # 4
    "right_ear",      # 5
    "left_eye_aspect",# 6
    "right_eye_aspect",# 7
    "head_pitch",     # 8
    "head_yaw",       # 9
    "head_roll",      # 10
]

# Ablation conditions: name -> feature indices
ABLATION_CONDITIONS = {
    "iris_only":       [0, 1, 2, 3],
    "iris_ear":        [0, 1, 2, 3, 4, 5],
    "iris_head":       [0, 1, 2, 3, 8, 9, 10],
    "iris_ear_head":   [0, 1, 2, 3, 4, 5, 8, 9, 10],
    "all_features":    list(range(11)),
}


def run_ablation(X, y, screen_w, screen_h, model_type="ridge"):
    """Run LOOCV for each feature subset condition."""
    results = {}

    for condition_name, feature_indices in ABLATION_CONDITIONS.items():
        X_subset = X[:, feature_indices]
        feature_names_used = [FEATURE_NAMES[i] for i in feature_indices]

        reg = GazeRegressor(model_type, screen_w, screen_h)
        metrics = reg.evaluate_loocv(X_subset, y)

        results[condition_name] = {
            "features": feature_names_used,
            "n_features": len(feature_indices),
            "mean_error_px": round(metrics["mean_error_px"], 2),
            "median_error_px": round(metrics["median_error_px"], 2),
            "mean_error_deg": round(metrics["mean_error_deg"], 3),
            "p95_error_px": round(metrics["p95_error_px"], 2),
        }

    return results


def print_ablation_table(results):
    """Print formatted ablation results table."""
    print(f"\n{'Condition':<20} {'#Feat':>5} {'Mean(px)':>10} {'Med(px)':>10} {'P95(px)':>10} {'Mean(°)':>10}")
    print("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["mean_error_px"]):
        print(
            f"{name:<20} {r['n_features']:>5} "
            f"{r['mean_error_px']:>10.1f} {r['median_error_px']:>10.1f} "
            f"{r['p95_error_px']:>10.1f} {r['mean_error_deg']:>10.3f}"
        )


def main():
    parser = argparse.ArgumentParser(description="OCULAR Feature Ablation Study")
    parser.add_argument("--input", "-i", required=True, help="Path to calibration .npz file")
    parser.add_argument("--model", "-m", default="ridge", choices=["ridge", "svr", "rf"])
    parser.add_argument("--output", "-o", default=None, help="Output JSON path for results")
    args = parser.parse_args()

    # Load calibration data
    data = np.load(args.input, allow_pickle=False)
    X = data["X"]
    y = data["y"]
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}

    screen_w = metadata.get("screen_w", 1920)
    screen_h = metadata.get("screen_h", 1080)

    print(f"[+] Loaded {len(X)} calibration samples from {args.input}")
    print(f"[+] Screen: {screen_w}x{screen_h}, Model: {args.model}")
    print(f"[+] Running {len(ABLATION_CONDITIONS)} ablation conditions...")

    results = run_ablation(X, y, screen_w, screen_h, args.model)
    print_ablation_table(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "input_file": args.input,
                "model_type": args.model,
                "n_samples": len(X),
                "screen_w": screen_w,
                "screen_h": screen_h,
                "conditions": results,
            }, f, indent=2)
        print(f"\n[+] Results saved to {args.output}")


if __name__ == "__main__":
    main()
