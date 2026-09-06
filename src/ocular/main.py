import argparse
import os
import sys
import time
import cv2
import numpy as np

# Ensure module directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from .adaptive import AdaptiveCalibrationEngine
    from .blink import BlinkDetector
    from .calibration import CalibrationSession
    from .camera import Camera
    from .evaluation import EvaluationFramework
    from .features import FeatureExtractor
    from .gaze import GazeRegressor
    from .interaction import InteractionController
    from .tracker import FaceTracker
except ImportError:
    from adaptive import AdaptiveCalibrationEngine
    from blink import BlinkDetector
    from calibration import CalibrationSession
    from camera import Camera
    from evaluation import EvaluationFramework
    from features import FeatureExtractor
    from gaze import GazeRegressor
    from interaction import InteractionController
    from tracker import FaceTracker


def run_stream(args):
    """Run live visual tracking verification HUD."""
    print("[+] Starting Live Tracking Stream...")
    cam = Camera(args.camera)
    if not cam.open():
        print(f"[ERROR] Failed to open camera {args.camera}")
        return

    tracker = FaceTracker()
    extractor = FeatureExtractor()
    blink_det = BlinkDetector()

    print("[i] Streaming live. Press 'q' to exit.")
    frame_count = 0
    t0 = time.perf_counter()

    try:
        while True:
            success, frame = cam.read()
            if not success:
                break

            found = tracker.process(frame)
            if found:
                tracker.draw_debug_overlay(frame)
                features = extractor.extract(tracker.landmarks, frame.shape)

                if features is not None:
                    # Blink detection
                    l_ear, r_ear = features[4], features[5]
                    blink_event = blink_det.update(l_ear, r_ear)

                    # Draw Head Pose Axes
                    pitch, yaw, roll, rvec, tvec = extractor.estimate_head_pose(tracker.landmarks, frame.shape)
                    extractor.draw_head_pose_axes(frame, rvec, tvec)

                    # HUD text
                    cv2.putText(
                        frame,
                        f"Pose: P:{pitch:.1f} Y:{yaw:.1f} R:{roll:.1f}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"EAR L:{l_ear:.2f} R:{r_ear:.2f} | Event: {blink_event}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0) if blink_event == "NONE" else (0, 0, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "Face: SEARCHING...",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            frame_count += 1
            fps = frame_count / (time.perf_counter() - t0)
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("OCULAR - Live Tracking Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()
        tracker.release()
        cv2.destroyAllWindows()


def run_calibrate(args):
    """Run interactive calibration routine."""
    print(f"[+] Launching Calibration Session ({args.pattern})...")
    screen_w, screen_h = CalibrationSession.get_screen_resolution()

    cam = Camera(args.camera)
    if not cam.open():
        print(f"[ERROR] Failed to open camera {args.camera}")
        return

    tracker = FaceTracker()
    extractor = FeatureExtractor()
    session = CalibrationSession(screen_w, screen_h)

    pts = session.generate_grid(screen_w, screen_h, args.pattern)
    print(f"[+] Total Targets to Calibrate: {len(pts)}")

    X, y = session.run_interactive(
        cam,
        tracker,
        extractor,
        points=pts,
        dwell_seconds=args.dwell,
    )

    cam.release()
    tracker.release()

    if X is not None and len(X) > 0:
        if getattr(args, "session", None):
            out_file = args.session if args.session.endswith(".npz") else os.path.join("calibration", f"{args.session}.npz")
        else:
            out_file = args.output

        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        metadata = {
            "pattern": args.pattern,
            "screen_w": screen_w,
            "screen_h": screen_h,
            "timestamp": time.time(),
            "samples": len(X),
        }
        session.save_session(out_file, X, y, metadata)
        print(f"[SUCCESS] Calibration dataset saved ({len(X)} points) -> {out_file}")

        # Automatically evaluate LOOCV
        reg = GazeRegressor("ridge", screen_w, screen_h)
        metrics = reg.evaluate_loocv(X, y)
        print("\n--- Calibration Quality Metrics ---")
        print(f"Mean LOOCV Error: {metrics['mean_error_px']:.1f} px ({metrics['mean_error_deg']:.2f}°)")
        print(f"Median LOOCV Error: {metrics['median_error_px']:.1f} px ({metrics['median_error_deg']:.2f}°)")
        print(f"95th Percentile Error: {metrics['p95_error_px']:.1f} px")
    else:
        print("[!] Calibration aborted or insufficient data collected.")


def run_train(args):
    """Train gaze regression model from saved calibration session."""
    in_file = args.input
    if getattr(args, "session", None):
        in_file = args.session if args.session.endswith(".npz") else os.path.join("calibration", f"{args.session}.npz")

    if not os.path.exists(in_file):
        print(f"[ERROR] Calibration session file not found: {in_file}")
        print("[i] Please run 'ocular calibrate' first to collect calibration data.")
        return

    print(f"[+] Training Gaze Regressor from {in_file} (Model: {args.model})...")
    X, y, meta = CalibrationSession.load_session(in_file)

    screen_w = meta.get("screen_w", 1920)
    screen_h = meta.get("screen_h", 1080)

    reg = GazeRegressor(args.model, screen_w, screen_h)
    reg.fit(X, y)

    metrics = reg.evaluate_loocv(X, y)
    print("\n--- Cross-Validation Results ---")
    print(f"Mean Error: {metrics['mean_error_px']:.1f} px ({metrics['mean_error_deg']:.2f}°)")
    print(f"Median Error: {metrics['median_error_px']:.1f} px ({metrics['median_error_deg']:.2f}°)")
    print(f"P95 Error: {metrics['p95_error_px']:.1f} px")

    reg.save(args.output_dir)
    if getattr(args, "session", None):
        session_model_dir = os.path.join(args.output_dir, args.session)
        reg.save(session_model_dir)
    print(f"[SUCCESS] Trained model saved to: {args.output_dir}")


def run_interact(args):
    """Run real-time gaze interaction loop (cursor, dwell, scroll)."""
    print("[+] Initializing Real-Time Gaze Interaction...")
    screen_w, screen_h = CalibrationSession.get_screen_resolution()

    model_dir = args.model_dir
    if getattr(args, "session", None):
        candidate_dir = os.path.join(args.model_dir, args.session)
        if os.path.exists(candidate_dir):
            model_dir = candidate_dir

    reg = GazeRegressor(args.model, screen_w, screen_h)
    try:
        reg.load(model_dir)
        print(f"[+] Loaded trained {args.model.upper()} model from {model_dir}.")
    except Exception as e:
        print(f"[ERROR] Could not load model from {model_dir}: {e}")
        print("[i] Please run calibration and training first.")
        return

    cam = Camera(args.camera)
    if not cam.open():
        print(f"[ERROR] Failed to open camera {args.camera}")
        return

    tracker = FaceTracker()
    extractor = FeatureExtractor()
    blink_det = BlinkDetector()
    enable_os = args.enable_os_cursor or (getattr(args, "mode", "all") == "cursor")
    controller = InteractionController(screen_w, screen_h, enable_os_cursor=enable_os)

    print("[+] Interaction Loop active. Press 'q' to exit.")

    # Create interaction visualizer window
    vis_w, vis_h = 640, 360
    cv2.namedWindow("OCULAR - Gaze Visualizer", cv2.WINDOW_NORMAL)

    try:
        while True:
            success, frame = cam.read()
            if not success:
                break

            found = tracker.process(frame)
            if found:
                features = extractor.extract(tracker.landmarks, frame.shape)
                if features is not None:
                    # Blink detection
                    l_ear, r_ear = features[4], features[5]
                    blink_event = blink_det.update(l_ear, r_ear)

                    # Gaze prediction
                    raw_x, raw_y = reg.predict(features)

                    # Interaction processing
                    state = controller.process_frame(raw_x, raw_y, blink_event)

                    # Render mini desktop visualization HUD
                    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

                    # Map screen coords to mini canvas
                    cx = int(state["cursor_x"] / screen_w * vis_w)
                    cy = int(state["cursor_y"] / screen_h * vis_h)

                    # Dwell progress ring
                    if state["dwell_progress"] > 0.0:
                        ring_rad = max(4, int(20 * state["dwell_progress"]))
                        cv2.circle(vis, (cx, cy), ring_rad, (0, 165, 255), 2)

                    # Cursor point
                    color = (0, 255, 0) if state["dwell_triggered"] else (255, 255, 0)
                    cv2.circle(vis, (cx, cy), 5, color, -1)

                    # Scrolling margin visualizers
                    scroll_top_h = int(vis_h * 0.18)
                    cv2.line(vis, (0, scroll_top_h), (vis_w, scroll_top_h), (100, 100, 100), 1)
                    cv2.line(vis, (0, vis_h - scroll_top_h), (vis_w, vis_h - scroll_top_h), (100, 100, 100), 1)

                    hud = f"Gaze: ({int(state['cursor_x'])}, {int(state['cursor_y'])}) | Dwell: {int(state['dwell_progress']*100)}%"
                    cv2.putText(vis, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    cv2.imshow("OCULAR - Gaze Visualizer", vis)

            cv2.imshow("Camera View", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.release()
        tracker.release()
        cv2.destroyAllWindows()


def run_benchmark(args):
    """Run automated benchmarks."""
    import subprocess
    script = os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "compare_models.py")
    subprocess.run([sys.executable, script])


def main():
    parser = argparse.ArgumentParser(
        description="OCULAR: Real-Time Ocular Tracking and Gaze-Aware Interaction Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Stream
    p_stream = subparsers.add_parser("stream", help="Live facial & iris tracking diagnostic stream")
    p_stream.add_argument("--camera", default="0", help="Camera device index or path")

    # 2. Calibrate
    p_calib = subparsers.add_parser("calibrate", help="Execute visual calibration session")
    p_calib.add_argument("--camera", default="0", help="Camera device index or path")
    p_calib.add_argument("--pattern", default="9-point", choices=["5-point", "9-point", "13-point", "16-point"])
    p_calib.add_argument("--dwell", type=float, default=1.8, help="Dwell fixation duration per target in seconds")
    p_calib.add_argument("--session", "-s", default=None, help="Calibration session name (saves to calibration/<session>.npz)")
    p_calib.add_argument("--output", "-o", default="calibration/session_default.npz", help="Output calibration path")

    # 3. Train
    p_train = subparsers.add_parser("train", help="Train gaze regression model from calibration session")
    p_train.add_argument("--session", "-s", default=None, help="Calibration session name (loads calibration/<session>.npz)")
    p_train.add_argument("--input", "-i", default="calibration/session_default.npz", help="Input calibration file")
    p_train.add_argument("--model", "-m", default="rf", choices=["ridge", "svr", "rf", "mlp"], help="Regression model type")
    p_train.add_argument("--output-dir", default="models", help="Directory to save trained model")

    # 4. Interact
    p_interact = subparsers.add_parser("interact", help="Launch live gaze interaction (cursor, dwell, scroll)")
    p_interact.add_argument("--camera", default="0", help="Camera device index or path")
    p_interact.add_argument("--session", "-s", default=None, help="Calibration session name to load model from")
    p_interact.add_argument("--model", "-m", default="rf", choices=["ridge", "svr", "rf", "mlp"], help="Regression model type")
    p_interact.add_argument("--model-dir", default="models", help="Directory containing trained model")
    p_interact.add_argument("--mode", default="all", choices=["all", "cursor", "dwell", "scroll"], help="Interaction mode focus")
    p_interact.add_argument("--enable-os-cursor", action="store_true", help="Control physical OS mouse cursor")

    # 5. Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run model and calibration comparison benchmarks")

    args = parser.parse_args()

    if args.command == "stream":
        run_stream(args)
    elif args.command == "calibrate":
        run_calibrate(args)
    elif args.command == "train":
        run_train(args)
    elif args.command == "interact":
        run_interact(args)
    elif args.command == "benchmark":
        run_benchmark(args)


if __name__ == "__main__":
    main()
