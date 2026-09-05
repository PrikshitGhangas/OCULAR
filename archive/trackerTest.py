import cv2
import time
from camera import Camera
from tracker import FaceTracker


def main():
    print("=" * 60)
    print("OCULAR - Milestone 2 Tracker Verification")
    print("=" * 60)

    camera = Camera("/dev/video0")
    if not camera.open():
        print("[!] Failed to open camera device at /dev/video0. Retrying default index 0...")
        camera = Camera(0)
        if not camera.open():
            print("[ERROR] Unable to open camera.")
            return

    tracker = FaceTracker()
    print("[+] Camera and FaceTracker initialized successfully.")
    print("[i] Press 'q' to exit.")

    frame_count = 0
    start_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("[!] Frame acquisition dropped.")
                break

            found = tracker.process(frame)
            if found:
                tracker.draw_debug_overlay(frame)

                left_iris, right_iris = tracker.get_iris_centers(frame.shape)
                status_text = f"Face: DETECTED | Left Iris: {left_iris} | Right Iris: {right_iris}"
                color = (0, 255, 0)
            else:
                status_text = "Face: SEARCHING..."
                color = (0, 0, 255)

            # Draw HUD
            cv2.putText(
                frame,
                status_text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            frame_count += 1
            elapsed = time.perf_counter() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(
                frame,
                f"FPS: {current_fps:.1f}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("OCULAR - M2 Tracker Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        tracker.release()
        cv2.destroyAllWindows()

        elapsed = time.perf_counter() - start_time
        print("\nSession Summary:")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Elapsed Time: {elapsed:.2f} s")
        if elapsed > 0:
            print(f"Measured Throughput: {frame_count / elapsed:.2f} FPS")


if __name__ == "__main__":
    main()
