import cv2
import time
from camera import Camera

camera = Camera("/dev/video0")

if not camera.open():
    print("Failed to open camera")
    exit()

detector = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2026may.onnx",
    "",
    (640, 360)
)

detector.setInputSize((640, 360))

frames = 0
total_read = 0
total_detect = 0

start = time.perf_counter()

while frames < 100:

    # -------------------------
    # Camera
    # -------------------------
    read_start = time.perf_counter()

    success, frame = camera.read()

    read_end = time.perf_counter()

    if not success:
        print("Failed to read frame")
        break

    total_read += read_end - read_start

    # -------------------------
    # Resize
    # -------------------------
    small_frame = cv2.resize(
        frame,
        (640, 360)
    )

    # -------------------------
    # YuNet
    # -------------------------
    detect_start = time.perf_counter()

    _, faces = detector.detect(small_frame)

    detect_end = time.perf_counter()

    total_detect += detect_end - detect_start

    # -------------------------
    # Eye ROI calculation
    # -------------------------
    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]

            scale_x = frame.shape[1] / 640
            scale_y = frame.shape[0] / 360

            x = int(x * scale_x)
            y = int(y * scale_y)
            w = int(w * scale_x)
            h = int(h * scale_y)

            eye_y = y + int(h * 0.25)
            eye_h = int(h * 0.30)

            left_x = x + int(w * 0.10)
            left_w = int(w * 0.35)

            right_x = x + int(w * 0.55)
            right_w = int(w * 0.35)

            # Face
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            # Left eye
            cv2.rectangle(
                frame,
                (left_x, eye_y),
                (left_x + left_w, eye_y + eye_h),
                (0, 255, 0),
                2
            )

            # Right eye
            cv2.rectangle(
                frame,
                (right_x, eye_y),
                (right_x + right_w, eye_y + eye_h),
                (0, 255, 0),
                2
            )

    # -------------------------
    # Display
    # -------------------------
    cv2.imshow("OCULAR - Performance Test", frame)

    frames += 1

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

end = time.perf_counter()

camera.release()
cv2.destroyAllWindows()

# -------------------------
# Results
# -------------------------

elapsed = end - start

print()
print("========== OCULAR PERFORMANCE ==========")
print(f"Frames: {frames}")
print(f"Total time: {elapsed:.3f} s")
print(f"Measured FPS: {frames / elapsed:.3f}")
print(
    f"Average camera.read(): "
    f"{(total_read / frames) * 1000:.3f} ms"
)
print(
    f"Average YuNet detect(): "
    f"{(total_detect / frames) * 1000:.3f} ms"
)
print("=========================================")