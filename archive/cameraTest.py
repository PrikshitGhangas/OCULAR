import cv2
import time
from camera import Camera

camera = Camera("/dev/video0")

if not camera.open():
    print("Failed to open camera")
    exit()

# YuNet
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

    # -----------------
    # Camera
    # -----------------
    read_start = time.perf_counter()

    success, frame = camera.read()

    read_end = time.perf_counter()

    if not success:
        print("Failed to read frame")
        break

    total_read += read_end - read_start

    # -----------------
    # Resize for YuNet
    # -----------------
    small_frame = cv2.resize(
        frame,
        (640, 360)
    )

    # -----------------
    # Face detection
    # -----------------
    detect_start = time.perf_counter()

    _, faces = detector.detect(small_frame)

    detect_end = time.perf_counter()

    total_detect += detect_end - detect_start

    # -----------------
    # Draw detections
    # -----------------
    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]

            # Convert coordinates from
            # 640x360 → 1280x720
            x *= 2
            y *= 2
            w *= 2
            h *= 2

            x, y, w, h = map(int, (x, y, w, h))

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    # -----------------
    # Display
    # -----------------
    cv2.imshow(
        "OCULAR - Face Detection",
        frame
    )

    frames += 1

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

end = time.perf_counter()

camera.release()
cv2.destroyAllWindows()

# -----------------
# Results
# -----------------

elapsed = end - start

print()
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