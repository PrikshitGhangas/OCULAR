import cv2


def find_pupil(eye):
    """
    Find a pupil candidate inside an eye ROI.

    Returns:
        (x, y, radius) in eye-ROI coordinates,
        or None if no suitable pupil is found.
    """

    if eye is None or eye.size == 0:
        return None

    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    # Reduce small image noise.
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dark regions become white.
    _, threshold = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    best = None
    best_score = 0

    eye_area = eye.shape[0] * eye.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore tiny noise and enormous regions.
        if area < 20:
            continue

        if area > eye_area * 0.5:
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = (
            4 * 3.14159265 * area /
            (perimeter * perimeter)
        )

        if circularity < 0.35:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)

        if radius < 2 or radius > min(eye.shape[:2]) * 0.35:
            continue

        # Prefer reasonably circular, compact dark regions.
        score = area * circularity

        if score > best_score:
            best_score = score
            best = (
                int(x),
                int(y),
                int(radius)
            )

    return best