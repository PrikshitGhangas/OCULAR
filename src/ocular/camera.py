import cv2
import platform
import shutil
import subprocess


class Camera:
    """
    Cross-platform camera acquisition abstraction.

    Provides a clean, uniform interface for camera streaming on both Linux (V4L2)
    and Windows (DirectShow/Default), maintaining consistent frame resolution,
    fourcc compression, and framerate configurations.
    """

    def __init__(self, source=0, width=1280, height=720, fps=30):
        """
        Initialize Camera configuration.

        Args:
            source: Device index (int, e.g., 0) or device path (str, e.g., "/dev/video0")
            width: Target frame width in pixels
            height: Target frame height in pixels
            fps: Target frames per second
        """
        # Parse source: if given string of integer digits, convert to int
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        self.width = width
        self.height = height
        self.fps = fps
        self.capture = None
        self.system = platform.system()

    def open(self):
        """
        Initialize camera device with platform-specific optimizations.

        Returns:
            bool: True if camera opened successfully, False otherwise.
        """
        # Linux V4L2 optimization (disable dynamic exposure framerate drops)
        if self.system == "Linux" and isinstance(self.source, str):
            if shutil.which("v4l2-ctl"):
                try:
                    subprocess.run(
                        [
                            "v4l2-ctl",
                            "-d",
                            self.source,
                            "--set-ctrl=exposure_dynamic_framerate=0",
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass

        # Select appropriate OpenCV capture backend
        if self.system == "Linux":
            backend = cv2.CAP_V4L2
        elif self.system == "Windows":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_ANY

        # Open video capture
        if isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source, backend)
        else:
            self.capture = cv2.VideoCapture(self.source, backend)

        # Fallback to default backend if platform backend failed
        if not self.capture.isOpened():
            self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            return False

        # Apply camera stream properties
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.capture.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        return self.capture.isOpened()

    def is_opened(self):
        """Check if camera capture is currently open and valid."""
        return self.capture is not None and self.capture.isOpened()

    def read(self):
        """
        Read the next frame from the camera stream.

        Returns:
            tuple (bool, numpy.ndarray): (success flag, image frame in BGR format)
        """
        if self.capture is None:
            return False, None
        return self.capture.read()

    def release(self):
        """Release underlying camera device and hardware resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()