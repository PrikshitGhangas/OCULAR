import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from camera import Camera


class TestCamera(unittest.TestCase):
    def test_camera_init_default(self):
        cam = Camera()
        self.assertEqual(cam.source, 0)
        self.assertEqual(cam.width, 1280)
        self.assertEqual(cam.height, 720)
        self.assertEqual(cam.fps, 30)
        self.assertIsNone(cam.capture)

    def test_camera_init_string_index(self):
        cam = Camera("0")
        self.assertEqual(cam.source, 0)  # Should convert to int

    def test_camera_init_device_path(self):
        cam = Camera("/dev/video0")
        self.assertEqual(cam.source, "/dev/video0")

    def test_read_without_open(self):
        cam = Camera()
        success, frame = cam.read()
        self.assertFalse(success)
        self.assertIsNone(frame)

    def test_release_without_open(self):
        cam = Camera()
        cam.release()  # Should not raise
        self.assertIsNone(cam.capture)

    def test_is_opened_without_open(self):
        cam = Camera()
        self.assertFalse(cam.is_opened())

    def test_context_manager(self):
        cam = Camera(999)  # Non-existent camera
        with cam:
            pass  # Should not raise even if camera doesn't exist
        self.assertIsNone(cam.capture)


if __name__ == "__main__":
    unittest.main()
