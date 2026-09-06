import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from blink import BlinkDetector


class TestBlinkDetector(unittest.TestCase):
    def setUp(self):
        self.det = BlinkDetector()

    def test_open_eyes_no_event(self):
        event = self.det.update(0.30, 0.30)
        self.assertEqual(event, "NONE")

    def test_none_ear_returns_none_event(self):
        event = self.det.update(None, None)
        self.assertEqual(event, "NONE")
        event = self.det.update(0.30, None)
        self.assertEqual(event, "NONE")

    def test_natural_blink_detection(self):
        self.det.update(0.10, 0.10)  # Frame 1: closed
        self.det.update(0.10, 0.10)  # Frame 2: closed
        event = self.det.update(0.30, 0.30)  # Frame 3: reopen
        self.assertEqual(event, "NATURAL_BLINK")

    def test_deliberate_blink_detection(self):
        for _ in range(8):
            self.det.update(0.10, 0.10)
        event = self.det.update(0.30, 0.30)
        self.assertEqual(event, "DELIBERATE_BLINK")

    def test_left_wink_detection(self):
        for _ in range(3):
            self.det.update(0.10, 0.30)
        event = self.det.update(0.30, 0.30)
        self.assertEqual(event, "LEFT_WINK")

    def test_right_wink_detection(self):
        for _ in range(3):
            self.det.update(0.30, 0.10)
        event = self.det.update(0.30, 0.30)
        self.assertEqual(event, "RIGHT_WINK")

    def test_is_blinking(self):
        self.assertFalse(self.det.is_blinking(0.30, 0.30))
        self.assertTrue(self.det.is_blinking(0.10, 0.10))
        self.assertTrue(self.det.is_blinking(0.10, 0.30))
        self.assertFalse(self.det.is_blinking(None, 0.30))

    def test_single_frame_close_no_event(self):
        self.det.update(0.10, 0.10)  # 1 frame closed
        event = self.det.update(0.30, 0.30)  # Reopen
        self.assertEqual(event, "NONE")


if __name__ == "__main__":
    unittest.main()
