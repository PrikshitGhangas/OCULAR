import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from blink import BlinkDetector
from interaction import DwellDetector, GazeCursor, GazeScroller


class TestInteraction(unittest.TestCase):
    def test_dwell_detector_timing(self):
        # 300ms threshold, 50px radius, 500ms cooldown
        detector = DwellDetector(threshold_ms=300, radius_px=50, cooldown_ms=500)

        t = 1000.0
        # Frame 1: initial fixation at (500, 500)
        fired, progress = detector.update(500, 500, timestamp_ms=t)
        self.assertFalse(fired)
        self.assertEqual(progress, 0.0)

        # Frame 2: 150ms later, slightly moved (505, 502) -> 50% progress
        t += 150.0
        fired, progress = detector.update(505, 502, timestamp_ms=t)
        self.assertFalse(fired)
        self.assertAlmostEqual(progress, 0.5, places=1)

        # Frame 3: 310ms total elapsed -> fired!
        t += 160.0
        fired, progress = detector.update(504, 501, timestamp_ms=t)
        self.assertTrue(fired)
        self.assertEqual(progress, 1.0)

        # Frame 4: immediately after firing -> cooldown in effect
        t += 100.0
        fired, progress = detector.update(504, 501, timestamp_ms=t)
        self.assertFalse(fired)

    def test_gaze_scroller(self):
        scroller = GazeScroller(screen_h=1000, margin_ratio=0.20, max_speed=10)
        # Looking in middle (500) -> 0 speed
        self.assertEqual(scroller.update(500), 0)

        # Looking near top (50) -> positive speed (up)
        self.assertGreater(scroller.update(50), 0)

        # Looking near bottom (950) -> negative speed (down)
        self.assertLess(scroller.update(950), 0)

    def test_blink_detector_states(self):
        blink = BlinkDetector(ear_threshold=0.20, natural_min_frames=2, deliberate_min_frames=8)

        # Eyes open
        self.assertEqual(blink.update(0.30, 0.30), BlinkDetector.EVENT_NONE)

        # Natural blink: closed for 3 frames, then open
        blink.update(0.10, 0.10)
        blink.update(0.10, 0.10)
        blink.update(0.10, 0.10)
        event = blink.update(0.30, 0.30)
        self.assertEqual(event, BlinkDetector.EVENT_NATURAL_BLINK)

        # Deliberate blink: closed for 10 frames, then open
        for _ in range(10):
            blink.update(0.08, 0.08)
        event = blink.update(0.30, 0.30)
        self.assertEqual(event, BlinkDetector.EVENT_DELIBERATE_BLINK)


if __name__ == "__main__":
    unittest.main()
