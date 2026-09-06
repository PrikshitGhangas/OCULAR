import sys
import os
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from filters import OneEuroFilter, PointFilter2D, ExponentialMovingAverage, SimpleMovingAverage


class TestFilters(unittest.TestCase):
    def test_one_euro_filter_stability(self):
        f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.1)
        # Feed identical numbers - output should match
        val = 100.0
        t = 0.0
        for _ in range(10):
            res = f.filter(val, t)
            t += 1.0 / 30.0
        self.assertAlmostEqual(res, 100.0, places=2)

    def test_one_euro_filter_attenuates_jitter(self):
        f = OneEuroFilter(freq=30.0, mincutoff=0.5, beta=0.01)
        t = 0.0
        outputs = []
        for i in range(20):
            # Alternating jitter
            jitter = 5.0 if i % 2 == 0 else -5.0
            outputs.append(f.filter(100.0 + jitter, t))
            t += 1.0 / 30.0

        # Filtered jitter amplitude should be significantly lower than raw jitter (10.0)
        raw_spread = 10.0
        filtered_spread = max(outputs[5:]) - min(outputs[5:])
        self.assertLess(filtered_spread, raw_spread * 0.7)

    def test_point_filter_2d(self):
        pf = PointFilter2D()
        sx, sy = pf.filter(1920 / 2, 1080 / 2)
        self.assertAlmostEqual(sx, 960.0)
        self.assertAlmostEqual(sy, 540.0)

    def test_moving_averages(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        self.assertEqual(ema.filter(10.0), 10.0)
        self.assertEqual(ema.filter(20.0), 15.0)

        sma = SimpleMovingAverage(window_size=3)
        sma.filter(10.0)
        sma.filter(20.0)
        avg = sma.filter(30.0)
        self.assertEqual(avg, 20.0)


if __name__ == "__main__":
    unittest.main()
