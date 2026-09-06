import os
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from profiler import PerformanceProfiler


class TestPerformanceProfiler(unittest.TestCase):
    def test_basic_timing(self):
        p = PerformanceProfiler()
        p.start("test_stage")
        time.sleep(0.01)
        elapsed = p.stop("test_stage")
        self.assertGreater(elapsed, 5.0)  # At least 5ms
        self.assertLess(elapsed, 200.0)   # Under 200ms

    def test_context_manager(self):
        p = PerformanceProfiler()
        with p.measure("ctx_stage"):
            time.sleep(0.01)
        stats = p.get_stats("ctx_stage")
        self.assertEqual(stats["samples"], 1)
        self.assertGreater(stats["mean_ms"], 5.0)

    def test_multiple_samples(self):
        p = PerformanceProfiler()
        for _ in range(5):
            with p.measure("multi"):
                pass
        stats = p.get_stats("multi")
        self.assertEqual(stats["samples"], 5)

    def test_window_size(self):
        p = PerformanceProfiler(window_size=3)
        for _ in range(10):
            with p.measure("windowed"):
                pass
        stats = p.get_stats("windowed")
        self.assertEqual(stats["samples"], 3)

    def test_get_all_stats(self):
        p = PerformanceProfiler()
        with p.measure("a"):
            pass
        with p.measure("b"):
            pass
        all_stats = p.get_all_stats()
        self.assertIn("a", all_stats)
        self.assertIn("b", all_stats)

    def test_fps_calculation(self):
        p = PerformanceProfiler()
        with p.measure("stage"):
            time.sleep(0.01)
        fps = p.get_fps()
        self.assertGreater(fps, 0)
        self.assertLess(fps, 200)

    def test_empty_stats(self):
        p = PerformanceProfiler()
        stats = p.get_stats("nonexistent")
        self.assertEqual(stats["samples"], 0)

    def test_reset(self):
        p = PerformanceProfiler()
        with p.measure("test"):
            pass
        p.reset()
        self.assertEqual(len(p.get_all_stats()), 0)

    def test_stop_without_start(self):
        p = PerformanceProfiler()
        elapsed = p.stop("never_started")
        self.assertEqual(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
