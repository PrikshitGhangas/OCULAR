"""
OCULAR Performance Profiler.

Provides per-stage timing instrumentation for the gaze tracking pipeline.

Usage:
    profiler = PerformanceProfiler()
    with profiler.measure("camera_read"):
        frame = cam.read()
    with profiler.measure("landmark_detection"):
        landmarks = tracker.process(frame)
    profiler.report()
"""

import time
from collections import defaultdict


class PerformanceProfiler:
    """
    Lightweight per-stage latency profiler for the OCULAR pipeline.

    Collects timing samples for named stages and reports statistics.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self._timings = defaultdict(list)
        self._start_times = {}

    def start(self, stage_name):
        """Begin timing a named stage."""
        self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name):
        """End timing a named stage and record the duration."""
        if stage_name not in self._start_times:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_times[stage_name]) * 1000.0
        del self._start_times[stage_name]

        timings = self._timings[stage_name]
        timings.append(elapsed_ms)
        if len(timings) > self.window_size:
            timings.pop(0)

        return elapsed_ms

    class _Timer:
        """Context manager for stage timing."""
        def __init__(self, profiler, stage_name):
            self.profiler = profiler
            self.stage_name = stage_name

        def __enter__(self):
            self.profiler.start(self.stage_name)
            return self

        def __exit__(self, *args):
            self.profiler.stop(self.stage_name)

    def measure(self, stage_name):
        """Return a context manager that times the given stage."""
        return self._Timer(self, stage_name)

    def get_stats(self, stage_name):
        """Get timing statistics for a stage."""
        timings = self._timings.get(stage_name, [])
        if not timings:
            return {"mean_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0, "samples": 0}

        return {
            "mean_ms": sum(timings) / len(timings),
            "min_ms": min(timings),
            "max_ms": max(timings),
            "samples": len(timings),
        }

    def get_all_stats(self):
        """Get timing statistics for all stages."""
        return {name: self.get_stats(name) for name in self._timings}

    def get_total_ms(self):
        """Get sum of mean latencies across all stages."""
        return sum(s["mean_ms"] for s in self.get_all_stats().values())

    def get_fps(self):
        """Estimate FPS from total pipeline latency."""
        total = self.get_total_ms()
        return 1000.0 / total if total > 0 else 0.0

    def report(self):
        """Print a formatted timing report."""
        stats = self.get_all_stats()
        if not stats:
            print("[Profiler] No timing data collected.")
            return

        total = self.get_total_ms()
        fps = self.get_fps()

        print(f"\n{'Stage':<25} {'Mean(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Samples':>8}")
        print("-" * 68)
        for name, s in stats.items():
            print(f"{name:<25} {s['mean_ms']:>10.2f} {s['min_ms']:>10.2f} {s['max_ms']:>10.2f} {s['samples']:>8}")
        print("-" * 68)
        print(f"{'TOTAL':<25} {total:>10.2f}")
        print(f"{'Est. FPS':<25} {fps:>10.1f}")

    def reset(self):
        """Clear all timing data."""
        self._timings.clear()
        self._start_times.clear()
