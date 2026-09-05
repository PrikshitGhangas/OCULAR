import collections
import math
import time


class OneEuroFilter:
    """
    1€ (One Euro) Filter for human-computer interaction and noisy signal smoothing.

    Dynamically modulates low-pass cutoff frequency depending on signal velocity:
    - At low speeds: Low cutoff frequency attenuates high-frequency sensor noise and jitter.
    - At high speeds: High cutoff frequency eliminates lag and latency during rapid ballistic movement.

    Reference:
        Casiez, G., Roussel, N., & Vogel, D. (2012).
        1€ filter: a simple speed-based low-pass filter for noisy input in interactive systems.
        CHI 2012.
    """

    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.1, dcutoff=1.0):
        """
        Initialize One Euro Filter parameters.

        Args:
            freq: Expected signal sampling rate in Hz (default: 30.0)
            mincutoff: Minimum cutoff frequency in Hz (lower values reduce jitter when still)
            beta: Speed adjustment coefficient (higher values decrease lag during fast movements)
            dcutoff: Cutoff frequency for derivative filtering in Hz
        """
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)

        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _smoothing_factor(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x, t=None):
        """
        Apply filter to incoming scalar value x at timestamp t.

        Args:
            x: Input value (float)
            t: Timestamp in seconds (float, default: current monotonic time)

        Returns:
            float: Smoothed output value
        """
        if t is None:
            t = time.perf_counter()

        if self.x_prev is None:
            self.x_prev = float(x)
            self.dx_prev = 0.0
            self.t_prev = t
            return float(x)

        dt = t - self.t_prev
        if dt <= 1e-5:
            dt = 1.0 / max(1.0, self.freq)

        # Estimate derivative (speed)
        dx = (x - self.x_prev) / dt
        a_d = self._smoothing_factor(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        # Compute adaptive cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._smoothing_factor(cutoff, dt)

        # Filter value
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat

    def reset(self):
        """Reset internal filter state."""
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


class PointFilter2D:
    """
    Two-dimensional spatial filter employing dual One Euro Filters for X and Y coordinates.
    """

    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.1, dcutoff=1.0):
        self.filter_x = OneEuroFilter(freq, mincutoff, beta, dcutoff)
        self.filter_y = OneEuroFilter(freq, mincutoff, beta, dcutoff)

    def filter(self, x, y, t=None):
        """Filter a 2D coordinate pair (x, y)."""
        sx = self.filter_x.filter(x, t)
        sy = self.filter_y.filter(y, t)
        return sx, sy

    def reset(self):
        self.filter_x.reset()
        self.filter_y.reset()


class ExponentialMovingAverage:
    """
    Exponential Moving Average (EMA) filter.
    """

    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.value = None

    def filter(self, x):
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.value = None


class SimpleMovingAverage:
    """
    Rolling window simple moving average.
    """

    def __init__(self, window_size=5):
        self.window_size = int(window_size)
        self.window = collections.deque(maxlen=window_size)

    def filter(self, x):
        self.window.append(float(x))
        return sum(self.window) / len(self.window)

    def reset(self):
        self.window.clear()
