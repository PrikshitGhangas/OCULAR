import math
import time
import numpy as np

try:
    from .filters import PointFilter2D
except ImportError:
    from filters import PointFilter2D


class GazeCursor:
    """
    Translates filtered gaze coordinates into smooth OS cursor positions.
    """

    def __init__(
        self,
        screen_w=1920,
        screen_h=1080,
        freq=30.0,
        mincutoff=1.0,
        beta=0.1,
        enable_os_cursor=True,
    ):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.enable_os_cursor = enable_os_cursor

        # Smoothing filter
        self.filter = PointFilter2D(freq=freq, mincutoff=mincutoff, beta=beta)
        self.current_pos = (screen_w / 2.0, screen_h / 2.0)

        self._pyautogui = None
        if self.enable_os_cursor:
            try:
                import pyautogui

                pyautogui.PAUSE = 0
                pyautogui.FAILSAFE = True
                self._pyautogui = pyautogui
            except Exception:
                self._pyautogui = None

    def update(self, raw_x, raw_y, timestamp=None):
        """
        Filter incoming raw gaze coordinates and optionally move system cursor.

        Returns:
            tuple (float, float): Smoothed (x, y) coordinates
        """
        # Constrain to screen boundaries
        cx = float(np.clip(raw_x, 0.0, self.screen_w))
        cy = float(np.clip(raw_y, 0.0, self.screen_h))

        sx, sy = self.filter.filter(cx, cy, timestamp)
        self.current_pos = (sx, sy)

        if self._pyautogui is not None:
            try:
                self._pyautogui.moveTo(int(sx), int(sy))
            except Exception:
                pass

        return sx, sy

    def apply_magnetism(self, target_points, snap_radius=60):
        """
        Snap current cursor position to nearest UI target if within attraction radius.
        """
        cx, cy = self.current_pos
        for tx, ty in target_points:
            dist = math.hypot(cx - tx, cy - ty)
            if dist <= snap_radius:
                return float(tx), float(ty)
        return cx, cy


class DwellDetector:
    """
    Fixation and dwell-selection controller with spatial stability gating,
    progress tracking, and post-activation refractory cooldown.
    """

    def __init__(self, threshold_ms=600, radius_px=50, cooldown_ms=1000):
        self.threshold_ms = threshold_ms
        self.radius_px = radius_px
        self.cooldown_ms = cooldown_ms

        self.dwell_start_time = None
        self.dwell_center = None
        self.last_activation_time = 0

    def update(self, current_x, current_y, timestamp_ms=None):
        """
        Process current gaze position and evaluate dwell state.

        Returns:
            tuple (bool, float): (triggered_flag, progress_ratio [0.0 - 1.0])
        """
        if timestamp_ms is None:
            timestamp_ms = time.perf_counter() * 1000.0

        # Enforce refractory cooldown
        if (timestamp_ms - self.last_activation_time) < self.cooldown_ms:
            return False, 0.0

        # Initialize dwell fixation center
        if self.dwell_center is None:
            self.dwell_center = (current_x, current_y)
            self.dwell_start_time = timestamp_ms
            return False, 0.0

        # Check distance from fixation anchor
        dist = math.hypot(current_x - self.dwell_center[0], current_y - self.dwell_center[1])

        if dist > self.radius_px:
            # Gaze broke fixation boundary: reset anchor
            self.dwell_center = (current_x, current_y)
            self.dwell_start_time = timestamp_ms
            return False, 0.0

        elapsed = timestamp_ms - self.dwell_start_time
        progress = float(np.clip(elapsed / self.threshold_ms, 0.0, 1.0))

        if elapsed >= self.threshold_ms:
            # Dwell activation threshold satisfied
            self.last_activation_time = timestamp_ms
            self.dwell_center = None
            self.dwell_start_time = None
            return True, 1.0

        return False, progress

    def reset(self):
        """Clear active dwell tracking state."""
        self.dwell_start_time = None
        self.dwell_center = None


class GazeScroller:
    """
    Hands-free document/window scroller triggered by peripheral vertical gaze margins.
    """

    def __init__(self, screen_h=1080, margin_ratio=0.18, max_speed=15):
        self.screen_h = screen_h
        self.margin_ratio = margin_ratio
        self.max_speed = max_speed
        self.top_boundary = screen_h * margin_ratio
        self.bottom_boundary = screen_h * (1.0 - margin_ratio)

        self._pyautogui = None
        try:
            import pyautogui

            self._pyautogui = pyautogui
        except Exception:
            pass

    def update(self, gaze_y):
        """
        Evaluate vertical gaze position and issue scroll event.

        Returns:
            int: Scroll magnitude (positive for up, negative for down, 0 for neutral)
        """
        scroll_amount = 0

        if gaze_y < self.top_boundary:
            # Gaze directed into top margin: scroll up
            depth = 1.0 - (gaze_y / self.top_boundary)
            scroll_amount = int(self.max_speed * depth)

        elif gaze_y > self.bottom_boundary:
            # Gaze directed into bottom margin: scroll down
            depth = (gaze_y - self.bottom_boundary) / (self.screen_h - self.bottom_boundary)
            scroll_amount = -int(self.max_speed * depth)

        if scroll_amount != 0 and self._pyautogui is not None:
            try:
                self._pyautogui.scroll(scroll_amount)
            except Exception:
                pass

        return scroll_amount


class InteractionController:
    """
    Comprehensive multimodal interaction manager uniting cursor motion,
    dwell selection, blink gesture dispatch, and Midas Touch safeguards.
    """

    def __init__(self, screen_w=1920, screen_h=1080, enable_os_cursor=False):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cursor = GazeCursor(screen_w, screen_h, enable_os_cursor=enable_os_cursor)
        self.dwell = DwellDetector(threshold_ms=600, radius_px=50, cooldown_ms=1000)
        self.scroller = GazeScroller(screen_h)

        self.last_dwell_event = False
        self.last_dwell_progress = 0.0
        self.last_scroll_speed = 0

    def process_frame(self, raw_pred_x, raw_pred_y, blink_event="NONE"):
        """
        Process single prediction frame into interaction state.

        Returns:
            dict: Comprehensive interaction state frame
        """
        t_ms = time.perf_counter() * 1000.0

        # 1. Smooth cursor coordinate
        sx, sy = self.cursor.update(raw_pred_x, raw_pred_y)

        # 2. Evaluate Dwell
        dwell_triggered, progress = self.dwell.update(sx, sy, t_ms)
        self.last_dwell_event = dwell_triggered
        self.last_dwell_progress = progress

        # 3. Evaluate Scrolling
        scroll_speed = self.scroller.update(sy)
        self.last_scroll_speed = scroll_speed

        # 4. Trigger mouse click if dwell fired
        if dwell_triggered and self.cursor._pyautogui is not None:
            try:
                self.cursor._pyautogui.click()
            except Exception:
                pass

        # 5. Intentional Blink action (e.g. deliberate long blink triggers click)
        if blink_event == "DELIBERATE_BLINK" and self.cursor._pyautogui is not None:
            try:
                self.cursor._pyautogui.click()
            except Exception:
                pass

        return {
            "cursor_x": sx,
            "cursor_y": sy,
            "dwell_triggered": dwell_triggered,
            "dwell_progress": progress,
            "scroll_speed": scroll_speed,
            "blink_event": blink_event,
        }
