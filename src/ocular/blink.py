class BlinkDetector:
    """
    Ocular state analyzer distinguishing natural blinks, deliberate command blinks,
    and unilateral winks via Eye Aspect Ratio (EAR) temporal tracking.
    """

    STATE_OPEN = "OPEN"
    STATE_BLINKING = "BLINKING"

    EVENT_NONE = "NONE"
    EVENT_NATURAL_BLINK = "NATURAL_BLINK"
    EVENT_DELIBERATE_BLINK = "DELIBERATE_BLINK"
    EVENT_LEFT_WINK = "LEFT_WINK"
    EVENT_RIGHT_WINK = "RIGHT_WINK"

    def __init__(
        self,
        ear_threshold=0.20,
        natural_min_frames=2,
        deliberate_min_frames=8,
        wink_discrepancy=0.08,
    ):
        """
        Initialize BlinkDetector.

        Args:
            ear_threshold: EAR boundary below which an eye is classified closed (default: 0.20)
            natural_min_frames: Minimum closed frames required to qualify as natural blink
            deliberate_min_frames: Minimum closed frames required to qualify as deliberate gesture
            wink_discrepancy: Difference between left and right EAR to register a wink
        """
        self.ear_threshold = ear_threshold
        self.natural_min_frames = natural_min_frames
        self.deliberate_min_frames = deliberate_min_frames
        self.wink_discrepancy = wink_discrepancy

        self.both_closed_count = 0
        self.left_closed_count = 0
        self.right_closed_count = 0

    def update(self, left_ear, right_ear):
        """
        Process current frame EAR values and return detected event.

        Args:
            left_ear: Left eye EAR scalar (float)
            right_ear: Right eye EAR scalar (float)

        Returns:
            str: One of EVENT_NONE, EVENT_NATURAL_BLINK, EVENT_DELIBERATE_BLINK,
                 EVENT_LEFT_WINK, EVENT_RIGHT_WINK
        """
        if left_ear is None or right_ear is None:
            return self.EVENT_NONE

        avg_ear = (left_ear + right_ear) / 2.0
        left_closed = left_ear < self.ear_threshold
        right_closed = right_ear < self.ear_threshold

        event = self.EVENT_NONE

        # Both eyes closed
        if left_closed and right_closed:
            self.both_closed_count += 1
            self.left_closed_count = 0
            self.right_closed_count = 0
            return self.EVENT_NONE

        # Left closed, right open (potential left wink)
        elif left_closed and not right_closed and (right_ear - left_ear) >= self.wink_discrepancy:
            self.left_closed_count += 1
            self.both_closed_count = 0
            self.right_closed_count = 0
            return self.EVENT_NONE

        # Right closed, left open (potential right wink)
        elif right_closed and not left_closed and (left_ear - right_ear) >= self.wink_discrepancy:
            self.right_closed_count += 1
            self.both_closed_count = 0
            self.left_closed_count = 0
            return self.EVENT_NONE

        # Eyes have now reopened: evaluate duration
        if self.both_closed_count > 0:
            if self.both_closed_count >= self.deliberate_min_frames:
                event = self.EVENT_DELIBERATE_BLINK
            elif self.both_closed_count >= self.natural_min_frames:
                event = self.EVENT_NATURAL_BLINK
            self.both_closed_count = 0

        elif self.left_closed_count > 0:
            if self.left_closed_count >= self.natural_min_frames:
                event = self.EVENT_LEFT_WINK
            self.left_closed_count = 0

        elif self.right_closed_count > 0:
            if self.right_closed_count >= self.natural_min_frames:
                event = self.EVENT_RIGHT_WINK
            self.right_closed_count = 0

        return event

    def is_blinking(self, left_ear, right_ear):
        """Check if user is currently in the middle of a closed-eye phase."""
        if left_ear is None or right_ear is None:
            return False
        return (left_ear < self.ear_threshold) or (right_ear < self.ear_threshold)
