"""
OCULAR: Real-Time Ocular Tracking and Gaze-Aware Interaction Framework.
"""

from .adaptive import AdaptiveCalibrationEngine, OnlineAdaptiveRefiner
from .blink import BlinkDetector
from .calibration import CalibrationSession
from .camera import Camera
from .evaluation import EvaluationFramework
from .features import FeatureExtractor
from .filters import ExponentialMovingAverage, OneEuroFilter, PointFilter2D, SimpleMovingAverage
from .gaze import GazeRegressor
from .interaction import DwellDetector, GazeCursor, GazeScroller, InteractionController
from .profiler import PerformanceProfiler
from .tracker import FaceTracker

__version__ = "1.0.0"

__all__ = [
    "Camera",
    "FaceTracker",
    "FeatureExtractor",
    "CalibrationSession",
    "AdaptiveCalibrationEngine",
    "OnlineAdaptiveRefiner",
    "GazeRegressor",
    "OneEuroFilter",
    "PointFilter2D",
    "ExponentialMovingAverage",
    "SimpleMovingAverage",
    "BlinkDetector",
    "GazeCursor",
    "DwellDetector",
    "GazeScroller",
    "InteractionController",
    "PerformanceProfiler",
    "EvaluationFramework",
]
