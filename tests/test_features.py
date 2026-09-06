import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "ocular")))

from features import FeatureExtractor


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarksList:
    def __init__(self, count=478):
        self.landmark = [MockLandmark(0.5, 0.5, 0.0) for _ in range(count)]


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_feature_names_length(self):
        self.assertEqual(len(self.extractor.FEATURE_NAMES), 11)

    def test_extract_mock_landmarks(self):
        mock_lms = MockLandmarksList(478)
        # Position left eye landmarks
        mock_lms.landmark[33] = MockLandmark(0.3, 0.4)    # Outer
        mock_lms.landmark[133] = MockLandmark(0.4, 0.4)   # Inner
        mock_lms.landmark[159] = MockLandmark(0.35, 0.38) # Top
        mock_lms.landmark[145] = MockLandmark(0.35, 0.42) # Bottom
        mock_lms.landmark[468] = MockLandmark(0.35, 0.40) # Iris center

        # Position right eye landmarks
        mock_lms.landmark[263] = MockLandmark(0.7, 0.4)
        mock_lms.landmark[362] = MockLandmark(0.6, 0.4)
        mock_lms.landmark[386] = MockLandmark(0.65, 0.38)
        mock_lms.landmark[374] = MockLandmark(0.65, 0.42)
        mock_lms.landmark[473] = MockLandmark(0.65, 0.40)

        # Pose landmarks
        mock_lms.landmark[1] = MockLandmark(0.5, 0.5)     # Nose
        mock_lms.landmark[152] = MockLandmark(0.5, 0.7)   # Chin
        mock_lms.landmark[61] = MockLandmark(0.45, 0.65)  # Mouth L
        mock_lms.landmark[291] = MockLandmark(0.55, 0.65) # Mouth R

        features = self.extractor.extract(mock_lms, (720, 1280, 3))
        self.assertIsNotNone(features)
        self.assertEqual(len(features), 11)

        # Iris ratios should be approximately centered (around 0.5)
        self.assertAlmostEqual(features[0], 0.5, places=1)
        self.assertAlmostEqual(features[1], 0.5, places=1)


if __name__ == "__main__":
    unittest.main()
