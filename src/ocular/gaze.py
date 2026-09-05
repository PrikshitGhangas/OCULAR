import math
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


class GazeRegressor:
    """
    Gaze estimation regressor mapping feature vectors to 2D continuous screen coordinates.

    Supports Ridge Regression (with Polynomial expansion), Support Vector Regression (SVR),
    Random Forest, and Multi-Layer Perceptron (MLP).
    """

    MODEL_RIDGE = "ridge"
    MODEL_SVR = "svr"
    MODEL_RF = "rf"
    MODEL_MLP = "mlp"

    def __init__(self, model_type="ridge", screen_w=1920, screen_h=1080):
        self.model_type = model_type
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.model_x = self._build_pipeline(model_type, coord="x")
        self.model_y = self._build_pipeline(model_type, coord="y")
        self.is_trained = False

    def _build_pipeline(self, model_type, coord="x"):
        """Construct scikit-learn estimation pipeline."""
        if model_type == self.MODEL_RIDGE:
            # Baseline: Polynomial Feature expansion with L2 Ridge regularizer
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                    ("ridge", Ridge(alpha=10.0 if coord == "y" else 5.0)),
                ]
            )

        elif model_type == self.MODEL_SVR:
            # Non-linear Radial Basis Function SVR
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", SVR(kernel="rbf", C=50.0, gamma="scale", epsilon=5.0)),
                ]
            )

        elif model_type == self.MODEL_RF:
            # Random Forest ensemble with shallow depth for small calibration sets
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=100, max_depth=4, min_samples_leaf=1, random_state=42
                        ),
                    ),
                ]
            )

        elif model_type == self.MODEL_MLP:
            # Small multi-layer perceptron
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(32, 16),
                            activation="relu",
                            solver="adam",
                            alpha=0.01,
                            max_iter=1500,
                            random_state=42,
                        ),
                    ),
                ]
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        """
        Train coordinate regressors on calibration samples.

        Args:
            X: Feature matrix of shape (N, 11)
            y: Target coordinates of shape (N, 2) [X_screen, Y_screen]
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(X) < 3:
            raise ValueError("At least 3 calibration points required to fit regressor.")

        y_x = y[:, 0]
        y_y = y[:, 1]

        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)
        self.is_trained = True

    def predict(self, X):
        """
        Predict continuous 2D screen coordinates from input features.

        Args:
            X: Feature vector of shape (11,) or batch of shape (N, 11)

        Returns:
            numpy.ndarray: (2,) screen (x, y) or (N, 2)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot predict: GazeRegressor has not been trained.")

        X = np.asarray(X, dtype=np.float64)
        single_sample = X.ndim == 1
        if single_sample:
            X = X.reshape(1, -1)

        pred_x = self.model_x.predict(X)
        pred_y = self.model_y.predict(X)

        # Clip predictions to screen boundaries with generous buffer
        pred_x = np.clip(pred_x, 0.0, float(self.screen_w))
        pred_y = np.clip(pred_y, 0.0, float(self.screen_h))

        result = np.column_stack((pred_x, pred_y))
        return result[0] if single_sample else result

    def evaluate_loocv(self, X, y):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) on calibration data.

        Returns:
            dict: Comprehensive error metrics (mean, median, p95, visual angle)
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        loo = LeaveOneOut()
        errors_px = []

        for train_idx, test_idx in loo.split(X):
            train_x, test_x = X[train_idx], X[test_idx]
            train_y, test_y = y[train_idx], y[test_idx]

            fold_regressor = GazeRegressor(
                model_type=self.model_type,
                screen_w=self.screen_w,
                screen_h=self.screen_h,
            )
            fold_regressor.fit(train_x, train_y)
            pred = fold_regressor.predict(test_x)

            err = np.linalg.norm(pred - test_y)
            errors_px.append(float(err))

        errors_arr = np.array(errors_px)
        mean_err = float(np.mean(errors_arr))
        median_err = float(np.median(errors_arr))
        p95_err = float(np.percentile(errors_arr, 95))

        # Visual angle estimation assuming standard 24" monitor at 60cm distance
        deg_mean = self.pixels_to_degrees(mean_err)
        deg_median = self.pixels_to_degrees(median_err)

        return {
            "model_type": self.model_type,
            "samples": len(X),
            "mean_error_px": mean_err,
            "median_error_px": median_err,
            "p95_error_px": p95_err,
            "mean_error_deg": deg_mean,
            "median_error_deg": deg_median,
            "point_errors_px": errors_px,
        }

    @staticmethod
    def pixels_to_degrees(error_px, screen_w_px=1920, screen_w_cm=53.0, distance_cm=60.0):
        """
        Convert pixel error distance to degrees of visual angle.
        """
        cm_per_px = screen_w_cm / screen_w_px
        error_cm = error_px * cm_per_px
        rad = math.atan(error_cm / distance_cm)
        return float(math.degrees(rad))

    def save(self, directory="models"):
        """Save trained models to directory using joblib."""
        os.makedirs(directory, exist_ok=True)
        joblib.dump(self.model_x, os.path.join(directory, f"gaze_model_x_{self.model_type}.joblib"))
        joblib.dump(self.model_y, os.path.join(directory, f"gaze_model_y_{self.model_type}.joblib"))

        config = {
            "model_type": self.model_type,
            "screen_w": self.screen_w,
            "screen_h": self.screen_h,
        }
        joblib.dump(config, os.path.join(directory, f"gaze_config_{self.model_type}.joblib"))

    def load(self, directory="models"):
        """Load trained models from directory."""
        config = joblib.load(os.path.join(directory, f"gaze_config_{self.model_type}.joblib"))
        self.screen_w = config["screen_w"]
        self.screen_h = config["screen_h"]

        self.model_x = joblib.load(
            os.path.join(directory, f"gaze_model_x_{self.model_type}.joblib")
        )
        self.model_y = joblib.load(
            os.path.join(directory, f"gaze_model_y_{self.model_type}.joblib")
        )
        self.is_trained = True
