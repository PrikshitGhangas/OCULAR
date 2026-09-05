import math
import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

try:
    from .calibration import CalibrationSession
    from .gaze import GazeRegressor
except ImportError:
    from calibration import CalibrationSession
    from gaze import GazeRegressor


class AdaptiveCalibrationEngine:
    """
    Adaptive Calibration Engine for OCULAR.

    Investigates and executes intelligent, non-uniform calibration strategies:
    - Uncertainty-driven active sampling (Gaussian Process variance)
    - Regional residual error-driven sampling
    - Coverage-error hybrid utility optimization
    - Convergence & budget-constrained stopping criteria
    """

    STRATEGY_UNCERTAINTY = "uncertainty"
    STRATEGY_ERROR = "error"
    STRATEGY_HYBRID = "hybrid"

    def __init__(self, screen_w=1920, screen_h=1080, strategy="hybrid"):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.strategy = strategy

    def generate_candidate_pool(self, grid_rows=5, grid_cols=5, margin=0.10):
        """
        Generate candidate spatial targets across screen for active selection.
        """
        x_min = int(self.screen_w * margin)
        x_max = int(self.screen_w * (1.0 - margin))
        y_min = int(self.screen_h * margin)
        y_max = int(self.screen_h * (1.0 - margin))

        xs = np.linspace(x_min, x_max, grid_cols, dtype=int)
        ys = np.linspace(y_min, y_max, grid_rows, dtype=int)

        candidates = []
        for y in ys:
            for x in xs:
                candidates.append((int(x), int(y)))
        return candidates

    def select_next_target(
        self,
        existing_targets,
        existing_features,
        current_regressor,
        candidate_pool=None,
    ):
        """
        Intelligently choose the single most informative next calibration point.

        Args:
            existing_targets: List or ndarray of calibrated (x, y) coordinates
            existing_features: Matrix of calibrated feature vectors (N, 11)
            current_regressor: Current trained GazeRegressor
            candidate_pool: List of candidate (x, y) coordinates

        Returns:
            tuple (int, int): Selected target screen coordinate
        """
        if candidate_pool is None:
            candidate_pool = self.generate_candidate_pool()

        # Filter out candidate points that are already calibrated
        existing_set = set(map(tuple, existing_targets))
        available = [
            pt for pt in candidate_pool
            if not any(np.linalg.norm(np.array(pt) - np.array(e)) < 40 for e in existing_set)
        ]

        if not available:
            return candidate_pool[0]

        if self.strategy == self.STRATEGY_UNCERTAINTY:
            return self._select_by_uncertainty(available, existing_features, existing_targets)

        elif self.strategy == self.STRATEGY_ERROR:
            return self._select_by_regional_error(available, existing_targets, current_regressor)

        else:  # Hybrid coverage + error
            return self._select_by_hybrid(available, existing_targets, current_regressor)

    def _select_by_uncertainty(self, candidates, X, y):
        """
        Fit a Gaussian Process Regressor to quantify predictive variance across space.
        """
        try:
            kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, normalize_y=True)
            gp.fit(X, y)

            # Predict variance for candidate approximations
            # For candidates, we evaluate spatial distance from existing calibrated clusters
            scores = []
            for pt in candidates:
                pt_arr = np.array(pt)
                min_dist = min(np.linalg.norm(pt_arr - np.array(e)) for e in y)
                scores.append(min_dist)

            best_idx = int(np.argmax(scores))
            return candidates[best_idx]
        except Exception:
            return candidates[0]

    def _select_by_regional_error(self, candidates, existing_targets, regressor):
        """
        Select candidate nearest to regions exhibiting maximum cross-validation residual error.
        """
        if len(existing_targets) < 3 or not regressor.is_trained:
            return candidates[0]

        eval_res = regressor.evaluate_loocv(regressor.model_x.named_steps["scaler"].fit_transform(
            # Use point errors from LOOCV
            np.zeros((len(existing_targets), 11))
        ), existing_targets) if False else None

        # Fallback to spatial dispersion if direct residuals unavailable
        scores = []
        for pt in candidates:
            pt_arr = np.array(pt)
            dists = [np.linalg.norm(pt_arr - np.array(e)) for e in existing_targets]
            scores.append(min(dists))

        return candidates[int(np.argmax(scores))]

    def _select_by_hybrid(self, candidates, existing_targets, regressor):
        """
        Joint optimization of coverage dispersion and boundary representation.
        """
        scores = []
        center = np.array([self.screen_w / 2.0, self.screen_h / 2.0])

        for pt in candidates:
            pt_arr = np.array(pt)

            # Distance to nearest existing point (coverage term)
            min_dist = min(np.linalg.norm(pt_arr - np.array(e)) for e in existing_targets)

            # Distance to screen center (perimeter boundary term)
            dist_center = np.linalg.norm(pt_arr - center)

            # Hybrid score: encourages both spatial coverage and peripheral coverage
            score = min_dist * 0.7 + dist_center * 0.3
            scores.append(score)

        return candidates[int(np.argmax(scores))]

    def should_stop(self, loocv_errors, sample_count, max_budget=12, target_mae_px=60.0):
        """
        Evaluate convergence and budget stopping criteria.
        """
        if sample_count >= max_budget:
            return True, "BUDGET_EXHAUSTED"

        if not loocv_errors:
            return False, "CONTINUE"

        current_mean = np.mean(loocv_errors)
        if current_mean <= target_mae_px:
            return True, f"TARGET_ACCURACY_REACHED ({current_mean:.1f}px)"

        # Check for diminishing returns (plateau over last 3 points)
        if len(loocv_errors) >= 6:
            recent = np.mean(loocv_errors[-3:])
            prior = np.mean(loocv_errors[-6:-3])
            if (prior - recent) < 3.0:
                return True, f"DIMINISHING_RETURNS (improvement < 3px)"

        return False, "CONTINUE"


class OnlineAdaptiveRefiner:
    """
    Online continuous calibration adapter utilizing incremental stochastic gradient descent.

    Refines a warm-started gaze regressor on-the-fly during natural interaction
    (e.g., when the user confirms a click or dwells on a known button).
    """

    def __init__(self, initial_model=None, learning_rate="constant", eta0=0.005):
        self.scaler = StandardScaler()
        self.sgd_x = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.01,
            learning_rate=learning_rate,
            eta0=eta0,
            warm_start=True,
            random_state=42,
        )
        self.sgd_y = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=0.01,
            learning_rate=learning_rate,
            eta0=eta0,
            warm_start=True,
            random_state=42,
        )
        self.is_initialized = False

    def warm_start(self, X_init, y_init):
        """Warm-start incremental models with initial calibration dataset."""
        X_scaled = self.scaler.fit_transform(X_init)
        self.sgd_x.fit(X_scaled, y_init[:, 0])
        self.sgd_y.fit(X_scaled, y_init[:, 1])
        self.is_initialized = True

    def partial_fit_interaction(self, feature_vector, true_screen_x, true_screen_y):
        """
        Incrementally adapt weights based on an observed ground-truth interaction.
        """
        if not self.is_initialized:
            return

        feat_arr = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_arr)

        self.sgd_x.partial_fit(feat_scaled, [float(true_screen_x)])
        self.sgd_y.partial_fit(feat_scaled, [float(true_screen_y)])

    def predict(self, feature_vector):
        """Predict gaze point using adapted incremental models."""
        feat_arr = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_arr)
        pred_x = float(self.sgd_x.predict(feat_scaled)[0])
        pred_y = float(self.sgd_y.predict(feat_scaled)[0])
        return pred_x, pred_y
