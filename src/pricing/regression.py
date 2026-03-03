from enum import StrEnum

import numpy as np


class BasisType(StrEnum):
    """
    Polynomial bases available for Longstaff-Schwartz regression.

    All generate the same polynomial space (mathematically equivalent),
    but orthogonal bases provide better numerical stability
    than the standard monomial basis (POWER).
    """
    POWER     = 'power'     # Standard monomials: 1, x, x², x³, …
    LAGUERRE  = 'laguerre'  # exp(-x/2) * L_k(x)  (L&S paper basis)
    HERMITE   = 'hermite'   # Probabilistic Hermite polynomials (Hermite)
    LEGENDRE  = 'legendre'  # Legendre polynomials
    CHEBYSHEV = 'chebyshev' # Chebyshev polynomials of the first kind


class Regression:
    """
    Polynomial regression for Longstaff-Schwartz.

    Improvements vs naive regression (np.polyfit):
    - Choice of polynomial basis (BasisType) via explicit design matrix
    - Automatic input normalization (essential for LAGUERRE/HERMITE)
    - Solution via np.linalg.lstsq (robust to singular cases)
    - Residual standard deviation computed after fit
    - Exercise threshold: exercise only if IV > reg + threshold * residual_std

    Normalization:
    - LAGUERRE  : X_norm = X / mean(X)         → values around 1 (domain ≥ 0)
    - Others    : X_norm = (X - mean) / std     → z-score, values around 0

    Since all bases are polynomials, the least-squares solution is
    unique and identical regardless of the basis (cf. lecture 1/7/2026 slide 4).
    The basis only affects the numerical conditioning of the system.
    """

    def __init__(self, degree: int = 1,
                 basis: BasisType = BasisType.POWER,
                 residual_threshold: float = 0.0,
                 normalize: bool = True):
        """
        Parameters
        ----------
        degree             : polynomial degree (2 = quadratic as in L&S,
                             3 = cubic by default)
        basis              : polynomial basis (see BasisType)
        residual_threshold : fraction of residual standard deviation added to the threshold
                             0.0 → standard LS behavior
                             0.1 → exercise if IV > reg + 0.1 * residual_std
        normalize          : if True (default), normalizes inputs before
                             building the design matrix
        """
        self.degree = degree
        self.basis = BasisType(basis)
        self.residual_threshold = residual_threshold
        self.normalize = normalize
        self._coeffs: np.ndarray = None
        self._residual_std: float = 0.0
        # Normalization parameters, learned in fit()
        self._x_loc: float = 0.0
        self._x_scale: float = 1.0

    # ------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------

    def _fit_normalization(self, X: np.ndarray) -> None:
        """Computes and stores normalization parameters on training data."""
        if not self.normalize:
            self._x_loc, self._x_scale = 0.0, 1.0
            return
        mean = float(np.mean(X))
        if self.basis == BasisType.LAGUERRE:
            # Domain must remain positive: X_norm = X / mean → mean = 1
            self._x_loc = 0.0
            self._x_scale = mean if mean > 0 else 1.0
        else:
            # Standard z-score
            self._x_loc = mean
            std = float(np.std(X))
            self._x_scale = std if std > 0 else 1.0

    def _normalize_x(self, X: np.ndarray) -> np.ndarray:
        """Applies the normalization learned during fit."""
        return (X - self._x_loc) / self._x_scale

    # ------------------------------------------------------------------
    # Design matrix (polynomial basis)
    # ------------------------------------------------------------------

    def _design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Builds the Phi matrix of shape (n, degree+1) in the chosen basis.
        Inputs are normalized before polynomial evaluation.

        Phi[i, k] = k-th basis function evaluated at X_norm[i].
        """
        X_n = self._normalize_x(X)
        d   = self.degree + 1
        eye = np.eye(d)

        if self.basis == BasisType.POWER:
            return np.column_stack([X_n ** k for k in range(d)])

        elif self.basis == BasisType.LAGUERRE:
            # L&S paper: φ_k(x) = exp(-x/2) * L_k(x)
            # With X_n ≈ 1 for ATM, exp(-0.5) ≈ 0.6 → no blow-up
            w = np.exp(-X_n / 2)
            return np.column_stack([
                w * np.polynomial.laguerre.lagval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.HERMITE:
            return np.column_stack([
                np.polynomial.hermite_e.hermeval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.LEGENDRE:
            return np.column_stack([
                np.polynomial.legendre.legval(X_n, eye[k]) for k in range(d)
            ])

        elif self.basis == BasisType.CHEBYSHEV:
            return np.column_stack([
                np.polynomial.chebyshev.chebval(X_n, eye[k]) for k in range(d)
            ])

        raise ValueError(f"Unknown polynomial basis: {self.basis}")

    # ------------------------------------------------------------------
    # Fit / Predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regression":
        """
        Least-squares regression in the chosen basis.
        Learns normalization on X before solving the system.
        """
        self._fit_normalization(X)
        Phi = self._design_matrix(X)
        self._coeffs, _, _, _ = np.linalg.lstsq(Phi, y, rcond=None)
        self._residual_std = float(np.std(y - Phi @ self._coeffs))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts E[continuation | S_t] in the chosen basis, clamped to 0.
        Uses normalization learned during the last fit().
        """
        if self._coeffs is None:
            raise ValueError("Call fit() before predict().")
        return np.maximum(self._design_matrix(X) @ self._coeffs, 0.0)

    # ------------------------------------------------------------------
    # Exercise decision (Longstaff-Schwartz)
    # ------------------------------------------------------------------

    def exercise_decision(self,
                          S_at_step: np.ndarray,
                          intrinsic: np.ndarray,
                          continuation_discounted: np.ndarray) -> np.ndarray:
        """
        Optimal exercise decision at one time step.

        Exercise condition with threshold (cf. lecture 1/7/2026 forward price example):
            Exercise if IV(S) > E[continuation | S] + residual_threshold * residual_std

        Parameters
        ----------
        S_at_step               : underlying price at this step, shape (num_paths,)
        intrinsic               : intrinsic value, shape (num_paths,)
        continuation_discounted : future cash flow discounted by one step, shape (num_paths,)
        """
        itm_mask = intrinsic > 0
        n_itm = int(np.sum(itm_mask))

        if n_itm > self.degree + 1:
            self.fit(S_at_step[itm_mask], continuation_discounted[itm_mask])
            estimated = self.predict(S_at_step)
            margin = self._residual_std * self.residual_threshold
            return np.where(intrinsic > estimated + margin,
                            intrinsic,
                            continuation_discounted)
        else:
            return np.where(intrinsic > continuation_discounted,
                            intrinsic,
                            continuation_discounted)