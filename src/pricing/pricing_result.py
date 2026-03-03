"""
PricingResult -- Unified dataclass for all pricing results.

Each method of MonteCarloModel (and BlackScholes) returns a PricingResult
instead of a plain dict or float, enabling:
  - standardised display (str / repr)
  - confidence interval computation
  - easy comparison with an analytical reference
  - aggregation in ConvergenceStudy
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class PricingResult:
    """
    Unified pricing result container.

    Attributes
    ----------
    price : float
        Estimated option price.
    std_error : float
        MC standard error (0 for analytical methods).
    num_paths : int
        Number of paths used (0 for analytical).
    elapsed_s : float
        Wall-clock computation time in seconds.
    method : str
        Method name (e.g. 'MC-European', 'LS-American', 'Black-Scholes').
    num_steps : int
        Number of time steps (0 for methods without discretisation).
    extra : dict
        Free-form metadata (poly_basis, antithetic, ...).
    """

    price: float
    std_error: float = 0.0
    num_paths: int = 0
    elapsed_s: float = 0.0
    method: str = "unknown"
    num_steps: int = 0
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Confidence intervals
    # ------------------------------------------------------------------

    def confidence_interval(self, alpha: float = 0.05) -> tuple[float, float]:
        """
        Two-sided confidence interval at level (1-alpha).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level (0.05 -> 95% CI).

        Returns
        -------
        (lower, upper) : tuple[float, float]
        """
        if self.std_error <= 0 or self.num_paths <= 1:
            return (self.price, self.price)
        # Normal quantile (Student -> Normal approximation for large n)
        from scipy.stats import norm as _norm
        z = _norm.ppf(1 - alpha / 2)
        margin = z * self.std_error
        return (self.price - margin, self.price + margin)

    # ------------------------------------------------------------------
    # Comparison against reference
    # ------------------------------------------------------------------

    def relative_error(self, reference: float) -> float:
        """
        Relative error against an analytical reference price.

            rel_error = (price - reference) / reference

        Returns
        -------
        float : relative error (positive = overestimate)
        """
        if abs(reference) < 1e-12:
            return float('nan')
        return (self.price - reference) / reference

    def in_confidence_interval(self, reference: float, alpha: float = 0.05) -> bool:
        """Returns True if `reference` lies within the (1-alpha) confidence interval."""
        lo, hi = self.confidence_interval(alpha)
        return lo <= reference <= hi

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        ic = self.confidence_interval()
        parts = [
            f"[{self.method}]",
            f"price={self.price:.4f}",
        ]
        if self.std_error > 0:
            parts.append(f"se={self.std_error:.4f}")
            parts.append(f"IC95=[{ic[0]:.4f}, {ic[1]:.4f}]")
        if self.num_paths > 0:
            parts.append(f"N={self.num_paths:,}")
        if self.elapsed_s > 0:
            parts.append(f"t={self.elapsed_s:.2f}s")
        return "  ".join(parts)

    def __repr__(self) -> str:
        return (f"PricingResult(price={self.price:.4f}, se={self.std_error:.4f}, "
                f"N={self.num_paths}, method='{self.method}')")


# ------------------------------------------------------------------
# Utilitaire : timer contextuel
# ------------------------------------------------------------------

class _Timer:
    """Context manager pour mesurer un temps d'exécution."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t0

    @property
    def seconds(self) -> float:
        return getattr(self, 'elapsed', 0.0)


def timed() -> _Timer:
    """
    Retourne un timer contextuel.

    Usage ::

        with timed() as t:
            price = model.price_european_mc(...)
        print(f"Done in {t.seconds:.2f}s")
    """
    return _Timer()
