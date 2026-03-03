import numpy as np


class BrownianMotion:
    """
    Generates Brownian motions for Monte Carlo simulation.

    Uses np.random.default_rng (Generator object) for reproducibility
    without side effects on NumPy’s global random state.

    Supports discrete dividends via the div/jdiv parameters of generate_paths
    (lecture formula 5-Feb):
        j < jdiv  : standard GBM
        j = jdiv  : GBM + dividend subtraction (ex-div jump)
        j > jdiv  : GBM starting from S(jdiv) after the drop

    Black-Scholes (GBM) formula:
        S(t+dt) = S(t) * exp((r - q - 0.5*σ²)*dt + σ*dW),  dW ~ N(0, dt)
    """

    def __init__(self, num_paths: int, num_steps: int, T: float,
                 antithetic: bool = True, seed=None):
        """
        Parameters
        ----------
        num_paths  : number of Monte Carlo paths
        num_steps  : number of time steps between 0 and T
        T          : maturity in years
        antithetic : if True, generates antithetic paths (-dW)
        seed       : seed for np.random.default_rng (int, SeedSequence, …)
        """
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.T = T
        self.antithetic = antithetic
        self.dt = T / num_steps if num_steps > 0 else T
        # Generator object: local state, no side effects on global np.random
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Generation of Brownian increments
    # ------------------------------------------------------------------

    def generate_increments_scalar(self) -> tuple:
        """
        Generates ONE increment dW ~ N(0, dt) using the Generator.

        Returns
        -------
        dW      : float, main increment
        dW_anti : float (-dW) if antithetic=True, otherwise None
        """
        Z = self._rng.standard_normal()
        dW = Z * np.sqrt(self.dt)
        return dW, -dW if self.antithetic else None

    def generate_increments_vectorized(self) -> np.ndarray:
        """
        Generates the full matrix of increments: shape (num_paths, num_steps).
        """
        return self._rng.standard_normal((self.num_paths, self.num_steps)) * np.sqrt(self.dt)

    # ------------------------------------------------------------------
    # Generation of price paths (GBM + optional discrete dividend)
    # ------------------------------------------------------------------

    def generate_paths(self, S0: float, r: float, sigma: float,
                       q: float = 0.0,
                       div: float = 0.0, jdiv: int = None) -> tuple:
        """
        Full S(t) paths for all time steps.

        Discrete dividend (lecture formula 5-Feb):
          j < jdiv  : S(j*dt) = S((j-1)*dt) * exp(drift*dt + σ*dW)
          j = jdiv  : same, THEN S(jdiv*dt) -= div  (ex-dividend jump)
          j > jdiv  : GBM starting from S(jdiv) after the drop

        Parameters
        ----------
        S0, r, sigma, q : standard Black-Scholes parameters
        div  : absolute discrete dividend amount (0 = no dividend)
        jdiv : ex-div step index (1-based, 1 ≤ jdiv ≤ num_steps, None = absent)

        Returns
        -------
        S_paths      : ndarray (num_paths, num_steps+1)
        S_paths_anti : ndarray (num_paths, num_steps+1) or None
        """
        dW = self.generate_increments_vectorized()
        drift = (r - q - 0.5 * sigma ** 2) * self.dt

        S = np.zeros((self.num_paths, self.num_steps + 1))
        S[:, 0] = S0
        for step in range(self.num_steps):
            S[:, step + 1] = S[:, step] * np.exp(drift + sigma * dW[:, step])
            if jdiv is not None and step + 1 == jdiv and div > 0:
                S[:, step + 1] = np.maximum(S[:, step + 1] - div, 0.0)

        if not self.antithetic:
            return S, None

        S_anti = np.zeros((self.num_paths, self.num_steps + 1))
        S_anti[:, 0] = S0
        for step in range(self.num_steps):
            S_anti[:, step + 1] = S_anti[:, step] * np.exp(drift + sigma * (-dW[:, step]))
            if jdiv is not None and step + 1 == jdiv and div > 0:
                S_anti[:, step + 1] = np.maximum(S_anti[:, step + 1] - div, 0.0)

        return S, S_anti

    def generate_terminal_prices(self, S0: float, r: float, sigma: float,
                                 q: float = 0.0) -> tuple:
        """
        Generates only S(T) in a single step (European pricing without discrete dividend).

        Note: for options with discrete dividends, use generate_paths
              with enough steps to capture the ex-div jump.

        Returns
        -------
        S_T      : ndarray (num_paths,)
        S_T_anti : ndarray (num_paths,) or None
        """
        Z = self._rng.standard_normal(self.num_paths)
        W_T = Z * np.sqrt(self.T)
        drift = (r - q - 0.5 * sigma ** 2) * self.T
        S_T = S0 * np.exp(drift + sigma * W_T)
        if not self.antithetic:
            return S_T, None
        return S_T, S0 * np.exp(drift - sigma * W_T)