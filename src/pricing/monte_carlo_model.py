from datetime import date

import numpy as np

from src.models.brownian_motion import BrownianMotion
from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from .regression import BasisType, Regression


class MonteCarloModel:
    def __init__(self, num_simulations: int, market: Market, option: OptionTrade, 
                 pricing_date: date, seed=None):
        """
        Monte Carlo Model for European and American Option Pricing
        
        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo paths (N)
        market : Market
            Market parameters (underlying, vol, rate, div)
        option : OptionTrade
            Option parameters (strike, maturity, call/put)
        pricing_date : date
            Valuation date
        seed : int, optional
            Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.seed = seed
        # No global np.random.seed: each BrownianMotion manages its own
        # Generator object (default_rng) with self.seed for local reproducibility.

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_market_params(self):
        """Extracts (T, S0, sigma, r, q=0) from market and option objects.
        The discrete dividend is handled separately via _get_jdiv."""
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        return T, S0, sigma, r, 0.0

    def _get_jdiv(self, num_steps: int) -> tuple:
        """
        Computes (div, jdiv) for the discrete dividend (cash subtraction model).

        The ex-dividend step is the first step j such that j*dt >= tex (ceiling).

        Returns
        -------
        div  : float, dividend amount (0.0 if absent)
        jdiv : int or None, 1-based index of the ex-dividend step
        """
        if (self.market.ex_div_date is None
                or self.market.ex_div_date >= self.option.mat_date
                or self.market.div_a <= 0):
            return 0.0, None
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        tex = (self.market.ex_div_date - self.pricing_date).days / 365.0
        dt = T / num_steps
        jdiv = int(np.ceil(tex / dt))
        return self.market.div_a, min(max(jdiv, 1), num_steps)

    def _payoff_vec(self, S: np.ndarray) -> np.ndarray:
        """Vectorized payoff: max(S-K, 0) for calls or max(K-S, 0) for puts."""
        if self.option.is_a_call():
            return np.maximum(S - self.option.strike, 0)
        #return np.where(S <= self.option.strike, 1.0, 0.0)  # one-touch binary option with American exercise
        return np.maximum(self.option.strike - S, 0)

    def _num_paths(self, antithetic: bool) -> int:
        """Number of independent paths based on the antithetic variates flag."""
        return self.num_simulations // 2 if antithetic else self.num_simulations

    # ------------------------------------------------------------------
    # European Pricing — Scalar
    # ------------------------------------------------------------------

    def price_european(self, antithetic=True) -> dict:
        """
        Price a European option using scalar (path-by-path) Monte Carlo.

        1. For each path: draw dW, compute S_T via GBM, apply payoff, discount.
        2. Average the discounted payoffs.
        """
        T, S0, sigma, r, q = self._get_market_params()

        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'payoffs': []}

        num_paths = self._num_paths(antithetic)
        df = np.exp(-r * T)
        # BrownianMotion with 1 step = 1 increment dW per path (single jump to T)
        bm = BrownianMotion(1, 1, T, antithetic=antithetic, seed=self.seed)
        discounted_payoffs = []

        for _ in range(num_paths):
            dW, dW_anti = bm.generate_increments_scalar()
            drift = (r - q - 0.5 * sigma ** 2) * T
            S_T = S0 * np.exp(drift + sigma * dW)
            payoff = self.option.pay_off(S_T) * df

            if antithetic:
                S_T_anti = S0 * np.exp(drift + sigma * dW_anti)
                payoff_anti = self.option.pay_off(S_T_anti) * df
                discounted_payoffs.append((payoff + payoff_anti) / 2)
            else:
                discounted_payoffs.append(payoff)

        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        return {'price': price, 'std_error': std_error, 'payoffs': discounted_payoffs}
    
    def price_european_vectorized(self, antithetic=True, num_steps_div: int = 100) -> dict:
        """
        Price a European option using vectorized Monte Carlo.

        If a discrete dividend is present, uses generate_paths (with
        num_steps_div time steps) to capture the ex-dividend drop.
        Otherwise, uses generate_terminal_prices (single jump to T, faster).

        Parameters
        ----------
        num_steps_div : number of time steps used when div > 0 (default 100).
        """
        T, S0, sigma, r, q = self._get_market_params()

        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'payoffs': np.array([])}

        num_paths = self._num_paths(antithetic)
        df = np.exp(-r * T)
        div, jdiv = self._get_jdiv(num_steps_div)

        if jdiv is not None:
            # Discrete dividend: full multi-step path simulation
            bm = BrownianMotion(num_paths, num_steps_div, T,
                                antithetic=antithetic, seed=self.seed)
            S_paths, S_paths_anti = bm.generate_paths(S0, r, sigma, q,
                                                      div=div, jdiv=jdiv)
            S_T      = S_paths[:, -1]
            S_T_anti = S_paths_anti[:, -1] if antithetic else None
        else:
            # No dividend: single jump S0 -> S(T), faster
            bm = BrownianMotion(num_paths, 1, T, antithetic=antithetic, seed=self.seed)
            S_T, S_T_anti = bm.generate_terminal_prices(S0, r, sigma, q)

        payoffs = self._payoff_vec(S_T) * df
        
        final_payoffs = (payoffs + self._payoff_vec(S_T_anti) * df) / 2 if antithetic else payoffs

        price = np.mean(final_payoffs)
        std_error = np.std(final_payoffs, ddof=1) / np.sqrt(len(final_payoffs))
        return {'price': price, 'std_error': std_error, 'payoffs': final_payoffs}

    def price_american_naive(self, num_steps: int = 252, antithetic=True) -> dict:
        """
        Price an American option with naive scalar backward induction.

        For each path:
          - Simulates S(t) step by step via BrownianMotion (scalar mode)
          - Backward sweep: value = max(intrinsic, continuation * df)
        """
        T, S0, sigma, r, q = self._get_market_params()

        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'num_steps': num_steps, 'payoffs': []}

        num_paths = self._num_paths(antithetic)
        dt = T / num_steps
        df = np.exp(-r * dt)
        drift = (r - q - 0.5 * sigma ** 2) * dt

        div, jdiv = self._get_jdiv(num_steps)
        bm = BrownianMotion(1, num_steps, T, antithetic=antithetic, seed=self.seed)
        american_prices = []

        for _ in range(num_paths):
            S, S_anti = S0, S0
            S_path, S_path_anti = [S0], [S0]
            for step in range(num_steps):
                dW, dW_anti = bm.generate_increments_scalar()
                S = S * np.exp(drift + sigma * dW)
                if jdiv is not None and step + 1 == jdiv and div > 0:
                    S = max(S - div, 0.0)
                S_path.append(S)
                if antithetic:
                    S_anti = S_anti * np.exp(drift + sigma * dW_anti)
                    if jdiv is not None and step + 1 == jdiv and div > 0:
                        S_anti = max(S_anti - div, 0.0)
                    S_path_anti.append(S_anti)

            # Backward induction
            value = self.option.pay_off(S_path[-1])
            for step in range(num_steps - 1, -1, -1):
                value = max(self.option.pay_off(S_path[step]), value * df)

            if antithetic:
                value_anti = self.option.pay_off(S_path_anti[-1])
                for step in range(num_steps - 1, -1, -1):
                    value_anti = max(self.option.pay_off(S_path_anti[step]), value_anti * df)
                american_prices.append((value + value_anti) / 2)
            else:
                american_prices.append(value)

        price = np.mean(american_prices)
        std_error = np.std(american_prices, ddof=1) / np.sqrt(len(american_prices))
        return {'price': price, 'std_error': std_error, 'num_steps': num_steps,
                'payoffs': american_prices}
    
    def price_american_naive_vectorized(self, num_steps: int = 252, antithetic=True) -> dict:
        """
        Price an American option with vectorized backward induction.

        Same algorithm as price_american_naive but fully vectorized.
        Uses BrownianMotion.generate_paths() -> matrix of shape (num_paths, num_steps+1).
        """
        T, S0, sigma, r, q = self._get_market_params()

        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'num_steps': num_steps,
                    'payoffs': np.array([])}

        num_paths = self._num_paths(antithetic)
        df = np.exp(-r * T / num_steps)

        div, jdiv = self._get_jdiv(num_steps)
        bm = BrownianMotion(num_paths, num_steps, T, antithetic=antithetic, seed=self.seed)
        S_paths, S_paths_anti = bm.generate_paths(S0, r, sigma, q, div=div, jdiv=jdiv)

        # Backward induction - primary paths
        values = self._payoff_vec(S_paths[:, -1])
        for step in range(num_steps - 1, -1, -1):
            values = np.maximum(self._payoff_vec(S_paths[:, step]), values * df)

        if antithetic:
            values_anti = self._payoff_vec(S_paths_anti[:, -1])
            for step in range(num_steps - 1, -1, -1):
                values_anti = np.maximum(self._payoff_vec(S_paths_anti[:, step]), values_anti * df)
            american_prices = (values + values_anti) / 2
        else:
            american_prices = values

        price = np.mean(american_prices)
        std_error = np.std(american_prices, ddof=1) / np.sqrt(len(american_prices))
        return {'price': price, 'std_error': std_error, 'num_steps': num_steps,
                'payoffs': american_prices}

    def price_american_longstaff_schwartz_vectorized(self, num_steps: int = 252,
                                                     poly_degree: int = 3,
                                                     poly_basis: BasisType = BasisType.POWER,
                                                     residual_threshold: float = 0.0,
                                                     antithetic=True) -> dict:
        """
        Price an American option using Longstaff-Schwartz regression (vectorized).

        Improvement over naive method: polynomial regression E[continuation | S_t]
        on ITM paths to estimate the continuation value at each exercise date.

        Parameters
        ----------
        poly_basis          : polynomial basis type (BasisType.POWER default,
                              LAGUERRE recommended -- see original L&S paper)
        residual_threshold  : residual-based exercise threshold (0 = standard LS,
                              >0 reduces premature early exercise)
        """
        T, S0, sigma, r, q = self._get_market_params()

        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'num_steps': num_steps,
                    'poly_degree': poly_degree, 'payoffs': np.array([])}

        num_paths = self._num_paths(antithetic)
        df = np.exp(-r * T / num_steps)

        div, jdiv = self._get_jdiv(num_steps)
        bm = BrownianMotion(num_paths, num_steps, T, antithetic=antithetic, seed=self.seed)
        S_paths, S_paths_anti = bm.generate_paths(S0, r, sigma, q, div=div, jdiv=jdiv)

        # Initial cash flows at maturity
        cash_flow = self._payoff_vec(S_paths[:, -1])
        if antithetic:
            cash_flow_anti = self._payoff_vec(S_paths_anti[:, -1])

        # Backward induction with LS regression
        reg = Regression(degree=poly_degree, basis=poly_basis,
                         residual_threshold=residual_threshold)
        for step in range(num_steps - 1, -1, -1):
            continuation = cash_flow * df
            intrinsic = self._payoff_vec(S_paths[:, step])
            cash_flow = reg.exercise_decision(S_paths[:, step], intrinsic, continuation)

            if antithetic:
                continuation_anti = cash_flow_anti * df
                intrinsic_anti = self._payoff_vec(S_paths_anti[:, step])
                cash_flow_anti = reg.exercise_decision(
                    S_paths_anti[:, step], intrinsic_anti, continuation_anti)

        ls_prices = (cash_flow + cash_flow_anti) / 2 if antithetic else cash_flow
        price = np.mean(ls_prices)
        std_error = np.std(ls_prices, ddof=1) / np.sqrt(len(ls_prices)) if len(ls_prices) > 1 else 0.0
        return {'price': price, 'std_error': std_error, 'num_steps': num_steps,
                'poly_degree': poly_degree, 'payoffs': ls_prices}
