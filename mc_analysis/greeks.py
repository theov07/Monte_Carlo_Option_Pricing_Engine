"""
Monte Carlo Greeks via finite differences (bump & reprice).

Method: central finite differences with Common Random Numbers (CRN).
The same seed is used for both pricings in a bump, which substantially
reduces the variance of each Greek estimator.

Greeks computed:
  Delta  -- dP/dS0          bump of h_S  on the underlying
  Gamma  -- d2P/dS0^2       bump of h_S  (second-order central scheme)
  Vega   -- dP/dsigma        bump of h_v  on volatility
  Theta  -- dP/dt  (-dP/dT) 1 calendar day decay
  Rho    -- dP/dr            bump of h_r  on the rate

SE of Greeks = error propagation from elementary pricers:
  SE(Delta) = sqrt(SE_up^2 + SE_down^2) / (2h)   (central difference)
  SE(Gamma) = sqrt(SE_up^2 + 4*SE_0^2 + SE_down^2) / h^2
"""

import math
import numpy as np
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import Optional

from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src.regression import BasisType


# ─────────────────────────────────────────────────────────────────────────────
#  Structures de résultat
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GreekResult:
    """Value and standard error of a single Greek."""
    name: str
    value: float
    se: float

    def ci95(self) -> tuple:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)

    def __str__(self) -> str:
        return f"{self.name:<8}  {self.value:>10.5f}  ±{1.96*self.se:.5f}"


@dataclass
class AllGreeks:
    delta: GreekResult
    gamma: GreekResult
    vega:  GreekResult
    theta: GreekResult
    rho:   GreekResult
    price: float
    price_se: float


# ─────────────────────────────────────────────────────────────────────────────
#  Classe principale
# ─────────────────────────────────────────────────────────────────────────────

class MCGreeks:
    """
    Monte Carlo Greeks via finite differences + Common Random Numbers (CRN).

    Parameters
    ----------
    market : Market
    option : OptionTrade
    pricing_date : date
    num_paths : int
        Number of simulations per pricing call.
    antithetic : bool
        Use antithetic variates for variance reduction.
    seed : int or None
        Base seed; each internal pricing call uses this same seed
        to ensure CRN correlation across bumps.
    num_steps : int
        Number of time steps for American pricing (ignored for European).
    h_S : float
        Relative bump on S0 (default 0.5%).
    h_v : float
        Absolute bump on sigma (default 1%).
    h_r : float
        Absolute bump on r (default 1 bp = 0.0001).
    theta_days : float
        Temporal shift for Theta in calendar days (default 1).
    """

    def __init__(
        self,
        market: Market,
        option: OptionTrade,
        pricing_date: date,
        num_paths: int = 100_000,
        antithetic: bool = True,
        seed: Optional[int] = None,
        num_steps: int = 252,
        h_S: float = 0.005,
        h_v: float = 0.01,
        h_r: float = 0.0001,
        theta_days: float = 1.0,
    ):
        self.market       = market
        self.option       = option
        self.pricing_date = pricing_date
        self.num_paths    = num_paths
        self.antithetic   = antithetic
        self.seed         = seed
        self.num_steps    = num_steps
        self.h_S          = h_S
        self.h_v          = h_v
        self.h_r          = h_r
        self.theta_days   = theta_days

    # -- private helpers --------------------------------------------------

    def _new_market(self, *, S=None, vol=None, rate=None) -> Market:
        """Clones the market object with a single parameter changed."""
        return Market(
            underlying  = S    if S    is not None else self.market.underlying,
            vol         = vol  if vol  is not None else self.market.vol,
            rate        = rate if rate is not None else self.market.rate,
            div_a       = self.market.div_a,
            ex_div_date = self.market.ex_div_date,
        )

    def _price(self, market: Market, pricing_date: date) -> dict:
        """Runs a MC pricing with the given parameters (CRN via fixed seed)."""
        mc = MonteCarloModel(
            self.num_paths, market, self.option, pricing_date, seed=self.seed
        )
        if self.option.is_american():
            return mc.price_american_longstaff_schwartz_vectorized(
                num_steps   = self.num_steps,
                poly_basis  = BasisType.POWER,
                antithetic  = self.antithetic,
            )
        return mc.price_european_vectorized(antithetic=self.antithetic)

    # ── Greeks individuels ───────────────────────────────────────────────────

    def delta(self) -> GreekResult:
        """
        Delta = (P(S+hS) - P(S-hS)) / (2*hS)
        Central difference, absolute bump dS = h_S * S0.
        """
        S0 = self.market.underlying
        dS = self.h_S * S0

        r_up   = self._price(self._new_market(S=S0 + dS), self.pricing_date)
        r_down = self._price(self._new_market(S=S0 - dS), self.pricing_date)

        value = (r_up['price'] - r_down['price']) / (2 * dS)
        se    = math.sqrt(r_up['std_error']**2 + r_down['std_error']**2) / (2 * dS)
        return GreekResult('Delta', value, se)

    def gamma(self) -> GreekResult:
        """
        Gamma = (P(S+hS) - 2*P(S) + P(S-hS)) / hS^2
        """
        S0 = self.market.underlying
        dS = self.h_S * S0

        r_up   = self._price(self._new_market(S=S0 + dS), self.pricing_date)
        r_0    = self._price(self.market, self.pricing_date)
        r_down = self._price(self._new_market(S=S0 - dS), self.pricing_date)

        value = (r_up['price'] - 2 * r_0['price'] + r_down['price']) / dS**2
        se    = math.sqrt(
            r_up['std_error']**2 + 4 * r_0['std_error']**2 + r_down['std_error']**2
        ) / dS**2
        return GreekResult('Gamma', value, se)

    def vega(self) -> GreekResult:
        """
        Vega = (P(sigma+hv) - P(sigma-hv)) / (2*hv)
        Expressed per 1% vol move (x 0.01).
        """
        sig = self.market.vol
        dv  = self.h_v

        r_up   = self._price(self._new_market(vol=sig + dv), self.pricing_date)
        r_down = self._price(self._new_market(vol=sig - dv), self.pricing_date)

        # Vega pour +1 % de vol (convention usuelle)
        value = (r_up['price'] - r_down['price']) / (2 * dv) * 0.01
        se    = math.sqrt(r_up['std_error']**2 + r_down['std_error']**2) / (2 * dv) * 0.01
        return GreekResult('Vega', value, se)

    def theta(self) -> GreekResult:
        """
        Theta = -(P(t+dt) - P(t)) / dt  (convention: daily time decay)
        dt = theta_days / 365.
        """
        dt_days = timedelta(days=self.theta_days)
        date_shifted = self.pricing_date + dt_days

        # Guard: do not shift past maturity
        if date_shifted >= self.option.mat_date:
            return GreekResult('Theta', float('nan'), float('nan'))

        r_t    = self._price(self.market, self.pricing_date)
        r_t1   = self._price(self.market, date_shifted)

        dt_yr  = self.theta_days / 365.0
        value  = -(r_t1['price'] - r_t['price']) / dt_yr
        se     = math.sqrt(r_t['std_error']**2 + r_t1['std_error']**2) / dt_yr
        return GreekResult('Theta', value, se)

    def rho(self) -> GreekResult:
        """
        Rho = (P(r+hr) - P(r-hr)) / (2*hr)
        Expressed per 1 bp (x 0.0001).
        """
        rate = self.market.rate
        dr   = self.h_r

        r_up   = self._price(self._new_market(rate=rate + dr), self.pricing_date)
        r_down = self._price(self._new_market(rate=rate - dr), self.pricing_date)

        # Rho pour +1 bp
        value = (r_up['price'] - r_down['price']) / (2 * dr) * 0.0001
        se    = math.sqrt(r_up['std_error']**2 + r_down['std_error']**2) / (2 * dr) * 0.0001
        return GreekResult('Rho', value, se)

    # -- combined computation ---------------------------------------------

    def all_greeks(self) -> AllGreeks:
        """
        Computes all Greeks in a single pass.

        Note: Delta and Gamma share the same pricings (S+/-dS and S0) to
        avoid redundant calls and improve CRN coherence.
        """
        S0 = self.market.underlying
        dS = self.h_S * S0

        # Shared pricings for Delta and Gamma
        r_up   = self._price(self._new_market(S=S0 + dS), self.pricing_date)
        r_0    = self._price(self.market, self.pricing_date)
        r_down = self._price(self._new_market(S=S0 - dS), self.pricing_date)

        # Delta
        d_val = (r_up['price'] - r_down['price']) / (2 * dS)
        d_se  = math.sqrt(r_up['std_error']**2 + r_down['std_error']**2) / (2 * dS)
        delta = GreekResult('Delta', d_val, d_se)

        # Gamma
        g_val = (r_up['price'] - 2 * r_0['price'] + r_down['price']) / dS**2
        g_se  = math.sqrt(
            r_up['std_error']**2 + 4 * r_0['std_error']**2 + r_down['std_error']**2
        ) / dS**2
        gamma = GreekResult('Gamma', g_val, g_se)

        # Vega
        sig  = self.market.vol
        dv   = self.h_v
        rv_up   = self._price(self._new_market(vol=sig + dv), self.pricing_date)
        rv_down = self._price(self._new_market(vol=sig - dv), self.pricing_date)
        v_val = (rv_up['price'] - rv_down['price']) / (2 * dv) * 0.01
        v_se  = math.sqrt(rv_up['std_error']**2 + rv_down['std_error']**2) / (2 * dv) * 0.01
        vega  = GreekResult('Vega', v_val, v_se)

        # Theta
        dt_days    = timedelta(days=self.theta_days)
        date_shift = self.pricing_date + dt_days
        if date_shift < self.option.mat_date:
            r_t1   = self._price(self.market, date_shift)
            dt_yr  = self.theta_days / 365.0
            t_val  = -(r_t1['price'] - r_0['price']) / dt_yr
            t_se   = math.sqrt(r_0['std_error']**2 + r_t1['std_error']**2) / dt_yr
        else:
            t_val, t_se = float('nan'), float('nan')
        theta = GreekResult('Theta', t_val, t_se)

        # Rho
        rate = self.market.rate
        dr   = self.h_r
        rr_up   = self._price(self._new_market(rate=rate + dr), self.pricing_date)
        rr_down = self._price(self._new_market(rate=rate - dr), self.pricing_date)
        rho_val = (rr_up['price'] - rr_down['price']) / (2 * dr) * 0.0001
        rho_se  = math.sqrt(rr_up['std_error']**2 + rr_down['std_error']**2) / (2 * dr) * 0.0001
        rho     = GreekResult('Rho', rho_val, rho_se)

        return AllGreeks(
            delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho,
            price=r_0['price'], price_se=r_0['std_error'],
        )

    # -- display ----------------------------------------------------------

    @staticmethod
    def print_greeks(g: AllGreeks, width: int = 52) -> None:
        """Prints the Greeks table with their standard errors and 95% CIs."""
        bar = "-" * width
        print(bar)
        print(f"  {'Greek':<8}  {'Value':>10}   {'SE':>10}")
        print(bar)
        for greek in (g.delta, g.gamma, g.vega, g.theta, g.rho):
            if math.isnan(greek.value):
                print(f"  {greek.name:<8}  {'n/a':>10}   {'n/a':>10}")
            else:
                print(f"  {greek.name:<8}  {greek.value:>10.5f}   {greek.se:>10.5f}")
        print(bar)


# ---------------------------------------------------------------------------
#  Standalone example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from datetime import date

    market  = Market(100, 0.20, 0.05, 3.0, date(2026, 10, 30))
    option  = OptionTrade(mat=date(2026, 12, 25), call_put='PUT', ex='EUROPEAN', k=100)
    pdate   = date(2026, 3, 1)

    g_calc  = MCGreeks(market, option, pdate, num_paths=100_000, antithetic=True, seed=42)
    results = g_calc.all_greeks()

    print()
    print("  Monte Carlo Greeks -- PUT EUROPEAN   S=100  K=100  sigma=20%  r=5%")
    MCGreeks.print_greeks(results)
    print()
