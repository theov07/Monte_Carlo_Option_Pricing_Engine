"""
Monte Carlo Greeks via finite differences (bump & reprice).

Method: central finite differences with Common Random Numbers (CRN).
The same seed is used for both pricings in a bump, which substantially
reduces the variance of each Greek estimator.

Greeks computed:
  Delta  -- dP/dS0           bump of h_S on the underlying
  Gamma  -- d2P/dS0^2        bump of h_S (second-order central scheme)
  Vega   -- dP/dsigma        bump of h_v on volatility
  Theta  -- dP/dt (-dP/dT)   1 calendar day decay
  Rho    -- dP/dr            bump of h_r on the rate

SE of Greeks = error propagation from elementary pricers:
  SE(Delta) = sqrt(SE_up^2 + SE_down^2) / (2h)    (central difference)
  SE(Gamma) = sqrt(SE_up^2 + 4*SE_0^2 + SE_down^2) / h^2
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta

from src.market import Market
from src.monte_carlo_model import MonteCarloModel
from src.option_trade import OptionTrade
from src.regression import BasisType

# ─────────────────────────────────────────────────────────────────────────────
# Result structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GreekResult:
    """Value and standard error of a single Greek."""

    name: str
    value: float
    se: float

    def ci95(self) -> tuple[float, float]:
        return (self.value - 1.96 * self.se, self.value + 1.96 * self.se)

    def __str__(self) -> str:
        return f"{self.name:<8}  {self.value:>10.5f}  ±{1.96 * self.se:.5f}"


@dataclass
class AllGreeks:
    delta: GreekResult
    gamma: GreekResult
    vega: GreekResult
    theta: GreekResult
    rho: GreekResult
    price: float
    price_se: float


@dataclass(frozen=True)
class GreeksConfig:
    """Configuration for Monte Carlo Greeks."""

    num_paths: int = 100_000
    antithetic: bool = True
    seed: int | None = None
    num_steps: int = 252  # for American (LS); ignored for European
    h_S: float = 0.005
    h_v: float = 0.01
    h_r: float = 0.0001
    theta_days: float = 1.0
    basis: BasisType = BasisType.POWER


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────


class MCGreeks:
    """
    Monte Carlo Greeks via finite differences + Common Random Numbers (CRN).

    Parameters
    ----------
    market : Market
    option : OptionTrade
    pricing_date : date
    config : GreeksConfig
        All numerical/MC parameters are stored in this config object so that
        __init__ stays under the "max args" checkstyle constraint.
    """

    def __init__(
        self,
        market: Market,
        option: OptionTrade,
        pricing_date: date,
        config: GreeksConfig | None = None,
    ) -> None:
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.cfg = config if config is not None else GreeksConfig()

    # ── private helpers ──────────────────────────────────────────────────────

    def _new_market(self, *, S: float | None = None, vol: float | None = None, rate: float | None = None) -> Market:
        """Clone the market object with one parameter changed."""
        return Market(
            underlying=S if S is not None else self.market.underlying,
            vol=vol if vol is not None else self.market.vol,
            rate=rate if rate is not None else self.market.rate,
            div_a=self.market.div_a,
            ex_div_date=self.market.ex_div_date,
        )

    def _price(self, market: Market, pricing_date: date) -> dict:
        """Run a MC pricing with the given parameters (CRN via fixed seed)."""
        mc = MonteCarloModel(self.cfg.num_paths, market, self.option, pricing_date, seed=self.cfg.seed)
        if self.option.is_american():
            return mc.price_american_longstaff_schwartz_vectorized(
                num_steps=self.cfg.num_steps,
                poly_basis=self.cfg.basis,
                antithetic=self.cfg.antithetic,
            )
        return mc.price_european_vectorized(antithetic=self.cfg.antithetic)

    def _price_S_bumps(self) -> tuple[dict, dict, dict, float]:
        """Shared pricing calls for Delta and Gamma (S0 +/- dS, S0)."""
        S0 = self.market.underlying
        dS = self.cfg.h_S * S0
        r_up = self._price(self._new_market(S=S0 + dS), self.pricing_date)
        r_0 = self._price(self.market, self.pricing_date)
        r_down = self._price(self._new_market(S=S0 - dS), self.pricing_date)
        return r_up, r_0, r_down, dS

    @staticmethod
    def _delta_from_shared(r_up: dict, r_down: dict, dS: float) -> GreekResult:
        value = (r_up["price"] - r_down["price"]) / (2.0 * dS)
        se = math.sqrt(r_up["std_error"] ** 2 + r_down["std_error"] ** 2) / (2.0 * dS)
        return GreekResult("Delta", value, se)

    @staticmethod
    def _gamma_from_shared(r_up: dict, r_0: dict, r_down: dict, dS: float) -> GreekResult:
        value = (r_up["price"] - 2.0 * r_0["price"] + r_down["price"]) / (dS**2)
        se = (
            math.sqrt(r_up["std_error"] ** 2 + 4.0 * r_0["std_error"] ** 2 + r_down["std_error"] ** 2) / (dS**2)
        )
        return GreekResult("Gamma", value, se)

    def _vega(self) -> GreekResult:
        sig = self.market.vol
        dv = self.cfg.h_v
        r_up = self._price(self._new_market(vol=sig + dv), self.pricing_date)
        r_down = self._price(self._new_market(vol=sig - dv), self.pricing_date)
        value = (r_up["price"] - r_down["price"]) / (2.0 * dv) * 0.01
        se = math.sqrt(r_up["std_error"] ** 2 + r_down["std_error"] ** 2) / (2.0 * dv) * 0.01
        return GreekResult("Vega", value, se)

    def _theta(self, r_0: dict) -> GreekResult:
        dt_days = timedelta(days=self.cfg.theta_days)
        date_shift = self.pricing_date + dt_days

        if date_shift >= self.option.mat_date:
            return GreekResult("Theta", float("nan"), float("nan"))

        r_t1 = self._price(self.market, date_shift)
        dt_yr = self.cfg.theta_days / 365.0
        value = -(r_t1["price"] - r_0["price"]) / dt_yr
        se = math.sqrt(r_0["std_error"] ** 2 + r_t1["std_error"] ** 2) / dt_yr
        return GreekResult("Theta", value, se)

    def _rho(self) -> GreekResult:
        rate = self.market.rate
        dr = self.cfg.h_r
        r_up = self._price(self._new_market(rate=rate + dr), self.pricing_date)
        r_down = self._price(self._new_market(rate=rate - dr), self.pricing_date)
        value = (r_up["price"] - r_down["price"]) / (2.0 * dr) * 0.0001
        se = math.sqrt(r_up["std_error"] ** 2 + r_down["std_error"] ** 2) / (2.0 * dr) * 0.0001
        return GreekResult("Rho", value, se)

    # ── public API ───────────────────────────────────────────────────────────

    def delta(self) -> GreekResult:
        r_up, _, r_down, dS = self._price_S_bumps()
        return self._delta_from_shared(r_up, r_down, dS)

    def gamma(self) -> GreekResult:
        r_up, r_0, r_down, dS = self._price_S_bumps()
        return self._gamma_from_shared(r_up, r_0, r_down, dS)

    def vega(self) -> GreekResult:
        return self._vega()

    def theta(self) -> GreekResult:
        r_0 = self._price(self.market, self.pricing_date)
        return self._theta(r_0)

    def rho(self) -> GreekResult:
        return self._rho()

    def all_greeks(self) -> AllGreeks:
        """Compute all Greeks with shared calls for Delta/Gamma."""
        r_up, r_0, r_down, dS = self._price_S_bumps()
        delta = self._delta_from_shared(r_up, r_down, dS)
        gamma = self._gamma_from_shared(r_up, r_0, r_down, dS)
        vega = self._vega()
        theta = self._theta(r_0)
        rho = self._rho()
        return AllGreeks(
            delta=delta,
            gamma=gamma,
            vega=vega,
            theta=theta,
            rho=rho,
            price=r_0["price"],
            price_se=r_0["std_error"],
        )

    # ── display ──────────────────────────────────────────────────────────────

    @staticmethod
    def print_greeks(g: AllGreeks, width: int = 52) -> None:
        """Print a table of Greeks with standard errors."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Standalone example
# ─────────────────────────────────────────────────────────────────────────────



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