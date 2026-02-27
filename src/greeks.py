"""
greeks.py — Calcul des Greeks Monte Carlo par différences finies (bump & reprice)

Méthode : différences finies centrées avec Common Random Numbers (CRN).
Le même seed est utilisé pour les deux pricings d'un bump, ce qui réduit
fortement la variance de l'estimateur de chaque Greek.

Greeks calculés :
  Delta  — ∂P/∂S₀          bump de h_S  sur le sous-jacent
  Gamma  — ∂²P/∂S₀²        bump de h_S  (schéma centré d'ordre 2)
  Vega   — ∂P/∂σ            bump de h_v  sur la volatilité
  Theta  — ∂P/∂t  (–∂P/∂T) décroissance en 1 jour calendaire
  Rho    — ∂P/∂r            bump de h_r  sur le taux

SE des Greeks = propagation de l'erreur standard des priceurs élémentaires :
  SE(ΔP / 2h) = sqrt(SE_up² + SE_down²) / (2h)   (différence centrale)
  SE(Gamma)   = sqrt(SE_up² + 4·SE_0² + SE_down²) / h²

Usage rapide
------------
from greeks import MCGreeks

g = MCGreeks(
    market=market,
    option=option,
    pricing_date=pricing_date,
    num_paths=100_000,
    antithetic=True,
    seed=42,
)
results = g.all_greeks()
g.print_greeks(results)
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
    """Valeur + erreur standard d'un Greek."""
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
    Greeks Monte Carlo par différences finies + Common Random Numbers.

    Parameters
    ----------
    market : Market
    option : OptionTrade
    pricing_date : date
    num_paths : int
        Nombre de simulations par pricing.
    antithetic : bool
        Réduction de variance par variables antithétiques.
    seed : int or None
        Graine de base ; chaque pricing interne utilise cette même graine
        pour garantir la corrélation CRN.
    num_steps : int
        Nombre de pas de temps pour l'américain (ignoré pour l'européen).
    h_S : float
        Bump relatif sur S₀  (défaut 0.5 %).
    h_v : float
        Bump absolu  sur σ   (défaut 1 %).
    h_r : float
        Bump absolu  sur r   (défaut 1 bp = 0.0001).
    theta_days : float
        Décalage temporel pour Theta en jours calendaires (défaut 1).
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

    # ── helpers internes ────────────────────────────────────────────────────

    def _new_market(self, *, S=None, vol=None, rate=None) -> Market:
        """Clone le market en changeant un seul paramètre."""
        return Market(
            underlying  = S    if S    is not None else self.market.underlying,
            vol         = vol  if vol  is not None else self.market.vol,
            rate        = rate if rate is not None else self.market.rate,
            div_a       = self.market.div_a,
            ex_div_date = self.market.ex_div_date,
        )

    def _price(self, market: Market, pricing_date: date) -> dict:
        """Lance un pricing MC avec les paramètres donnés (CRN via seed fixe)."""
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
        Δ = (P(S+hS) – P(S–hS)) / (2·hS)
        Différence centrale, bump absolu dS = h_S * S₀.
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
        Γ = (P(S+hS) – 2·P(S) + P(S–hS)) / hS²
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
        Vega = (P(σ+hv) – P(σ–hv)) / (2·hv)
        Exprimé pour 1 % de mouvement de vol (× 0.01).
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
        Θ = –(P(t+Δt) – P(t)) / Δt  (convention : perte par jour)
        Δt = theta_days / 365.
        """
        dt_days = timedelta(days=self.theta_days)
        date_shifted = self.pricing_date + dt_days

        # Vérification : ne pas dépasser la maturité
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
        ρ = (P(r+hr) – P(r–hr)) / (2·hr)
        Exprimé pour 1 bp (× 0.0001).
        """
        rate = self.market.rate
        dr   = self.h_r

        r_up   = self._price(self._new_market(rate=rate + dr), self.pricing_date)
        r_down = self._price(self._new_market(rate=rate - dr), self.pricing_date)

        # Rho pour +1 bp
        value = (r_up['price'] - r_down['price']) / (2 * dr) * 0.0001
        se    = math.sqrt(r_up['std_error']**2 + r_down['std_error']**2) / (2 * dr) * 0.0001
        return GreekResult('Rho', value, se)

    # ── Calcul groupé ───────────────────────────────────────────────────────

    def all_greeks(self) -> AllGreeks:
        """
        Calcule tous les Greeks en une seule passe.

        Note : Delta et Gamma partagent les mêmes pricings (S±dS + S₀) pour
        éviter les appels redondants et améliorer la cohérence CRN.
        """
        S0 = self.market.underlying
        dS = self.h_S * S0

        # Pricings partagés Delta / Gamma
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

    # ── Affichage ───────────────────────────────────────────────────────────

    @staticmethod
    def print_greeks(g: AllGreeks, width: int = 52) -> None:
        """Affiche le tableau des Greeks avec leurs SE et IC 95 %."""
        bar = "─" * width
        print(bar)
        print(f"  {'Greek':<8}  {'Valeur':>10}   {'SE':>10}")
        print(bar)
        for greek in (g.delta, g.gamma, g.vega, g.theta, g.rho):
            if math.isnan(greek.value):
                print(f"  {greek.name:<8}  {'n/a':>10}   {'n/a':>10}")
            else:
                print(f"  {greek.name:<8}  {greek.value:>10.5f}   {greek.se:>10.5f}")
        print(bar)


# ─────────────────────────────────────────────────────────────────────────────
#  Exemple autonome
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    from datetime import date

    market  = Market(100, 0.20, 0.05, 3.0, date(2026, 10, 30))
    option  = OptionTrade(mat=date(2026, 12, 25), call_put='PUT', ex='EUROPEAN', k=100)
    pdate   = date(2026, 3, 1)

    g_calc  = MCGreeks(market, option, pdate, num_paths=100_000, antithetic=True, seed=42)
    results = g_calc.all_greeks()

    print()
    print("  Greeks Monte Carlo — PUT EUROPEAN   S=100  K=100  σ=20 %  r=5 %")
    MCGreeks.print_greeks(results)
    print()
