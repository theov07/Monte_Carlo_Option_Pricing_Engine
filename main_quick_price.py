"""
main_quick_price.py — Pricing rapide d'une option

Modifier le bloc PARAMÈTRES ci-dessous, puis lancer :
    python main_quick_price.py
"""

import math
from datetime import date
from src.market import Market
from src.option_trade import OptionTrade
from src.black_scholes import BlackScholes
from src.monte_carlo_model import MonteCarloModel
from src.regression import BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES  ← modifier ici
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2026, 2, 21)
MATURITY     = date(2027, 2, 21)

UNDERLYING   = 102.45    # S₀
STRIKE       = 100.00    # K
VOL          = 0.28      # σ  (ex : 0.25 = 25 %)
RATE         = 0.04      # r  (ex : 0.04 = 4 %)

# Dividende discret  (mettre DIV_AMOUNT = 0 ou EX_DIV_DATE = None si absent)
DIV_AMOUNT   = 3.0
EX_DIV_DATE  = date(2026, 6, 9)   # None si pas de dividende

CALL_PUT     = 'CALL'        # 'CALL' ou 'PUT'
EXERCISE     = 'EUROPEAN'    # 'EUROPEAN' ou 'AMERICAN'

# Monte Carlo
MC_PATHS      = 50_000
MC_STEPS      = 100           # pas de temps (surtout utile pour l'américain)
MC_ANTITHETIC = True
MC_SEED       = 42

# ══════════════════════════════════════════════════════════════════════════════
#  CALCUL
# ══════════════════════════════════════════════════════════════════════════════

div_date = EX_DIV_DATE if DIV_AMOUNT > 0 else None
market   = Market(UNDERLYING, VOL, RATE, DIV_AMOUNT, div_date)
option   = OptionTrade(mat=MATURITY, call_put=CALL_PUT, ex=EXERCISE, k=STRIKE)
mc       = MonteCarloModel(MC_PATHS, market, option, PRICING_DATE, seed=MC_SEED)

print()
print("═" * 52)
print(f"  Option  : {CALL_PUT}  {EXERCISE}")
print(f"  S={UNDERLYING}  K={STRIKE}  σ={VOL:.0%}  r={RATE:.0%}")
if div_date:
    print(f"  Dividende : {DIV_AMOUNT} le {div_date}")
print(f"  Maturité : {MATURITY}  (évaluation : {PRICING_DATE})")
print("─" * 52)

# — Black-Scholes (européen uniquement)
if EXERCISE == 'EUROPEAN':
    bs_price = BlackScholes(market, option, PRICING_DATE).price()
    print(f"  Black-Scholes          :  {bs_price:>10.4f}")

# — Monte Carlo
if EXERCISE == 'EUROPEAN':
    r = mc.price_european_vectorized(antithetic=MC_ANTITHETIC)
else:
    r = mc.price_american_longstaff_schwartz_vectorized(
        num_steps=MC_STEPS,
        poly_basis=BasisType.LAGUERRE,
        antithetic=MC_ANTITHETIC,
    )

price = r['price']
se    = r['std_error']
N     = len(r['payoffs'])
sigma_payoffs = se * math.sqrt(N)   # écart-type brut des payoffs actualisés

print(f"  Monte Carlo ({MC_PATHS:,} paths)  :  {price:>10.4f}  ±{1.96*se:.4f}  (IC 95 %)")
print(f"  σ payoffs (brut)       :  {sigma_payoffs:>10.4f}  = SE × √N = {se:.4f} × √{N:,}")
print(f"  Variance payoffs       :  {sigma_payoffs**2:>10.4f}  = σ²")
print("═" * 52)
print()
