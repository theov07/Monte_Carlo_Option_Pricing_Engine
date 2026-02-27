"""
INPUT : PARAMÈTRES DU MARCHÉ, DE L'OPTION, ET DE LA SIMULATION MC
OUTPUT : prix de l'option par MC (avec IC) et par BS (si européen), Greeks MC, variance empirique du prix sur N_RUNS répétitions indépendantes.
"""

import math
import numpy as np
from datetime import date
from src.market import Market
from src.option_trade import OptionTrade
from src.black_scholes import BlackScholes
from src.monte_carlo_model import MonteCarloModel
from src.regression import BasisType
from src.greeks import MCGreeks

# ══════════════════════════════════════════════════════════════════════════════
#  PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2026, 2, 26)
MATURITY     = date(2027, 4, 26)

UNDERLYING   = 100    # S₀
STRIKE       = 90    # K
VOL          = 0.25      # σ  (ex : 0.25 = 25 %)
RATE         = 0.04      # r  (ex : 0.04 = 4 %)

# Dividende discret  (mettre DIV_AMOUNT = 0 ou EX_DIV_DATE = None si absent)
DIV_AMOUNT   = 3.0
EX_DIV_DATE  = date(2026, 6, 21)   # None si pas de dividende

CALL_PUT     = 'PUT'        # 'CALL' ou 'PUT'
EXERCISE     = 'AMERICAN'    # 'EUROPEAN' ou 'AMERICAN'

# Monte Carlo
MC_PATHS      = 100_000
MC_STEPS      = 250           # pas de temps (pour l'américain)
MC_ANTITHETIC = True
MC_SEED       = 2

# Répétitions indépendantes en changeant la seed pour estimer la variance empirique du prix
N_RUNS        = 10

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
print(f"  Maturité : {MATURITY}  (Date d'évaluation : {PRICING_DATE})")
print("─" * 52)

# Black-Scholes (pour européen uniquement)
if EXERCISE == 'EUROPEAN':
    bs_price = BlackScholes(market, option, PRICING_DATE).price()
    print(f"  Black-Scholes          :  {bs_price:>10.4f}")

# Monte Carlo
if EXERCISE == 'EUROPEAN':
    r = mc.price_european_vectorized(antithetic=MC_ANTITHETIC)
else:
    r = mc.price_american_longstaff_schwartz_vectorized(
        num_steps=MC_STEPS,
        poly_basis=BasisType.POWER,
        antithetic=MC_ANTITHETIC,
    )

price = r['price']
se    = r['std_error']
N     = len(r['payoffs'])
sigma_payoffs = se * math.sqrt(N)   # écart-type brut des payoffs actualisés

print(f"  Monte Carlo ({MC_PATHS:,} paths)  :  {price:>10.4f}  ±{1.96*se:.4f}  (IC 95 %)")
print(f"  σ payoffs (brut)       :  {sigma_payoffs:>10.4f}  = SE × √N = {se:.4f} × √{N:,}")
print(f"  Variance payoffs       :  {sigma_payoffs**2:>10.4f}  = σ²  (variance d'un payoff individuel)")
print(f"  Variance du prix (SE²) :  {se**2:>10.6f}  = σ²/N  (variance de l'estimateur)")
print("─" * 52)

# Greeks Monte Carlo (méthode par différence finies)
print(f"  Calcul des Greeks MC ({MC_PATHS:,} paths, CRN)...")
g_calc   = MCGreeks(
    market=market,
    option=option,
    pricing_date=PRICING_DATE,
    num_paths=MC_PATHS,
    antithetic=MC_ANTITHETIC,
    seed=MC_SEED,
    num_steps=MC_STEPS,
)
greeks = g_calc.all_greeks()
print(f"  {'Greek':<8}  {'Valeur':>10}   {'SE':>10}")
print("─" * 52)
for g in (greeks.delta, greeks.gamma, greeks.vega, greeks.theta, greeks.rho):
    if math.isnan(g.value):
        print(f"  {g.name:<8}  {'n/a':>10}   {'n/a':>10}")
    else:
        print(f"  {g.name:<8}  {g.value:>10.5f}   {g.se:>10.5f}")
print("─" * 52)

# Hedging 
delta_val = greeks.delta.value
gamma_val = greeks.gamma.value
hedge_sign = "acheter" if delta_val < 0 else "vendre"
print(f"  Delta hedge  : si long 1 option → {hedge_sign} {abs(delta_val):.4f} action(s)")
print(f"                 si long N options → {hedge_sign} N × {abs(delta_val):.4f} actions")
print(f"                 (pour N options short : inverser la direction)")
gamma_pct = gamma_val * UNDERLYING * 0.01
print(f"  Gamma ×(S₀×1%) = {gamma_val:.5f} × {UNDERLYING}×0.01 = {gamma_pct:.5f}")
print(f"                 → variation du delta pour un mouvement de 1 % du sous-jacent")
print("─" * 52)

# VARIANCE EMPIRIQUE 
print(f"  Calcul variance empirique sur {N_RUNS} runs...")
prices_runs = []
for seed in range(MC_SEED, MC_SEED + N_RUNS):
    mc_i = MonteCarloModel(MC_PATHS, market, option, PRICING_DATE, seed=seed)
    if EXERCISE == 'EUROPEAN':
        r_i = mc_i.price_european_vectorized(antithetic=MC_ANTITHETIC)
    else:
        r_i = mc_i.price_american_longstaff_schwartz_vectorized(
            num_steps=MC_STEPS,
            poly_basis=BasisType.POWER,
            antithetic=MC_ANTITHETIC,
        )
    prices_runs.append(r_i['price'])

prices_arr = np.array(prices_runs)
mean_runs  = np.mean(prices_arr)
var_runs   = np.var(prices_arr, ddof=1)
std_runs   = np.std(prices_arr, ddof=1)

print(f"  Moyenne  ({N_RUNS} runs)        :  {mean_runs:>10.4f}")
print(f"  Variance empirique prix  :  {var_runs:>10.6f}  (sur {N_RUNS} runs)")
print(f"  Écart-type inter-runs    :  {std_runs:>10.6f}  ≈ SE théorique = {se:.6f}")
print("═" * 52)
print()
