"""
INPUT : Market parameters, option parameters, and MC simulation settings.
OUTPUT : Option price via MC (with CI) and BS (if European), MC Greeks, empirical price variance over N_RUNS independent runs.
"""

import math
import numpy as np
from datetime import date
from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.regression import BasisType
from src.pricing.monte_carlo_model import MonteCarloModel
from src.pricing.black_scholes import BlackScholes
from src.pricing.greeks import MCGreeks, GreeksConfig



# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2026, 2, 26)
MATURITY     = date(2027, 4, 26)

UNDERLYING   = 100    # S₀
STRIKE       = 90    # K
VOL          = 0.25      # σ  (e.g. 0.25 = 25%)
RATE         = 0.04      # r  (e.g. 0.04 = 4%)

# Discrete dividend  (set DIV_AMOUNT = 0 or EX_DIV_DATE = None if absent)
DIV_AMOUNT   = 3.0
EX_DIV_DATE  = date(2026, 6, 21)   # None if no dividend

CALL_PUT     = 'PUT'        # 'CALL' or 'PUT'
EXERCISE     = 'AMERICAN'    # 'EUROPEAN' or 'AMERICAN'

# Monte Carlo
MC_PATHS      = 100_000
MC_STEPS      = 250           # time steps (for American options)
MC_ANTITHETIC = True
MC_SEED       = 2

# Independent runs with different seeds to estimate the empirical price variance
N_RUNS        = 10

# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTATION
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
    print(f"  Dividend  : {DIV_AMOUNT} on {div_date}")
print(f"  Maturity  : {MATURITY}  (Pricing date: {PRICING_DATE})")
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
sigma_payoffs = se * math.sqrt(N)   # raw std dev of discounted payoffs

print(f"  Monte Carlo ({MC_PATHS:,} paths)  :  {price:>10.4f}  ±{1.96*se:.4f}  (95% CI)")
print(f"  σ payoffs (raw)        :  {sigma_payoffs:>10.4f}  = SE × √N = {se:.4f} × √{N:,}")
print(f"  Variance payoffs       :  {sigma_payoffs**2:>10.4f}  = σ²  (variance of one individual payoff)")
print(f"  Price variance  (SE²)  :  {se**2:>10.6f}  = σ²/N  (estimator variance)")
print("─" * 52)

# Monte Carlo Greeks (finite difference method)
print(f"  Computing MC Greeks ({MC_PATHS:,} paths, CRN)...")

g_config = GreeksConfig(
    num_paths=MC_PATHS,
    antithetic=MC_ANTITHETIC,
    seed=MC_SEED,
    num_steps=MC_STEPS,
)

g_calc = MCGreeks(
    market=market,
    option=option,
    pricing_date=PRICING_DATE,
    config=g_config,
)

greeks = g_calc.all_greeks()


print(f"  {'Greek':<8}  {'Value':>10}   {'SE':>10}")
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
hedge_sign = "buy" if delta_val < 0 else "sell"
print(f"  Delta hedge  : if long 1 option -> {hedge_sign} {abs(delta_val):.4f} share(s)")
print(f"                 if long N options -> {hedge_sign} N × {abs(delta_val):.4f} shares")
print(f"                 (for N short options: reverse the direction)")
gamma_pct = gamma_val * UNDERLYING * 0.01
print(f"  Gamma ×(S₀×1%) = {gamma_val:.5f} × {UNDERLYING}×0.01 = {gamma_pct:.5f}")
print(f"                 → delta change for a 1% underlying move")
print("─" * 52)

# EMPIRICAL VARIANCE
print(f"  Computing empirical variance over {N_RUNS} runs...")
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

print(f"  Mean     ({N_RUNS} runs)        :  {mean_runs:>10.4f}")
print(f"  Empirical price variance :  {var_runs:>10.6f}  (over {N_RUNS} runs)")
print(f"  Inter-run std dev        :  {std_runs:>10.6f}  ~= theoretical SE = {se:.6f}")
print("═" * 52)
print()