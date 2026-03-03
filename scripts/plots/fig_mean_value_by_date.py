"""
fig_mean_value_by_date.py -- Mean option value at each time step (backward induction trace).

For each time step j ∈ {0, …, M}, computes:

    V_j = mean_paths[ cf_j ] × exp(−r × j × dt)

where cf_j is the optimal cashflow vector at step j during the LS backward pass:
  - European: cf_j = discounted future payoff  →  V_j constant (martingale)
  - American: cf_j = max(intrinsic, continuation) at each step  →  V_j ↗ toward t=0

Interpretation:
    V_j = expected present value of the option under the optimal strategy
          available from time t_j to maturity.

    V_0 = option price (all exercise decisions available)
    V_M = European-equivalent value (exercise only at maturity)
    V_0 − V_M = total early-exercise premium

Figure saved: figures/mean_option_value_by_date.png

Run:
    python mc_analysis/fig_mean_value_by_date.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from datetime import date


from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.black_scholes import BlackScholes
from src.models.brownian_motion import BrownianMotion
from src.pricing.regression import Regression, BasisType


# ══════════════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)      # T = 1 year exactly
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20
T         = 1.0
NUM_PATHS = 100_000
NUM_STEPS = 100
SEED      = 42

MARKET = Market(S0, SIGMA, R, 0.0, None)
PUT_EU = OptionTrade(mat=MAT_DATE, call_put='PUT', ex='EUROPEAN', k=K)
PUT_AM = OptionTrade(mat=MAT_DATE, call_put='PUT', ex='AMERICAN', k=K)

BS_PUT  = BlackScholes(MARKET, PUT_EU, PRICING_DATE).price()
BS_CALL = BlackScholes(MARKET, PUT_EU, PRICING_DATE)   # reuse for delta reference

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style -- matches existing project plots
plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 100,
})

# ══════════════════════════════════════════════════════════════════════════════
#  Path simulation  (single draw, shared for EU and AM)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  MEAN OPTION VALUE BY DATE")
print(f"  S0={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y")
print(f"  N={NUM_PATHS:,} paths  M={NUM_STEPS} steps  seed={SEED}")
print(f"  Black-Scholes PUT = {BS_PUT:.6f}")
print("=" * 60)

print(f"\nSimulating {NUM_PATHS:,} paths ({NUM_STEPS} steps)...")
bm = BrownianMotion(NUM_PATHS, NUM_STEPS, T, antithetic=False, seed=SEED)
S_paths, _ = bm.generate_paths(S0, R, SIGMA, 0.0, div=0.0, jdiv=None)
print(f"  S_paths shape: {S_paths.shape}  "
      f"(mean S(T) = {S_paths[:, -1].mean():.2f}, expected ≈ {S0 * np.exp(R*T):.2f})")

dt    = T / NUM_STEPS
df    = np.exp(-R * dt)
times = np.linspace(0, T, NUM_STEPS + 1)


def _put_payoff(S: np.ndarray) -> np.ndarray:
    return np.maximum(K - S, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  European backward trace
#  V_eu[j] = mean(cf_j) * exp(-r*j*dt)
#  cf_j = payoff discounted from maturity to step j+1, then one more step to j
#  => V_eu[j] = BS price for all j  (martingale property)
# ══════════════════════════════════════════════════════════════════════════════

print("Computing European backward trace...")
V_eu = np.empty(NUM_STEPS + 1)

cash_flow = _put_payoff(S_paths[:, -1])           # undiscounted payoff at T
V_eu[NUM_STEPS] = np.mean(cash_flow) * np.exp(-R * T)

for j in range(NUM_STEPS - 1, -1, -1):
    cash_flow = cash_flow * df                    # discount one step toward t=0
    V_eu[j]   = np.mean(cash_flow) * np.exp(-R * j * dt)

print(f"  V_eu[0] = {V_eu[0]:.5f}  (BS ref = {BS_PUT:.5f},  "
      f"error = {abs(V_eu[0] - BS_PUT):.2e})")
print(f"  Range: [{V_eu.min():.5f}, {V_eu.max():.5f}]  "
      f"(should be ~flat for European)")

# ══════════════════════════════════════════════════════════════════════════════
#  American backward trace (Longstaff-Schwartz)
#  V_am[j] = mean(cf_j_optimal) * exp(-r*j*dt)
#  cf_j = max(intrinsic, estimated continuation) per path
# ══════════════════════════════════════════════════════════════════════════════

print("Computing American backward trace (Longstaff-Schwartz, degree-3 Laguerre)...")
V_am = np.empty(NUM_STEPS + 1)

cash_flow    = _put_payoff(S_paths[:, -1])        # undiscounted payoff at T
V_am[NUM_STEPS] = np.mean(cash_flow) * np.exp(-R * T)

reg = Regression(degree=3, basis=BasisType.LAGUERRE)

for j in range(NUM_STEPS - 1, -1, -1):
    continuation = cash_flow * df                 # discount from j+1 to j
    intrinsic    = _put_payoff(S_paths[:, j])
    cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)
    V_am[j]      = np.mean(cash_flow) * np.exp(-R * j * dt)

am_price          = V_am[0]
early_exercise_px = am_price - V_eu[0]

print(f"  V_am[0] = {am_price:.5f}  (AM price)")
print(f"  Early exercise premium = {early_exercise_px:.5f}  "
      f"(= V_am[0] − V_eu[0])")
print(f"  Range: [{V_am.min():.5f}, {V_am.max():.5f}]  "
      f"(should increase left → t=0)")

# ══════════════════════════════════════════════════════════════════════════════
#  Figure
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(times, V_eu, 'o-', ms=3, lw=1.5, color='steelblue',
        label=f'European PUT  (BS = {BS_PUT:.4f})')
ax.plot(times, V_am, 'o-', ms=3, lw=1.5, color='tomato',
        label=f'American PUT  (LS price = {am_price:.4f})')

# Theoretical reference lines
ax.axhline(BS_PUT,   color='steelblue', ls=':', lw=1.2, alpha=0.55)
ax.axhline(am_price, color='tomato',    ls=':', lw=1.2, alpha=0.55)

# Shade the early-exercise premium
ax.fill_between(times, V_eu, V_am, alpha=0.12, color='magenta',
                label=f'Early exercise premium  (total = {early_exercise_px:.4f})')

ax.set_xlabel('t  (years)')
ax.set_ylabel('V(t) = mean(CF*) × e^{−rt}   [discounted to t = 0]')
ax.set_title(
    f'Mean Option Value by Date -- Backward Induction (Longstaff-Schwartz)\n'
    f'S₀={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y  '
    f'N={NUM_PATHS:,}  M={NUM_STEPS} steps')
ax.legend(fontsize=10)
ax.set_xlim(0, T)
plt.tight_layout()

out = os.path.join(FIGURES_DIR, 'mean_option_value_by_date.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n  Saved: {out}")
