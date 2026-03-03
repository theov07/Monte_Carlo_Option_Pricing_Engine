"""
fig_delta_convergence.py -- Delta estimator convergence diagnostics.

Figures generated:
  1. delta_convergence.png   -- MC Delta vs N with BS analytical reference and ±2 SE CI
  2. delta_se_loglog.png     -- SE(Delta) log-log with OLS slope fit vs −½ theory

Method:
  Central finite difference with Common Random Numbers (CRN):
      Delta ≈ (V(S0+ε) − V(S0−ε)) / (2ε)
  Same seed is used for V(S0+ε) and V(S0−ε), which cancels most MC noise.
  SE(Delta) ≈ sqrt(SE_up² + SE_down²) / (2ε)

Parameters:
  S0=100  K=100  r=5%  σ=20%  T=1y  ε=0.5  (h_S = 0.5%)

Run:
    python mc_analysis/fig_delta_convergence.py
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
from src.pricing.greeks import MCGreeks, GreeksConfig

# ══════════════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)      # T = 1 year exactly
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20
EPS  = 0.5                           # absolute bump (h_S * S0 = 0.005 * 100)
SEED = 42

MARKET  = Market(S0, SIGMA, R, 0.0, None)
CALL_EU = OptionTrade(mat=MAT_DATE, call_put='CALL', ex='EUROPEAN', k=K)

# Analytical reference
bs_delta = BlackScholes(MARKET, CALL_EU, PRICING_DATE).delta()

N_VALUES = [1_000, 5_000, 10_000, 50_000, 100_000]

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
#  Sweep over N
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("  DELTA CONVERGENCE DIAGNOSTICS")
print(f"  S0={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y  ε={EPS}")
print(f"  BS analytical Delta = {bs_delta:.6f}")
print(f"  h_S = {EPS/S0:.4f}  (ε/S0)  -- same as MCGreeks default")
print("=" * 60)

print("\nRunning bump-and-reprice (CRN, antithetic=False):")
print(f"  {'N':>8}  {'Delta':>10}  {'SE':>10}  {'|Δ-BS|':>10}  CI 95%")
print("  " + "-" * 62)

deltas, ses = [], []
for N in N_VALUES:
    g = MCGreeks(
        market       = MARKET,
        option       = CALL_EU,
        pricing_date = PRICING_DATE,
        config       = GreeksConfig(
            num_paths  = N,
            antithetic = False,     # no antithetic: SE scales cleanly as N^{-1/2}
            seed       = SEED,
            h_S        = EPS / S0,  # relative bump → dS = EPS = 0.5
        ),
    )
    res = g.delta()
    deltas.append(res.value)
    ses.append(res.se)
    lo, hi = res.value - 2 * res.se, res.value + 2 * res.se
    print(f"  {N:>8,}  {res.value:>10.6f}  {res.se:>10.6f}  "
          f"{abs(res.value - bs_delta):>10.6f}  [{lo:.5f}, {hi:.5f}]")

deltas = np.array(deltas)
ses    = np.array(ses)
N_arr  = np.array(N_VALUES, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
#  Figure 1 -- Delta vs N  (semilog-x)
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

ax.fill_between(N_arr, deltas - 2*ses, deltas + 2*ses,
                alpha=0.2, color='steelblue', label='±2 SE confidence interval')
ax.semilogx(N_arr, deltas, 'o-', color='steelblue', lw=2, ms=7,
            label='MC Delta  (CRN, central diff.)')
ax.axhline(bs_delta, color='tomato', ls='--', lw=2,
           label=f'Black–Scholes  Δ = {bs_delta:.5f}')

ax.set_xlabel('N  (number of paths)')
ax.set_ylabel('Delta  (∂V/∂S)')
ax.set_title(
    f'Delta Convergence vs Number of Paths\n'
    f'S₀={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y  ε={EPS}  (CRN bump-and-reprice)')
ax.legend(fontsize=10)
plt.tight_layout()

out1 = os.path.join(FIGURES_DIR, 'delta_convergence.png')
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  Saved: {out1}")

# ══════════════════════════════════════════════════════════════════════════════
#  Figure 2 -- SE(Delta) log-log
# ══════════════════════════════════════════════════════════════════════════════

# OLS fit in log-log space
log_N  = np.log(N_arr)
log_se = np.log(ses)
slope, intercept = np.polyfit(log_N, log_se, 1)
fit_se = np.exp(intercept) * N_arr ** slope

# Theoretical O(N^{-1/2}) reference anchored at first point
theo = ses[0] * (N_arr[0] / N_arr) ** 0.5

fig, ax = plt.subplots(figsize=(8, 5))

ax.loglog(N_arr, ses,    'o-', color='steelblue', lw=2, ms=7,
          label='Empirical SE(Δ)')
ax.loglog(N_arr, fit_se, '--', color='tomato', lw=2,
          label=f'OLS fit  slope = {slope:.3f}')
ax.loglog(N_arr, theo,   ':',  color='gray',   lw=1.5,
          label='Theoretical  O(N^{−½})')

# Annotate slope on fitted line
mid = len(N_arr) // 2
ax.annotate(
    f'slope ≈ {slope:.3f}',
    xy=(N_arr[mid], fit_se[mid]),
    xytext=(N_arr[mid] * 1.6, fit_se[mid] * 2.0),
    fontsize=10, color='tomato',
    arrowprops=dict(arrowstyle='->', color='tomato'),
)

ax.set_xlabel('N  (number of paths)')
ax.set_ylabel('SE(Delta)')
ax.set_title(
    f'Standard Error of Delta -- log-log  (fitted slope ≈ {slope:.3f})\n'
    f'Expected: slope = −0.5   [O(N^{{−1/2}}) convergence]')
ax.legend(fontsize=10)
plt.tight_layout()

out2 = os.path.join(FIGURES_DIR, 'delta_se_loglog.png')
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: {out2}")
