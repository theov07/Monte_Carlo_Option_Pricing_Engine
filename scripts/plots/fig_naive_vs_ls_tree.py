"""
fig_naive_vs_ls_tree.py -- Naive MC vs Longstaff-Schwartz vs Trinomial Tree.

For each N in [1k, 5k, 10k, 50k, 100k] computes:
  - Naive backward-induction American MC  (price_american_naive_vectorized)
  - Longstaff-Schwartz American MC        (price_american_longstaff_schwartz_vectorized)
  - Trinomial tree benchmark              (100 steps, computed once)

Figure: price vs N (log-scale on N) with ±2 SE confidence intervals.

Parameters: S0=100  K=100  r=5%  σ=20%  T=1y  M=100 steps
Save: figures/naive_vs_ls_tree.png

Run:
    python mc_analysis/fig_naive_vs_ls_tree.py
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
from src.models.brownian_motion import BrownianMotion
from src.pricing.regression import BasisType
from src.pricing.monte_carlo_model import MonteCarloModel
from src.benchmarks.trinomial_tree.tree import Tree
from src.benchmarks.trinomial_tree.trinomial_model import TrinomialModel


# ══════════════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════════════

PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)      # T = 1 year exactly
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20

NUM_STEPS    = 100            # MC discretisation and tree steps
SEED         = 42

MARKET  = Market(S0, SIGMA, R, 0.0, None)
PUT_AM  = OptionTrade(mat=MAT_DATE, call_put='PUT', ex='AMERICAN', k=K)

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
#  Trinomial tree reference  (single computation)
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 64)
print("  NAIVE MC  vs  LONGSTAFF-SCHWARTZ  vs  TRINOMIAL TREE")
print(f"  S0={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y  M={NUM_STEPS} steps")
print("=" * 64)

print(f"\nComputing trinomial tree reference ({NUM_STEPS} steps)...")
tree = Tree(NUM_STEPS, MARKET, PUT_AM, PRICING_DATE, prunning_threshold=1e-8)
tree.build_tree()
tm   = TrinomialModel(PRICING_DATE, tree)
TREE_PRICE = tm.price(PUT_AM, "backward")
print(f"  Tree price = {TREE_PRICE:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo sweep over N
# ══════════════════════════════════════════════════════════════════════════════

naive_prices, naive_ses = [], []
ls_prices,    ls_ses    = [], []

print(f"\n{'N':>8}  {'Naive price':>12}  {'Naive SE':>10}  "
      f"{'LS price':>12}  {'LS SE':>10}  {'|Naive-Tree|':>14}  {'|LS-Tree|':>12}")
print("  " + "-" * 90)

for N in N_VALUES:
    mc = MonteCarloModel(N, MARKET, PUT_AM, PRICING_DATE, seed=SEED)

    # Naive backward induction (vectorized)
    r_naive = mc.price_american_naive_vectorized(
        num_steps  = NUM_STEPS,
        antithetic = True,
    )
    # Longstaff-Schwartz (vectorized, degree-3 Power basis)
    r_ls = mc.price_american_longstaff_schwartz_vectorized(
        num_steps  = NUM_STEPS,
        poly_basis = BasisType.POWER,
        antithetic = True,
    )

    naive_prices.append(r_naive['price'])
    naive_ses.append(r_naive['std_error'])
    ls_prices.append(r_ls['price'])
    ls_ses.append(r_ls['std_error'])

    print(f"  {N:>6,}  {r_naive['price']:>12.5f}  {r_naive['std_error']:>10.5f}  "
          f"{r_ls['price']:>12.5f}  {r_ls['std_error']:>10.5f}  "
          f"{abs(r_naive['price'] - TREE_PRICE):>14.5f}  "
          f"{abs(r_ls['price'] - TREE_PRICE):>12.5f}")

naive_prices = np.array(naive_prices)
naive_ses    = np.array(naive_ses)
ls_prices    = np.array(ls_prices)
ls_ses       = np.array(ls_ses)
N_arr        = np.array(N_VALUES, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
#  Figure
# ══════════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(9, 5))

# ── Confidence bands
ax.fill_between(N_arr,
                naive_prices - 2 * naive_ses,
                naive_prices + 2 * naive_ses,
                alpha=0.15, color='steelblue')
ax.fill_between(N_arr,
                ls_prices - 2 * ls_ses,
                ls_prices + 2 * ls_ses,
                alpha=0.15, color='tomato')

# ── Price curves
ax.semilogx(N_arr, naive_prices, 'o-', color='steelblue', lw=2, ms=7,
            label='Naive backward-induction MC  (±2 SE)')
ax.semilogx(N_arr, ls_prices,    's-', color='tomato',    lw=2, ms=7,
            label='Longstaff-Schwartz MC  (±2 SE)')

# ── Tree reference
ax.axhline(TREE_PRICE, color='forestgreen', ls='--', lw=2,
           label=f'Trinomial tree  ({NUM_STEPS} steps)  =  {TREE_PRICE:.5f}')

ax.set_xlabel('N  (number of paths)')
ax.set_ylabel('American PUT price')
ax.set_title(
    f'Naive MC vs Longstaff-Schwartz vs Trinomial Tree\n'
    f'American PUT  S₀={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T=1y  M={NUM_STEPS} steps')
ax.legend(fontsize=10)
plt.tight_layout()

out = os.path.join(FIGURES_DIR, 'naive_vs_ls_tree.png')
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\n  Saved: {out}")
