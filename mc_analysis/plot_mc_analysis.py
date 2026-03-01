"""
plot_mc_analysis.py -- Educational visualisations for Monte Carlo option pricing.

Run:
    python plot_mc_analysis.py

Figures generated:
  1. Log-log convergence  (SE vs N, slope -0.5)
  2. Brownian increment distribution  (histogram + QQ-plot)
  3. Discounted martingale  (mean(e^{-rt}*S(t)) vs t, should equal S0)
  4. Longstaff-Schwartz regression at an intermediate step
     (ITM scatter, polynomial curve, intrinsic value, exercise boundary)
  5. Option value evolution during backward induction
     (EU = constant, AM = grows toward t=0 due to early exercise)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import date

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src.black_scholes import BlackScholes
from src.brownian_motion import BrownianMotion
from src.regression import Regression, BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  Parameters
# ══════════════════════════════════════════════════════════════════════════════
PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)   # T = 1 an exact
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20
T = 1.0

MARKET  = Market(S0, SIGMA, R, 0.0, None)
CALL_EU = OptionTrade(mat=MAT_DATE, call_put='CALL', ex='EUROPEAN', k=K)
PUT_EU  = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='EUROPEAN', k=K)
PUT_AM  = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='AMERICAN', k=K)

BS_CALL = BlackScholes(MARKET, CALL_EU, PRICING_DATE).price()
BS_PUT  = BlackScholes(MARKET, PUT_EU,  PRICING_DATE).price()

# Style global
plt.rcParams.update({
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.titlesize': 12, 'axes.labelsize': 11,
    'figure.dpi': 100,
})

# ══════════════════════════════════════════════════════════════════════════════
#  Utilitaires internes
# ══════════════════════════════════════════════════════════════════════════════

def _payoff_vec(S: np.ndarray, option: OptionTrade) -> np.ndarray:
    if option.is_a_call():
        return np.maximum(S - option.strike, 0)
    return np.maximum(option.strike - S, 0)


def _backward_value_trace(S_paths: np.ndarray, option: OptionTrade,
                           r: float, T: float, num_steps: int,
                           is_american: bool) -> np.ndarray:
    """
    Computes V_j = mean(cash_flow_j) * exp(-r*j*dt) for j = 0, ..., num_steps.

    V_j represents the (t=0-discounted) estimate of the option value
    starting from step j, following the optimal strategy from j to T.

    Properties:
    - European: V_j = price_EU for all j  (martingale)
    - American: V_j grows from V_T ~= price_EU to V_0 = price_AM
                (supermartingale: early exercise adds value)
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff_vec(S_paths[:, -1], option)   # at maturity, undiscounted
    V = np.empty(num_steps + 1)
    V[num_steps] = np.mean(cash_flow) * np.exp(-r * T)

    reg = Regression(degree=3, basis=BasisType.LAGUERRE) if is_american else None

    for j in range(num_steps - 1, -1, -1):
        if is_american:
            continuation = cash_flow * df
            intrinsic    = _payoff_vec(S_paths[:, j], option)
            cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)
        else:
            # European: simply discount (always keep the future CF)
            cash_flow = cash_flow * df

        V[j] = np.mean(cash_flow) * np.exp(-r * j * dt)

    return V


def _regression_snapshot(S_paths: np.ndarray, option: OptionTrade,
                          r: float, T: float, num_steps: int,
                          snap_step: int):
    """
    Returns the data at step `snap_step` during backward induction:
    S_itm, continuation_itm, intrinsic_all, cashflow_all_prev.

    Runs the LS backward pass from T down to snap_step, then captures the state.
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff_vec(S_paths[:, -1], option)
    reg = Regression(degree=3, basis=BasisType.LAGUERRE)

    for j in range(num_steps - 1, snap_step, -1):
        continuation = cash_flow * df
        intrinsic    = _payoff_vec(S_paths[:, j], option)
        cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)

    # At step snap_step: capture data before the exercise decision
    continuation_snap = cash_flow * df
    intrinsic_snap    = _payoff_vec(S_paths[:, snap_step], option)
    S_snap            = S_paths[:, snap_step]

    itm_mask          = intrinsic_snap > 0
    reg_snap          = Regression(degree=3, basis=BasisType.LAGUERRE)
    if itm_mask.sum() > 4:
        reg_snap.fit(S_snap[itm_mask], continuation_snap[itm_mask])

    return S_snap, continuation_snap, intrinsic_snap, itm_mask, reg_snap


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 -- Log-log convergence  (SE vs N)
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence():
    print("  [1/5] Log-log convergence...")

    n_values = [200, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 80_000, 200_000]
    se_list  = []

    for n in n_values:
        mc = MonteCarloModel(n, MARKET, CALL_EU, PRICING_DATE, seed=0)
        r  = mc.price_european_vectorized(antithetic=False)
        se_list.append(r['std_error'])

    ns  = np.array(n_values, dtype=float)
    ses = np.array(se_list)

    # Log-log regression -- theoretical slope -0.5
    slope, intercept, rval, _, _ = stats.linregress(np.log(ns), np.log(ses))
    fit_se = np.exp(intercept) * ns ** slope

    # Theoretical -0.5 line anchored at the first point
    c_theo = ses[0] * ns[0] ** 0.5
    theo   = c_theo / np.sqrt(ns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, ses,  'o-', color='steelblue', lw=2, ms=7, label='Empirical SE')
    ax.loglog(ns, fit_se, '--', color='tomato', lw=2,
              label=f'Log-log fit  (slope = {slope:.3f})')
    ax.loglog(ns, theo, ':', color='gray', lw=1.5,
              label='Theory  1/sqrt(N)  (slope = -0.5)')

    ax.set_xlabel('N  (number of paths)')
    ax.set_ylabel('Standard Error SE')
    ax.set_title(f'Monte Carlo Convergence -- log-log slope ~= {slope:.3f}\n'
                 f'(expected -0.5 by CLT)')
    ax.legend()
    ax.annotate(f'slope = {slope:.3f}', xy=(ns[-1], ses[-1]),
                xytext=(ns[-2]*0.4, ses[-1]*2.5),
                fontsize=10, color='tomato',
                arrowprops=dict(arrowstyle='->', color='tomato'))
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 -- Brownian increment distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_brownian_distribution():
    print("  [2/5] Brownian increment distribution...")

    N_BM, N_STEPS_BM = 20_000, 100
    dt = T / N_STEPS_BM
    bm = BrownianMotion(N_BM, N_STEPS_BM, T, antithetic=False, seed=42)
    dW = bm.generate_increments_vectorized()          # (N_BM, N_STEPS_BM)
    Z  = (dW / np.sqrt(dt)).flatten()                 # normalisé → N(0,1) théorique

    ks_stat, p_val = stats.kstest(Z, 'norm')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Histogram + theoretical density ────────────────────────────────
    ax = axes[0]
    ax.hist(Z, bins=80, density=True, color='steelblue', alpha=0.6,
            edgecolor='white', lw=0.3, label='Z = dW/sqrt(dt)  (observed)')
    x = np.linspace(-4.5, 4.5, 300)
    ax.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1) theoretical')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Z = dW / sqrt(dt)')
    ax.set_ylabel('Density')
    ax.set_title(f'Normalised Brownian Increments\n'
                 f'mu={np.mean(Z):.4f}  sigma={np.std(Z):.4f}  '
                 f'KS p-val={p_val:.4f}')
    ax.legend()

    # ── QQ-plot ──────────────────────────────────────────────────────────
    ax = axes[1]
    sample = np.random.default_rng(0).choice(Z, size=2_000, replace=False)
    (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(sample, dist='norm')
    ax.scatter(osm, osr, s=4, alpha=0.4, color='steelblue', label='Observed quantiles')
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope_qq * line_x + intercept_qq, 'r-', lw=2, label='Theoretical line')
    ax.set_xlabel('Theoretical quantiles N(0,1)')
    ax.set_ylabel('Observed quantiles')
    ax.set_title(f'QQ-plot — R² = {r_qq**2:.5f}')
    ax.legend()

    fig.suptitle('Normality Test for Brownian Increments  (dW ~ N(0, dt))',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 -- Discounted spot martingale property
# ══════════════════════════════════════════════════════════════════════════════
def plot_martingale():
    print("  [3/5] Discounted martingale...")

    N_M, NUM_STEPS = 30_000, 50
    bm = BrownianMotion(N_M, NUM_STEPS, T, antithetic=False, seed=7)
    S_paths, _ = bm.generate_paths(S0, R, SIGMA, 0.0)

    times  = np.arange(NUM_STEPS + 1) * (T / NUM_STEPS)
    disc_means = np.array([
        np.mean(np.exp(-R * t) * S_paths[:, j])
        for j, t in enumerate(times)
    ])
    disc_stds  = np.array([
        np.std(np.exp(-R * t) * S_paths[:, j]) / np.sqrt(N_M)
        for j, t in enumerate(times)
    ])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(times, disc_means, 'o-', ms=4, color='steelblue',
            label='mean(e^{-rt}*S(t))  (empirical)')
    ax.fill_between(times,
                    disc_means - 2 * disc_stds,
                    disc_means + 2 * disc_stds,
                    alpha=0.2, color='steelblue', label='+/-2 SE')
    ax.axhline(S0, color='red', ls='--', lw=1.5,
               label=f'S0 = {S0}  (theoretical value)')

    max_dev = np.max(np.abs(disc_means - S0))
    ax.set_xlabel('t  (years)')
    ax.set_ylabel('mean(e^{-rt} * S(t))')
    ax.set_title(f'Discounted spot martingale property\n'
                 f'E^Q[e^{{-rt}}*S(t)] = S0  --  max deviation = {max_dev:.4f}')
    ax.legend()
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 -- LS regression at an intermediate step
# ══════════════════════════════════════════════════════════════════════════════
def plot_regression_step():
    print("  [4/5] Longstaff-Schwartz regression at an intermediate step...")

    NUM_PATHS, NUM_STEPS = 10_000, 50
    SNAP = NUM_STEPS // 2    # affiche au pas t = T/2

    bm = BrownianMotion(NUM_PATHS, NUM_STEPS, T, antithetic=False, seed=42)
    S_paths, _ = bm.generate_paths(S0, R, SIGMA, 0.0)

    S_snap, cont_snap, iv_snap, itm_mask, reg_snap = _regression_snapshot(
        S_paths, PUT_AM, R, T, NUM_STEPS, SNAP
    )

    t_snap   = SNAP * (T / NUM_STEPS)
    S_itm    = S_snap[itm_mask]
    cont_itm = cont_snap[itm_mask]

    fig, ax = plt.subplots(figsize=(10, 6))

    # ITM path scatter (sub-sample for readability)
    idx_sub = np.random.default_rng(0).choice(len(S_itm),
                                               size=min(1500, len(S_itm)),
                                               replace=False)
    ax.scatter(S_itm[idx_sub], cont_itm[idx_sub],
               s=8, alpha=0.35, color='steelblue', label='Continuation (ITM paths)')

    # Polynomial regression curve
    S_grid = np.linspace(S_itm.min(), S_itm.max(), 300)
    reg_curve = reg_snap.predict(S_grid)
    ax.plot(S_grid, reg_curve, 'r-', lw=2.5, label='Polynomial regression (deg 3)')

    # Intrinsic value IV = max(K - S, 0)
    iv_grid = np.maximum(K - S_grid, 0)
    ax.plot(S_grid, iv_grid, 'g--', lw=2, label=f'Intrinsic value max(K-S,0)')

    # Exercise boundary: where reg ~= IV
    exercise_mask = iv_grid > reg_curve
    if exercise_mask.any() and (~exercise_mask).any():
        # Premier indice où on bascule (non-exercice → exercice)
        boundary_idx = np.where(np.diff(exercise_mask.astype(int)) != 0)[0]
        if len(boundary_idx) > 0:
            S_boundary = S_grid[boundary_idx[0]]
            ax.axvline(S_boundary, color='orange', lw=2, ls='-.',
                       label=f'Exercise boundary ~= {S_boundary:.1f}')
            ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 20],
                              S_grid.min(), S_boundary,
                              alpha=0.07, color='orange')
            ax.text(S_boundary - 3, 0.5, 'Exercise', color='darkorange', fontsize=9,
                    ha='right', va='bottom')
            ax.text(S_boundary + 1, 0.5, 'Hold', color='darkblue', fontsize=9,
                    ha='left', va='bottom')

    ax.axvline(K, color='gray', lw=1, ls=':', alpha=0.6, label=f'Strike K={K}')
    ax.set_xlabel('S(t)  --  Underlying price')
    ax.set_ylabel('Value (in t=0 dollars)')
    ax.set_title(f'Longstaff-Schwartz regression at step j={SNAP}  (t={t_snap:.2f}y)\n'
                 f'N_itm = {itm_mask.sum():,}  --  LAGUERRE basis deg 3')
    ax.legend(loc='upper right')
    ax.set_xlim(S_itm.min() - 2, min(K + 30, S_itm.max() + 2))
    ax.set_ylim(-0.5, None)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 -- Option value evolution during backward induction
# ══════════════════════════════════════════════════════════════════════════════
def plot_backward_value_trace():
    print("  [5/5] Option value evolution during backward pass...")

    NUM_PATHS, NUM_STEPS = 20_000, 100
    bm = BrownianMotion(NUM_PATHS, NUM_STEPS, T, antithetic=False, seed=42)
    S_paths, _ = bm.generate_paths(S0, R, SIGMA, 0.0)

    V_eu = _backward_value_trace(S_paths, PUT_EU, R, T, NUM_STEPS, is_american=False)
    V_am = _backward_value_trace(S_paths, PUT_AM, R, T, NUM_STEPS, is_american=True)

    times = np.arange(NUM_STEPS + 1) * (T / NUM_STEPS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: both curves ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(times, V_eu, 'b-',  lw=2, label='European (martingale)')
    ax.plot(times, V_am, 'r-',  lw=2, label='American (supermartingale)')
    ax.axhline(BS_PUT, color='blue', lw=1, ls='--', alpha=0.6,
               label=f'BS Put = {BS_PUT:.3f}')
    ax.axhline(V_am[0], color='red', lw=1, ls='--', alpha=0.6,
               label=f'AM MC price = {V_am[0]:.3f}')

    ax.set_xlabel('t  (read right to left = backward)')
    ax.set_ylabel('V(t) = mean(CF_t) * e^{-rt}')
    ax.set_title('V_j discounted to t=0 during backward pass\n'
                 '(EU: flat, AM: grows toward t=0)')
    ax.legend(fontsize=9)

    # ── Right: early exercise premium = V_AM - V_EU ────────────────────
    ax = axes[1]
    premium = V_am - V_eu
    ax.plot(times, premium, 'm-', lw=2, label='American premium V_AM - V_EU')
    ax.axhline(0, color='gray', lw=0.8)
    ax.fill_between(times, 0, premium, alpha=0.15, color='magenta')
    ax.set_xlabel('t  (years)')
    ax.set_ylabel('Early exercise premium')
    ax.set_title(f'Early exercise premium over time\n'
                 f'(max = {premium.min():.4f} at t=0, min = {premium.max():.4f} at t=T)')
    ax.legend()

    fig.suptitle('Backward induction: option value evolution\n'
                 f'(Put K={K}, S0={S0}, sigma={SIGMA:.0%}, r={R:.0%}, T={T}y)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'=' *64}")
    print("  MONTE CARLO PRICING -- GRAPHICAL ANALYSIS")
    print(f"  S0={S0}  K={K}  r={R:.0%}  sigma={SIGMA:.0%}  T={T}y")
    print(f"{'=' *64}\n")

    figs = [
        ("Figure 1 -- Log-log convergence",          plot_convergence),
        ("Figure 2 -- Brownian distribution",         plot_brownian_distribution),
        ("Figure 3 -- Discounted martingale",         plot_martingale),
        ("Figure 4 -- LS regression at a step",       plot_regression_step),
        ("Figure 5 -- Backward value trace EU vs AM", plot_backward_value_trace),
    ]

    fig_names = [
        'mc_convergence.png',
        'mc_brownian_distribution.png',
        'mc_martingale.png',
        'mc_regression_step.png',
        'mc_backward_trace.png',
    ]
    for (title, func), fname in zip(figs, fig_names):
        print(f"Generating: {title}")
        f = func()
        out = os.path.join(PLOTS_DIR, fname)
        f.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(f)
        print(f"  Saved: {out}")

    print(f"\n{'=' *64}")
    print("  All figures saved to plots/")
    print('=' *64)
