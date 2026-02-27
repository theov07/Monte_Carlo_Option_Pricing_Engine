"""
plot_mc_analysis.py — Visualisations pédagogiques du pricing Monte Carlo.

Exécuter :
    python plot_mc_analysis.py

Figures générées :
  1. Convergence log-log  (SE vs N, pente -0.5)
  2. Distribution des incréments Browniens  (histogramme + QQ-plot)
  3. Martingale discountée  (mean(e^{-rt}·S(t)) vs t, devrait être S₀)
  4. Régression Longstaff-Schwartz à un pas intermédiaire
     (scatter ITM, courbe polynomiale, valeur intrinsèque, borne d'exercice)
  5. Évolution de la valeur d'option pendant le backward induction
     (EU = constante, AM = croît vers t=0 grâce à l'exercice anticipé)
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
from src.brownien import BrownianMotion
from src.regression import Regression, BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  Paramètres
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
    Calcule V_j = mean(cash_flow_j) * exp(-r·j·dt) pour j = 0, …, num_steps.

    V_j représente l'estimation (discountée à t=0) de la valeur de l'option
    à partir du pas j, en suivant la stratégie optimale de j à T.

    Propriétés :
    - Européen : V_j = prix_EU pour tout j  (martingale)
    - Américain : V_j croît de V_T ≈ prix_EU à V_0 = prix_AM
                  (supermartingale : exercice anticipé ajoute de la valeur)
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff_vec(S_paths[:, -1], option)   # à maturité, non discounté
    V = np.empty(num_steps + 1)
    V[num_steps] = np.mean(cash_flow) * np.exp(-r * T)

    reg = Regression(degree=3, basis=BasisType.LAGUERRE) if is_american else None

    for j in range(num_steps - 1, -1, -1):
        if is_american:
            continuation = cash_flow * df
            intrinsic    = _payoff_vec(S_paths[:, j], option)
            cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)
        else:
            # Européen : simplement discount (on conserve toujours le CF futur)
            cash_flow = cash_flow * df

        V[j] = np.mean(cash_flow) * np.exp(-r * j * dt)

    return V


def _regression_snapshot(S_paths: np.ndarray, option: OptionTrade,
                          r: float, T: float, num_steps: int,
                          snap_step: int):
    """
    Retourne les données au pas `snap_step` pendant le backward induction :
    S_itm, continuation_itm, intrinsic_all, cashflow_all_prev.

    On exécute LS backward de T jusqu'à snap_step, puis on capture l'état.
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff_vec(S_paths[:, -1], option)
    reg = Regression(degree=3, basis=BasisType.LAGUERRE)

    for j in range(num_steps - 1, snap_step, -1):
        continuation = cash_flow * df
        intrinsic    = _payoff_vec(S_paths[:, j], option)
        cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)

    # Au pas snap_step : capturer les données avant la décision
    continuation_snap = cash_flow * df
    intrinsic_snap    = _payoff_vec(S_paths[:, snap_step], option)
    S_snap            = S_paths[:, snap_step]

    itm_mask          = intrinsic_snap > 0
    reg_snap          = Regression(degree=3, basis=BasisType.LAGUERRE)
    if itm_mask.sum() > 4:
        reg_snap.fit(S_snap[itm_mask], continuation_snap[itm_mask])

    return S_snap, continuation_snap, intrinsic_snap, itm_mask, reg_snap


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Convergence log-log  (SE vs N)
# ══════════════════════════════════════════════════════════════════════════════
def plot_convergence():
    print("  [1/5] Convergence log-log...")

    n_values = [200, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 80_000, 200_000]
    se_list  = []

    for n in n_values:
        mc = MonteCarloModel(n, MARKET, CALL_EU, PRICING_DATE, seed=0)
        r  = mc.price_european_vectorized(antithetic=False)
        se_list.append(r['std_error'])

    ns  = np.array(n_values, dtype=float)
    ses = np.array(se_list)

    # Régression log-log → pente théorique -0.5
    slope, intercept, rval, _, _ = stats.linregress(np.log(ns), np.log(ses))
    fit_se = np.exp(intercept) * ns ** slope

    # Droite théorique -0.5 calée sur le premier point
    c_theo = ses[0] * ns[0] ** 0.5
    theo   = c_theo / np.sqrt(ns)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ns, ses,  'o-', color='steelblue', lw=2, ms=7, label='SE empirique')
    ax.loglog(ns, fit_se, '--', color='tomato', lw=2,
              label=f'Régression log-log  (pente = {slope:.3f})')
    ax.loglog(ns, theo, ':', color='gray', lw=1.5,
              label='Théorie  1/√N  (pente = −0.5)')

    ax.set_xlabel('N  (nombre de trajectoires)')
    ax.set_ylabel('Erreur standard SE')
    ax.set_title(f'Convergence Monte Carlo — pente log-log ≈ {slope:.3f}\n'
                 f'(attendu −0.5 par le TCL)')
    ax.legend()
    ax.annotate(f'pente = {slope:.3f}', xy=(ns[-1], ses[-1]),
                xytext=(ns[-2]*0.4, ses[-1]*2.5),
                fontsize=10, color='tomato',
                arrowprops=dict(arrowstyle='->', color='tomato'))
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Distribution des incréments Browniens
# ══════════════════════════════════════════════════════════════════════════════
def plot_brownian_distribution():
    print("  [2/5] Distribution des incréments Browniens...")

    N_BM, N_STEPS_BM = 20_000, 100
    dt = T / N_STEPS_BM
    bm = BrownianMotion(N_BM, N_STEPS_BM, T, antithetic=False, seed=42)
    dW = bm.generate_increments_vectorized()          # (N_BM, N_STEPS_BM)
    Z  = (dW / np.sqrt(dt)).flatten()                 # normalisé → N(0,1) théorique

    ks_stat, p_val = stats.kstest(Z, 'norm')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Histogramme + densité théorique ─────────────────────────────────
    ax = axes[0]
    ax.hist(Z, bins=80, density=True, color='steelblue', alpha=0.6,
            edgecolor='white', lw=0.3, label='Z = dW/√dt  (observé)')
    x = np.linspace(-4.5, 4.5, 300)
    ax.plot(x, stats.norm.pdf(x), 'r-', lw=2, label='N(0,1) théorique')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Z = dW / √dt')
    ax.set_ylabel('Densité')
    ax.set_title(f'Incréments Browniens normalisés\n'
                 f'µ={np.mean(Z):.4f}  σ={np.std(Z):.4f}  '
                 f'KS p-val={p_val:.4f}')
    ax.legend()

    # ── QQ-plot ──────────────────────────────────────────────────────────
    ax = axes[1]
    sample = np.random.default_rng(0).choice(Z, size=2_000, replace=False)
    (osm, osr), (slope_qq, intercept_qq, r_qq) = stats.probplot(sample, dist='norm')
    ax.scatter(osm, osr, s=4, alpha=0.4, color='steelblue', label='Quantiles observés')
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope_qq * line_x + intercept_qq, 'r-', lw=2, label='Droite théorique')
    ax.set_xlabel('Quantiles théoriques N(0,1)')
    ax.set_ylabel('Quantiles observés')
    ax.set_title(f'QQ-plot — R² = {r_qq**2:.5f}')
    ax.legend()

    fig.suptitle('Test de normalité des incréments Browniens  (dW ~ N(0, dt))',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Propriété de martingale du spot discounté
# ══════════════════════════════════════════════════════════════════════════════
def plot_martingale():
    print("  [3/5] Martingale discountée...")

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
            label='mean(e^{−rt}·S(t))  (empirique)')
    ax.fill_between(times,
                    disc_means - 2 * disc_stds,
                    disc_means + 2 * disc_stds,
                    alpha=0.2, color='steelblue', label='±2 SE')
    ax.axhline(S0, color='red', ls='--', lw=1.5,
               label=f'S₀ = {S0}  (valeur théorique)')

    max_dev = np.max(np.abs(disc_means - S0))
    ax.set_xlabel('t  (années)')
    ax.set_ylabel('mean(e^{−rt} · S(t))')
    ax.set_title(f'Propriété de martingale du spot discounté\n'
                 f'E^Q[e^{{−rt}}·S(t)] = S₀  —  déviation max = {max_dev:.4f}')
    ax.legend()
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Régression LS à un pas intermédiaire
# ══════════════════════════════════════════════════════════════════════════════
def plot_regression_step():
    print("  [4/5] Régression Longstaff-Schwartz à un pas intermédiaire...")

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

    # Scatter chemins ITM (sous-echantillon pour la lisibilité)
    idx_sub = np.random.default_rng(0).choice(len(S_itm),
                                               size=min(1500, len(S_itm)),
                                               replace=False)
    ax.scatter(S_itm[idx_sub], cont_itm[idx_sub],
               s=8, alpha=0.35, color='steelblue', label='Continuation (paths ITM)')

    # Courbe de régression polynomiale
    S_grid = np.linspace(S_itm.min(), S_itm.max(), 300)
    reg_curve = reg_snap.predict(S_grid)
    ax.plot(S_grid, reg_curve, 'r-', lw=2.5, label='Régression polynomiale (deg 3)')

    # Valeur intrinsèque IV = max(K - S, 0)
    iv_grid = np.maximum(K - S_grid, 0)
    ax.plot(S_grid, iv_grid, 'g--', lw=2, label=f'Valeur intrinsèque max(K−S,0)')

    # Borne d'exercice : là où reg ≈ IV
    exercise_mask = iv_grid > reg_curve
    if exercise_mask.any() and (~exercise_mask).any():
        # Premier indice où on bascule (non-exercice → exercice)
        boundary_idx = np.where(np.diff(exercise_mask.astype(int)) != 0)[0]
        if len(boundary_idx) > 0:
            S_boundary = S_grid[boundary_idx[0]]
            ax.axvline(S_boundary, color='orange', lw=2, ls='-.',
                       label=f'Borne d\'exercice ≈ {S_boundary:.1f}')
            ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 20],
                              S_grid.min(), S_boundary,
                              alpha=0.07, color='orange')
            ax.text(S_boundary - 3, 0.5, 'Exercer', color='darkorange', fontsize=9,
                    ha='right', va='bottom')
            ax.text(S_boundary + 1, 0.5, 'Conserver', color='darkblue', fontsize=9,
                    ha='left', va='bottom')

    ax.axvline(K, color='gray', lw=1, ls=':', alpha=0.6, label=f'Strike K={K}')
    ax.set_xlabel('S(t)  —  Prix du sous-jacent')
    ax.set_ylabel('Valeur (en t=0 dollars)')
    ax.set_title(f'Régression Longstaff-Schwartz au pas j={SNAP}  (t={t_snap:.2f}y)\n'
                 f'N_itm = {itm_mask.sum():,}  —  Base LAGUERRE deg 3')
    ax.legend(loc='upper right')
    ax.set_xlim(S_itm.min() - 2, min(K + 30, S_itm.max() + 2))
    ax.set_ylim(-0.5, None)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — Évolution de la valeur d'option pendant le backward induction
# ══════════════════════════════════════════════════════════════════════════════
def plot_backward_value_trace():
    print("  [5/5] Évolution de la valeur option pendant le backward...")

    NUM_PATHS, NUM_STEPS = 20_000, 100
    bm = BrownianMotion(NUM_PATHS, NUM_STEPS, T, antithetic=False, seed=42)
    S_paths, _ = bm.generate_paths(S0, R, SIGMA, 0.0)

    V_eu = _backward_value_trace(S_paths, PUT_EU, R, T, NUM_STEPS, is_american=False)
    V_am = _backward_value_trace(S_paths, PUT_AM, R, T, NUM_STEPS, is_american=True)

    times = np.arange(NUM_STEPS + 1) * (T / NUM_STEPS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Gauche : les deux courbes ────────────────────────────────────────
    ax = axes[0]
    ax.plot(times, V_eu, 'b-',  lw=2, label='Européen (martingale)')
    ax.plot(times, V_am, 'r-',  lw=2, label='Américain (supermartingale)')
    ax.axhline(BS_PUT, color='blue', lw=1, ls='--', alpha=0.6,
               label=f'BS Put = {BS_PUT:.3f}')
    ax.axhline(V_am[0], color='red', lw=1, ls='--', alpha=0.6,
               label=f'Prix AM MC = {V_am[0]:.3f}')

    ax.set_xlabel('t  (lecture de droite à gauche = backward)')
    ax.set_ylabel('V(t) = mean(CF_t) × e^{−rt}')
    ax.set_title('V_j discountée à t=0 pendant le backward\n'
                 '(EU : plate, AM : croît vers t=0)')
    ax.legend(fontsize=9)

    # ── Droite : prime d'exercice anticipé = V_AM - V_EU ────────────────
    ax = axes[1]
    premium = V_am - V_eu
    ax.plot(times, premium, 'm-', lw=2, label='Prime américaine V_AM − V_EU')
    ax.axhline(0, color='gray', lw=0.8)
    ax.fill_between(times, 0, premium, alpha=0.15, color='magenta')
    ax.set_xlabel('t  (années)')
    ax.set_ylabel('Prime d\'exercice anticipé')
    ax.set_title(f'Prime d\'exercice anticipé au cours du temps\n'
                 f'(max = {premium.min():.4f} à t=0, min = {premium.max():.4f} à t=T)')
    ax.legend()

    fig.suptitle('Backward induction : évolution de la valeur de l\'option\n'
                 f'(Put K={K}, S₀={S0}, σ={SIGMA:.0%}, r={R:.0%}, T={T}y)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'═'*64}")
    print("  ANALYSE GRAPHIQUE DU PRICING MONTE CARLO")
    print(f"  S₀={S0}  K={K}  r={R:.0%}  σ={SIGMA:.0%}  T={T}y")
    print(f"{'═'*64}\n")

    figs = [
        ("Figure 1 — Convergence log-log",          plot_convergence),
        ("Figure 2 — Distribution Brownienne",       plot_brownian_distribution),
        ("Figure 3 — Martingale discountée",         plot_martingale),
        ("Figure 4 — Régression LS à un step",       plot_regression_step),
        ("Figure 5 — Backward value trace EU vs AM", plot_backward_value_trace),
    ]

    fig_names = [
        'mc_convergence.png',
        'mc_brownian_distribution.png',
        'mc_martingale.png',
        'mc_regression_step.png',
        'mc_backward_trace.png',
    ]
    for (title, func), fname in zip(figs, fig_names):
        print(f"Génération : {title}")
        f = func()
        out = os.path.join(PLOTS_DIR, fname)
        f.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(f)
        print(f"  ✓ Sauvegardé : {out}")

    print(f"\n{'═'*64}")
    print("  Toutes les figures sauvegardées dans plots/")
    print("═"*64)
