"""
verify_mc.py — Tests de vérification systématiques du pricing Monte Carlo.

Exécuter :
    python verify_mc.py

Tests implémentés (dans l'ordre des notes de cours) :
  1. Prix européen scalaire (call + put) vs Black-Scholes          → IC 3σ
  2. Convergence  : pente log-log de SE vs N  ≈ -0.5              → ±0.1
  3. Test de vraisemblance / normalité des incréments Browniens    → KS p > 0.05
  4. Martingale : mean(e^{-r·t}·S(t)) ≈ S₀ pour tout t            → < 0.5 %
  5. Parité Put-Call européenne                                     → IC 3σ
  6. Reproductibilité inter-seeds : std_seeds ≈ SE théorique       → ratio [0.7 ; 1.3]
  7. Prix américain put > prix européen put                         → strict
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

import numpy as np
from scipy import stats
from datetime import date

from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src.black_scholes import BlackScholes
from src.brownien import BrownianMotion
from src.regression import BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  Paramètres communs
# ══════════════════════════════════════════════════════════════════════════════
PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)   # T = 365/365 = 1.0 exact (2025 non bissextile)
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20
T = 1.0

MARKET   = Market(S0, SIGMA, R, 0.0, None)
CALL_EU  = OptionTrade(mat=MAT_DATE, call_put='CALL', ex='EUROPEAN', k=K)
PUT_EU   = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='EUROPEAN', k=K)
PUT_AM   = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='AMERICAN', k=K)

BS_CALL  = BlackScholes(MARKET, CALL_EU, PRICING_DATE).price()
BS_PUT   = BlackScholes(MARKET, PUT_EU,  PRICING_DATE).price()

# ══════════════════════════════════════════════════════════════════════════════
#  Infrastructure d'affichage
# ══════════════════════════════════════════════════════════════════════════════
_PASS = 0
_FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [OK]  {name}")
    else:
        _FAIL += 1
        print(f"  [!!]  {name}  ← ÉCHEC")
    if detail:
        print(f"        {detail}")


def section(title: str) -> None:
    bar = "─" * 64
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 — Prix européen scalaire (boucle path par path) vs Black-Scholes
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 1 : Prix Européen Scalaire vs Black-Scholes")
print(f"  Références BS  →  Call={BS_CALL:.4f}   Put={BS_PUT:.4f}")
print(f"  (S={S0}, K={K}, r={R:.0%}, σ={SIGMA:.0%}, T={T}y)")

N_SCALAR = 20_000
mc_c = MonteCarloModel(N_SCALAR, MARKET, CALL_EU, PRICING_DATE, seed=42)
mc_p = MonteCarloModel(N_SCALAR, MARKET, PUT_EU,  PRICING_DATE, seed=42)

rc = mc_c.price_european(antithetic=True)
rp = mc_p.price_european(antithetic=True)

print(f"\n  MC Call (N={N_SCALAR:,}) = {rc['price']:.4f}  SE={rc['std_error']:.4f}")
print(f"  MC Put  (N={N_SCALAR:,}) = {rp['price']:.4f}  SE={rp['std_error']:.4f}")

TOL = 3.0   # nombre d'écarts-types

check(
    "Call européen scalaire dans IC à 3σ",
    abs(rc['price'] - BS_CALL) < TOL * rc['std_error'],
    f"|{rc['price']:.4f} - {BS_CALL:.4f}| = {abs(rc['price']-BS_CALL):.4f} "
    f"vs 3σ = {TOL*rc['std_error']:.4f}"
)
check(
    "Put européen scalaire dans IC à 3σ",
    abs(rp['price'] - BS_PUT) < TOL * rp['std_error'],
    f"|{rp['price']:.4f} - {BS_PUT:.4f}| = {abs(rp['price']-BS_PUT):.4f} "
    f"vs 3σ = {TOL*rp['std_error']:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 — Convergence : pente log-log de SE vs N ≈ -0.5
#  Théorie : SE = σ_payoff / √N  →  log(SE) = cst - 0.5·log(N)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 2 : Convergence  log(SE) vs log(N)  — pente attendue = -0.5")

n_values = [500, 1_000, 2_000, 5_000, 10_000, 30_000, 80_000]
se_list  = []

print(f"\n  {'N':>8}   {'SE':>10}   {'log(N)':>8}   {'log(SE)':>10}")
print(f"  {'─'*46}")

for n in n_values:
    mc = MonteCarloModel(n, MARKET, CALL_EU, PRICING_DATE, seed=0)
    r  = mc.price_european_vectorized(antithetic=False)
    se = r['std_error']
    se_list.append(se)
    print(f"  {n:>8,}   {se:>10.5f}   {np.log(n):>8.3f}   {np.log(se):>10.4f}")

# Régression linéaire sur (log N, log SE)
log_n  = np.log(n_values)
log_se = np.log(se_list)
slope, intercept, r_value, _, _ = stats.linregress(log_n, log_se)

print(f"\n  Pente estimée = {slope:.4f}   (attendu ≈ -0.5)   R² = {r_value**2:.5f}")

check(
    "Pente log(SE)/log(N) ∈ [-0.6 ; -0.4]",
    -0.6 <= slope <= -0.4,
    f"Pente = {slope:.4f}"
)
check(
    "R² de la droite ≥ 0.99  (bon ajustement linéaire)",
    r_value**2 >= 0.99,
    f"R² = {r_value**2:.5f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 — Normalité des incréments Browniens
#  Théorie : dW ~ N(0, dt)  →  Z = dW / √dt ~ N(0, 1)
#  Test de Kolmogorov-Smirnov : H₀ = distribution normale
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 3 : Normalité des incréments Browniens  (KS test)")

N_BM  = 50_000
DT    = T / 100        # 100 pas de temps
bm    = BrownianMotion(N_BM, 100, T, antithetic=False, seed=123)
dW    = bm.generate_increments_vectorized()   # shape (N_BM, 100)
Z_all = (dW / np.sqrt(DT)).flatten()          # N_BM×100 réalisations de N(0,1)

ks_stat, p_value = stats.kstest(Z_all, 'norm')

print(f"\n  N incréments = {len(Z_all):,}")
print(f"  Moyenne      = {np.mean(Z_all):+.6f}   (attendu 0)")
print(f"  Écart-type   = {np.std(Z_all):.6f}    (attendu 1)")
print(f"  KS statistic = {ks_stat:.6f}")
print(f"  KS p-value   = {p_value:.4f}   (seuil α = 0.01)")

check(
    "Moyenne des Z ≈ 0  (|µ| < 0.005)",
    abs(np.mean(Z_all)) < 0.005,
    f"µ = {np.mean(Z_all):+.6f}"
)
check(
    "Écart-type des Z ≈ 1  (|σ-1| < 0.01)",
    abs(np.std(Z_all) - 1) < 0.01,
    f"σ = {np.std(Z_all):.6f}"
)
check(
    "KS test normalité p-value ≥ 0.01",
    p_value >= 0.01,
    f"p = {p_value:.4f}"
)

# Vérification par skewness et kurtosis
skew = stats.skew(Z_all)
kurt = stats.kurtosis(Z_all)   # excès de kurtosis, doit être ≈ 0
print(f"\n  Skewness     = {skew:+.4f}   (attendu 0)")
print(f"  Kurtosis exc = {kurt:+.4f}   (attendu 0)")
check(
    "Skewness ≈ 0  (|skew| < 0.05)",
    abs(skew) < 0.05,
    f"Skewness = {skew:+.4f}"
)
check(
    "Kurtosis excès ≈ 0  (|kurt| < 0.05)",
    abs(kurt) < 0.05,
    f"Kurtosis = {kurt:+.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 — Martingale : mean(e^{-r·t_j}·S(t_j)) ≈ S₀ pour tout j
#  Propriété fondamentale du GBM sous la mesure risque-neutre Q :
#    E^Q[e^{-r·t}·S(t)] = S₀  (processus discounté = martingale)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 4 : Martingale — mean(e^{-rt}·S(t)) = S₀ pour tout t")

N_MART  = 30_000
N_STEPS = 50
bm_m = BrownianMotion(N_MART, N_STEPS, T, antithetic=False, seed=7)
S_paths, _ = bm_m.generate_paths(S0, R, SIGMA, 0.0)

dt_m   = T / N_STEPS
errors = []

print(f"\n  {'Pas j':>6}   {'t_j':>6}   {'mean(e^{{-rt}}·S)':>18}   {'Erreur rel.':>12}")
print(f"  {'─'*50}")

# Afficher quelques valeurs régulièrement espacées
display_steps = list(range(0, N_STEPS + 1, N_STEPS // 5))
for j in range(N_STEPS + 1):
    t_j   = j * dt_m
    disc  = np.mean(np.exp(-R * t_j) * S_paths[:, j])
    err   = abs(disc - S0) / S0
    errors.append(err)
    if j in display_steps:
        print(f"  {j:>6}   {t_j:>6.3f}   {disc:>18.4f}   {err:>11.3%}")

max_err = max(errors)
print(f"\n  Erreur relative max (sur tous les pas) = {max_err:.4%}")

check(
    "Martingale : erreur relative max < 0.5 %",
    max_err < 0.005,
    f"Max erreur = {max_err:.4%}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 — Parité Put-Call européenne
#  C - P = S₀ - K·e^{-rT}  (sans dividende)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 5 : Parité Put-Call  C − P = S₀ − K·e^{−rT}")

import math
theoretical = S0 - K * math.exp(-R * T)

N_PAR = 50_000
mc_cv = MonteCarloModel(N_PAR, MARKET, CALL_EU, PRICING_DATE, seed=99)
mc_pv = MonteCarloModel(N_PAR, MARKET, PUT_EU,  PRICING_DATE, seed=99)

rv_c = mc_cv.price_european_vectorized(antithetic=True)
rv_p = mc_pv.price_european_vectorized(antithetic=True)

cp_diff = rv_c['price'] - rv_p['price']
# Erreur sur C-P ≈ sqrt(SE_C² + SE_P²)  (si indépendants)
se_diff = np.sqrt(rv_c['std_error']**2 + rv_p['std_error']**2)

print(f"\n  BS  C − P = {BS_CALL - BS_PUT:.4f}   (analytique exact)")
print(f"  MC  C     = {rv_c['price']:.4f}   SE={rv_c['std_error']:.4f}")
print(f"  MC  P     = {rv_p['price']:.4f}   SE={rv_p['std_error']:.4f}")
print(f"  MC  C − P = {cp_diff:.4f}   SE_diff={se_diff:.4f}")
print(f"  Théorique = {theoretical:.4f}")
print(f"  Écart     = {abs(cp_diff - theoretical):.4f}  vs  3σ = {3*se_diff:.4f}")

check(
    "Parité Put-Call dans IC à 3σ",
    abs(cp_diff - theoretical) < TOL * se_diff,
    f"|C-P - théo| = {abs(cp_diff - theoretical):.4f} < {TOL}×{se_diff:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 — Reproductibilité inter-seeds
#  Pour N fixé, les estimateurs avec graines différentes doivent avoir
#  std(prix_seeds) ≈ SE_théorique = std(payoffs) / √N
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 6 : Reproductibilité inter-seeds — std_seeds ≈ SE théorique")

N_SEED     = 10_000   # N pour chaque run
N_REPEATS  = 50       # nombre de seeds différentes

prices_seeds = []
se_single    = None

for seed in range(N_REPEATS):
    mc = MonteCarloModel(N_SEED, MARKET, CALL_EU, PRICING_DATE, seed=seed)
    r  = mc.price_european_vectorized(antithetic=True)
    prices_seeds.append(r['price'])
    if se_single is None:
        se_single = r['std_error']

std_seeds = np.std(prices_seeds, ddof=1)
ratio     = std_seeds / se_single

print(f"\n  N = {N_SEED:,}   répétitions = {N_REPEATS}")
print(f"  Moyenne des estimateurs  = {np.mean(prices_seeds):.4f}   (BS = {BS_CALL:.4f})")
print(f"  std inter-seeds          = {std_seeds:.4f}")
print(f"  SE théorique (1 run)     = {se_single:.4f}")
print(f"  Ratio std_seeds / SE     = {ratio:.3f}   (attendu ≈ 1)")

check(
    "BS dans l'intervalle [mean ± 2σ] des estimateurs",
    abs(np.mean(prices_seeds) - BS_CALL) < 2 * std_seeds,
    f"|{np.mean(prices_seeds):.4f} - {BS_CALL:.4f}| vs 2×{std_seeds:.4f}"
)
check(
    "Ratio std_seeds / SE ∈ [0.7 ; 1.3]",
    0.7 <= ratio <= 1.3,
    f"Ratio = {ratio:.3f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 — Prix américain > Prix européen  (put avec dividende nul)
#  Pour un put sans dividende, l'américain vaut toujours plus que l'européen.
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 7 : Prix Américain Put ≥ Prix Européen Put")

N_AM = 50_000
mc_am = MonteCarloModel(N_AM, MARKET, PUT_AM, PRICING_DATE, seed=42)
r_am  = mc_am.price_american_longstaff_schwartz_vectorized(
    num_steps=100, poly_degree=3, poly_basis=BasisType.LAGUERRE,
    antithetic=True
)

mc_eu = MonteCarloModel(N_AM, MARKET, PUT_EU, PRICING_DATE, seed=42)
r_eu  = mc_eu.price_european_vectorized(antithetic=True)

print(f"\n  Put américain (LS)     = {r_am['price']:.4f}   SE={r_am['std_error']:.4f}")
print(f"  Put européen  (MC)     = {r_eu['price']:.4f}   SE={r_eu['std_error']:.4f}")
print(f"  Put européen  (BS)     = {BS_PUT:.4f}")
print(f"  Prime américaine MC    = {r_am['price'] - r_eu['price']:+.4f}")

check(
    "Prix américain ≥ Prix européen (MC)",
    r_am['price'] >= r_eu['price'],
    f"{r_am['price']:.4f} >= {r_eu['price']:.4f}"
)
check(
    "Prix américain ≥ BS Put (analytique)",
    r_am['price'] >= BS_PUT - 2 * r_am['std_error'],
    f"{r_am['price']:.4f} >= {BS_PUT:.4f} - 2×{r_am['std_error']:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  BILAN
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{'═'*64}")
print(f"  BILAN : {_PASS}/{total} tests passés", end="")
if _FAIL == 0:
    print("  ✓  TOUS LES TESTS SONT VERTS")
else:
    print(f"  — {_FAIL} test(s) en échec")
print("═"*64)

# ── Graphique bilan ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(['Passed', 'Failed'], [_PASS, _FAIL],
              color=['#4caf50', '#f44336'], edgecolor='white', width=0.5)
for bar, val in zip(bars, [_PASS, _FAIL]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_ylim(0, max(_PASS + _FAIL, 1) * 1.25)
ax.set_ylabel('Nombre de tests')
ax.set_title(f'V\u00e9rification MC — bilan: {_PASS}/{_PASS+_FAIL} tests OK')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'verify_mc_summary.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"\n\u2713 Graphique sauvegard\u00e9 : {out}")