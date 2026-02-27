"""
test_mc_battery.py — Batterie de tests approfondis pour le pricing Monte Carlo.

Exécuter :
    python test_mc_battery.py

Différence avec verify_mc.py (tests de bout-en-bout) :
  → Ici on teste les COMPOSANTS algorithmiques isolément et on stresse
    les propriétés statistiques avec plus de rigueur.

5 suites :
  Suite 1 — exercise_decision  (isolé du backward loop)
  Suite 2 — Colonne discountée  (constante EU / non-constante AM)
  Suite 3 — Stabilité inter-seeds  (std, z-scores, déterminisme, couverture)
  Suite 4 — Réduction de variance antithétique
  Suite 5 — Propriétés économiques  (vega, monotonie, bornes, AM call)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

import math
import numpy as np
from scipy import stats
from scipy.stats import norm as _norm
from datetime import date

from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src.black_scholes import BlackScholes
from src.brownien import BrownianMotion
from src.regression import Regression, BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  Paramètres communs
# ══════════════════════════════════════════════════════════════════════════════
PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)   # T = 1 an exact (2025 non bissextile)
S0, K, R, SIGMA = 100.0, 100.0, 0.05, 0.20
T = 1.0

MARKET  = Market(S0, SIGMA, R, 0.0, None)
CALL_EU = OptionTrade(mat=MAT_DATE, call_put='CALL', ex='EUROPEAN', k=K)
PUT_EU  = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='EUROPEAN', k=K)
PUT_AM  = OptionTrade(mat=MAT_DATE, call_put='PUT',  ex='AMERICAN', k=K)
CALL_AM = OptionTrade(mat=MAT_DATE, call_put='CALL', ex='AMERICAN', k=K)

BS_CALL = BlackScholes(MARKET, CALL_EU, PRICING_DATE).price()
BS_PUT  = BlackScholes(MARKET, PUT_EU,  PRICING_DATE).price()

# ══════════════════════════════════════════════════════════════════════════════
#  Infrastructure d'affichage
# ══════════════════════════════════════════════════════════════════════════════
_PASS = _FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [OK]  {name}")
    else:
        _FAIL += 1
        print(f"  [!!]  {name}  ← ÉCHEC")
    if detail:
        print(f"        {detail}")
    return condition

def section(title: str) -> None:
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print("─" * 66)

def _payoff(S: np.ndarray, option: OptionTrade) -> np.ndarray:
    if option.is_a_call():
        return np.maximum(S - option.strike, 0)
    return np.maximum(option.strike - S, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper : backward induction pas à pas avec capture de la colonne discountée
#
#  À chaque step j, on capture :
#    V[j] = mean(cash_flow_j) × e^{−r·j·dt}
#
#  Interprétation :
#    cash_flow_j = la meilleure valeur atteignable depuis j (valeur à t=j)
#    × e^{−r·j·dt} = discount de j vers t=0
#
#  Propriétés théoriques :
#    Européen : V[j] = constante = prix_EU  (processus discounté = martingale)
#    Américain : V[j] CROÎT quand j → 0  (supermartingale, exercice anticipé)
# ══════════════════════════════════════════════════════════════════════════════

def _backward_column_means(S_paths: np.ndarray,
                            option: OptionTrade,
                            r: float, T: float, num_steps: int,
                            is_american: bool) -> np.ndarray:
    """
    Retourne V[j] = mean(cash_flow_j) * e^{-r*j*dt} pour j = 0, …, num_steps.
    On garde UNE SEULE colonne cash_flow en mémoire à chaque pas (pas de matrice).
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff(S_paths[:, -1], option)        # valeur à maturité T
    V = np.empty(num_steps + 1)
    V[num_steps] = float(np.mean(cash_flow)) * np.exp(-r * T)

    reg = Regression(degree=3, basis=BasisType.LAGUERRE) if is_american else None

    for j in range(num_steps - 1, -1, -1):
        if is_american:
            # Décision d'exercice : garde la même colonne cash_flow
            continuation = cash_flow * df
            intrinsic    = _payoff(S_paths[:, j], option)
            cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)
        else:
            # Européen : discount d'un pas, aucune décision d'exercice
            cash_flow = cash_flow * df

        # Discount de t_j → t=0 : on obtient la valeur "en monnaie d'aujourd'hui"
        V[j] = float(np.mean(cash_flow)) * np.exp(-r * j * dt)

    return V


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 1 — exercise_decision  (isolé du backward loop)
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 1 : exercise_decision — tests unitaires isolés")
print(f"  (Put K={K}, tests sur des données synthétiques de contrôle)")

rng = np.random.default_rng(42)
N = 3000

# ── 1.1 Deep ITM, continuation = 0 → on DOIT exercer sur toutes les paths ITM ──
S_itm   = rng.uniform(50, 85, N)           # all deep ITM for put K=100
iv_itm  = np.maximum(K - S_itm, 0)        # intrinsic > 0 everywhere
cont_0  = np.zeros(N)

reg_1 = Regression(degree=3, basis=BasisType.POWER)
res_11 = reg_1.exercise_decision(S_itm, iv_itm, cont_0)

check(
    "1.1  Deep ITM + continuation=0 → cash_flow = intrinsic",
    np.allclose(res_11, iv_itm, atol=1e-9),
    f"max|result − intrinsic| = {np.max(np.abs(res_11 - iv_itm)):.2e}"
)

# ── 1.2 OTM paths : intrinsic = 0, continuation > 0 → GARDER la continuation ──
S_otm   = rng.uniform(110, 150, N)         # all OTM for put K=100
iv_otm  = np.maximum(K - S_otm, 0)        # = 0 everywhere
cont_pos = rng.uniform(1, 10, N)

reg_2 = Regression(degree=3, basis=BasisType.POWER)
res_12 = reg_2.exercise_decision(S_otm, iv_otm, cont_pos)

check(
    "1.2  OTM (intrinsic=0) → cash_flow = continuation exactement",
    np.allclose(res_12, cont_pos, atol=1e-9),
    f"max|result − cont| = {np.max(np.abs(res_12 - cont_pos)):.2e}"
)

# ── 1.3 Continuation >> intrinsic → NE PAS exercer ──────────────────────────
S_mix     = rng.uniform(60, 100, N)
iv_small  = np.maximum(K - S_mix, 0)       # ≤ 40
cont_big  = np.full(N, 200.0)              # écrase toute valeur intrinsèque

reg_3 = Regression(degree=3, basis=BasisType.POWER)
res_13 = reg_3.exercise_decision(S_mix, iv_small, cont_big)

# Pour les paths ITM, la régression doit prédire ≈ 200 >> intrinsic → continuation
itm_mask_3 = iv_small > 0
check(
    "1.3  Continuation >> intrinsic → aucun cash_flow > continuation",
    np.all(res_13 <= cont_big + 1e-6),
    f"max(res) = {res_13.max():.4f}  max(cont) = {cont_big.max():.4f}"
)

# ── 1.4 Intrinsic >> continuation → TOUJOURS exercer ────────────────────────
S_xtm    = rng.uniform(50, 85, N)
iv_big   = np.maximum(K - S_xtm, 0)        # ∈ [15, 50]
cont_tiny = np.full(N, 1e-4)

reg_4 = Regression(degree=3, basis=BasisType.POWER)
res_14 = reg_4.exercise_decision(S_xtm, iv_big, cont_tiny)

itm_mask_4 = iv_big > 0
check(
    "1.4  Intrinsic >> continuation → cash_flow = intrinsic sur paths ITM",
    np.allclose(res_14[itm_mask_4], iv_big[itm_mask_4], atol=1e-6),
    f"max|result − intrinsic| ITM = {np.max(np.abs(res_14[itm_mask_4] - iv_big[itm_mask_4])):.2e}"
)

# ── 1.5 Qualité du fit : données synthétiques avec solution analytique connue ──
N5    = 8000
S5    = rng.uniform(70, 100, N5)
# "Vraie" continuation linéaire en S (connue)
a, b  = -0.8, 85.0
y_true = a * S5 + b                         # valeurs entre 5 et 29
y_obs  = y_true + rng.normal(0, 0.3, N5)   # bruit faible

reg_5 = Regression(degree=1, basis=BasisType.POWER)   # deg 1 suffit ici
reg_5.fit(S5, y_obs)
y_pred = reg_5.predict(S5)
r2 = np.corrcoef(y_pred, y_true)[0, 1] ** 2

check(
    "1.5  Fit linéaire (deg 1) sur données synthétiques : R²(pred, vrai) ≥ 0.999",
    r2 >= 0.999,
    f"R² = {r2:.6f}"
)

# ── 1.6 residual_std est bien calculé et cohérent avec les résidus réels ────
reg_6 = Regression(degree=3, basis=BasisType.POWER)
reg_6.fit(S5, y_obs)
residuals = y_obs - reg_6._design_matrix(S5) @ reg_6._coeffs
std_residuals_manual = float(np.std(residuals))
check(
    "1.6  _residual_std cohérent avec les résidus calculés manuellement",
    abs(reg_6._residual_std - std_residuals_manual) < 1e-10,
    f"_residual_std={reg_6._residual_std:.6f}  manuel={std_residuals_manual:.6f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 2 — Colonne discountée : constante (EU) vs croissante (AM)
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 2 : Colonne discountée  V[j] = mean(CF_j) × e^{−rjdt}")
print()
print("  Théorie :")
print("    Européen  → CF_j = payoff(S_T) × df^{T−j}  ⟹  V[j] EXACTEMENT constant")
print("    Américain → exercice anticipé enrichit CF_j  ⟹  V[j] croît quand j→0")

NUM_PATHS_COL = 40_000
NUM_STEPS_COL = 60

bm_col = BrownianMotion(NUM_PATHS_COL, NUM_STEPS_COL, T, antithetic=False, seed=99)
S_col, _ = bm_col.generate_paths(S0, R, SIGMA, 0.0)

print(f"\n  Simulation : {NUM_PATHS_COL:,} paths × {NUM_STEPS_COL} steps")

# ── 2.1 Européen : V[j] constant (identité algébrique exacte) ───────────────
V_eu = _backward_column_means(S_col, PUT_EU, R, T, NUM_STEPS_COL, is_american=False)

V_eu_range = float(V_eu.max() - V_eu.min())
V_eu_mean  = float(np.mean(V_eu))
cv_eu      = V_eu_range / V_eu_mean          # doit être < 1e-8 (flottant)

print(f"\n  EU Put  V_mean={V_eu_mean:.6f}  V_range={V_eu_range:.2e}  CV={cv_eu:.2e}")
print(f"  BS Put  = {BS_PUT:.6f}")

check(
    "2.1  EU : V[j] algébriquement constant — range/mean < 1e-8",
    cv_eu < 1e-8,
    f"range/mean = {cv_eu:.2e}  (flottant ≈ 0)"
)
check(
    "2.2  EU : V_mean proche du prix BS (erreur < 1.5%)",
    abs(V_eu_mean - BS_PUT) / BS_PUT < 0.015,
    f"V_mean={V_eu_mean:.4f}  BS={BS_PUT:.4f}  err={abs(V_eu_mean-BS_PUT)/BS_PUT:.3%}"
)

# ── 2.2 Américain : V[j] croît de V[T] → V[0] ──────────────────────────────
V_am = _backward_column_means(S_col, PUT_AM, R, T, NUM_STEPS_COL, is_american=True)

premium = V_am[0] - V_am[-1]
print(f"\n  AM Put  V[0]={V_am[0]:.4f}  V[T]={V_am[-1]:.4f}  ΔV = {premium:+.4f}")

check(
    "2.3  AM : V[0] > V[T]  (exercice anticipé augmente la valeur)",
    V_am[0] > V_am[-1],
    f"V_am[0]={V_am[0]:.4f}  V_am[T]={V_am[-1]:.4f}"
)
check(
    "2.4  AM : V[0] > V_EU[0]  (américain vaut plus que l'européen)",
    V_am[0] > V_eu[0],
    f"V_am[0]={V_am[0]:.4f}  V_eu[0]={V_eu[0]:.4f}"
)
check(
    "2.5  AM : prime d'exercice anticipé > 0.1  (non triviale)",
    premium > 0.1,
    f"Prime = {premium:.4f}"
)

# ── 2.3 Tendance globale de V_am : décroissante en j (croissante en remontant) ──
j_arr = np.arange(NUM_STEPS_COL + 1, dtype=float)
slope_am, _, r_am, _, _ = stats.linregress(j_arr, V_am)
check(
    "2.6  AM : pente de V_am en fonction de j est NÉGATIVE (croît en backward)",
    slope_am < 0,
    f"Pente = {slope_am:.6f}  R={r_am:.3f}"
)

# ── 2.4 EU vs AM : même V[T] (payoff terminal identique, même paths) ────────
check(
    "2.7  EU et AM ont le même V[T]  (même payoff terminal, mêmes paths)",
    abs(V_am[-1] - V_eu[-1]) < 1e-8,
    f"|V_am[T] − V_eu[T]| = {abs(V_am[-1] - V_eu[-1]):.2e}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 3 — Stabilité inter-seeds : std, z-scores, déterminisme, couverture
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 3 : Stabilité inter-seeds — std, z-scores, déterminisme")

N_RUN    = 8_000    # paths par run
N_SEEDS  = 80       # nombre de seeds différentes

prices = np.empty(N_SEEDS)
ses    = np.empty(N_SEEDS)

for seed in range(N_SEEDS):
    mc = MonteCarloModel(N_RUN, MARKET, CALL_EU, PRICING_DATE, seed=seed)
    r  = mc.price_european_vectorized(antithetic=True)
    prices[seed] = r['price']
    ses[seed]    = r['std_error']

std_prices = float(np.std(prices, ddof=1))
mean_se    = float(np.mean(ses))
ratio      = std_prices / mean_se

print(f"\n  N={N_RUN:,} paths × {N_SEEDS} seeds")
print(f"  mean(price)     = {np.mean(prices):.4f}   BS = {BS_CALL:.4f}")
print(f"  std(prices)     = {std_prices:.4f}   ← dispersion inter-seeds")
print(f"  mean(SE)        = {mean_se:.4f}   ← SE moyen prédit par 1 run")
print(f"  Ratio std/SE    = {ratio:.3f}   (attendu ≈ 1.0)")

# ── 3.1 std inter-seeds ≈ SE théorique ──────────────────────────────────────
check(
    "3.1  std(prices_seeds) / mean(SE) ∈ [0.7 ; 1.3]",
    0.7 <= ratio <= 1.3,
    f"Ratio = {ratio:.3f}"
)

# ── 3.2 Biais global < 0.5 % ────────────────────────────────────────────────
bias = abs(np.mean(prices) - BS_CALL) / BS_CALL
check(
    "3.2  Biais relatif de la moyenne < 0.5 %",
    bias < 0.005,
    f"Biais = {bias:.4%}   mean={np.mean(prices):.4f}  BS={BS_CALL:.4f}"
)

# ── 3.3 z-scores = (price_i − BS) / SE_i ~ N(0,1) ──────────────────────────
z_scores = (prices - BS_CALL) / ses

ks_stat, p_ks = stats.kstest(z_scores, 'norm')
print(f"\n  z-scores :  mean={np.mean(z_scores):+.3f}  std={np.std(z_scores):.3f}")
print(f"  KS stat={ks_stat:.4f}   p-value={p_ks:.4f}   (H₀ : ~ N(0,1))")

check(
    "3.3  z-scores ~ N(0,1) : KS p-value ≥ 0.05",
    p_ks >= 0.05,
    f"KS p = {p_ks:.4f}"
)
check(
    "3.4  |mean(z_scores)| < 0.3  (pas de biais systématique)",
    abs(np.mean(z_scores)) < 0.3,
    f"mean(z) = {np.mean(z_scores):+.3f}"
)
check(
    "3.5  std(z_scores) ∈ [0.7 ; 1.3]  (SE bien calibré)",
    0.7 <= np.std(z_scores) <= 1.3,
    f"std(z) = {np.std(z_scores):.3f}"
)

# ── 3.4 Couverture empirique des IC 95 % ────────────────────────────────────
z95       = _norm.ppf(0.975)
in_ic     = int(np.sum(np.abs(prices - BS_CALL) <= z95 * ses))
coverage  = in_ic / N_SEEDS
print(f"\n  Couverture IC 95 % : {in_ic}/{N_SEEDS} = {coverage:.1%}  (attendu ≈ 95 %)")

check(
    "3.6  Couverture IC 95 % ∈ [80 % ; 100 %]",
    0.80 <= coverage <= 1.00,
    f"Couverture = {coverage:.1%}"
)

# ── 3.5 Déterminisme : même seed → prix IDENTIQUE à la précision machine ────
mc_a = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=314)
mc_b = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=314)
pa = mc_a.price_european_vectorized()['price']
pb = mc_b.price_european_vectorized()['price']

check(
    "3.7  Déterminisme : seed=314 exécuté 2× → prix identique à 1e-12 près",
    abs(pa - pb) < 1e-12,
    f"|price_a − price_b| = {abs(pa - pb):.2e}"
)

# ── 3.6 Seeds différentes → prix différents (stochasticité réelle) ──────────
mc_s0 = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=0)
mc_s1 = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=1)
p0 = mc_s0.price_european_vectorized()['price']
p1 = mc_s1.price_european_vectorized()['price']

check(
    "3.8  Seeds différentes → prix différents  (seed=0 ≠ seed=1)",
    abs(p0 - p1) > 1e-6,
    f"price(seed=0)={p0:.6f}  price(seed=1)={p1:.6f}  diff={abs(p0-p1):.2e}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 4 — Réduction de variance par variables antithétiques
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 4 : Réduction de variance — antithétique vs brut")

# Même nombre de random draws au total : num_simulations fixé
N_VAR = 40_000

mc_anti  = MonteCarloModel(N_VAR, MARKET, CALL_EU, PRICING_DATE, seed=0)
mc_brute = MonteCarloModel(N_VAR, MARKET, CALL_EU, PRICING_DATE, seed=0)
r_anti   = mc_anti.price_european_vectorized(antithetic=True)
r_brute  = mc_brute.price_european_vectorized(antithetic=False)

se_anti  = r_anti['std_error']
se_brute = r_brute['std_error']
ratio_se = se_anti / se_brute

print(f"\n  N={N_VAR:,} paths  (antithétique : N/2 paires indépendantes)")
print(f"  SE antithétique = {se_anti:.5f}")
print(f"  SE brute        = {se_brute:.5f}")
print(f"  Ratio SE_anti / SE_brute = {ratio_se:.3f}   (attendu < 1)")

check(
    "4.1  SE antithétique < SE brute (réduction de variance confirmée)",
    se_anti < se_brute,
    f"{se_anti:.5f} < {se_brute:.5f}"
)
check(
    "4.2  Ratio SE_anti / SE_brute < 0.90  (réduction ≥ 10 %)",
    ratio_se < 0.90,
    f"Ratio = {ratio_se:.3f}"
)

# Vérification sur le put aussi (corrélation négative souvent plus forte)
mc_anti_put  = MonteCarloModel(N_VAR, MARKET, PUT_EU, PRICING_DATE, seed=0)
mc_brute_put = MonteCarloModel(N_VAR, MARKET, PUT_EU, PRICING_DATE, seed=0)
r_ap = mc_anti_put.price_european_vectorized(antithetic=True)
r_bp = mc_brute_put.price_european_vectorized(antithetic=False)
check(
    "4.3  SE antithétique < SE brute — PUT aussi",
    r_ap['std_error'] < r_bp['std_error'],
    f"SE_anti={r_ap['std_error']:.5f}  SE_brute={r_bp['std_error']:.5f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 5 — Propriétés économiques
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 5 : Propriétés économiques du pricing MC")

N_PROP = 40_000

# ── 5.1 Time value strictement positive pour option ATM ─────────────────────
mc_p_eu = MonteCarloModel(N_PROP, MARKET, PUT_EU, PRICING_DATE, seed=10)
r_p_eu  = mc_p_eu.price_european_vectorized(antithetic=True)
iv_atm  = max(K - S0, 0)   # = 0 pour put ATM

check(
    "5.1  Put EU ATM : prix > valeur intrinsèque (time value > 0)",
    r_p_eu['price'] > iv_atm,
    f"Prix={r_p_eu['price']:.4f}  IV={iv_atm:.4f}"
)

# ── 5.2 Borne inférieure : C ≥ max(S − K·e^{-rT}, 0) ───────────────────────
lb_call = max(S0 - K * math.exp(-R * T), 0)
lb_put  = max(K * math.exp(-R * T) - S0, 0)
tol_3se = 3.0

mc_c_lb = MonteCarloModel(N_PROP, MARKET, CALL_EU, PRICING_DATE, seed=11)
mc_p_lb = MonteCarloModel(N_PROP, MARKET, PUT_EU,  PRICING_DATE, seed=11)
rc_lb   = mc_c_lb.price_european_vectorized(antithetic=True)
rp_lb   = mc_p_lb.price_european_vectorized(antithetic=True)

check(
    f"5.2  Call EU ≥ lb = max(S−K·e^{{-rT}},0) = {lb_call:.4f}  (à 3σ)",
    rc_lb['price'] >= lb_call - tol_3se * rc_lb['std_error'],
    f"Prix={rc_lb['price']:.4f}  LB={lb_call:.4f}"
)
check(
    f"5.3  Put  EU ≥ lb = max(K·e^{{-rT}}−S,0) = {lb_put:.4f}  (à 3σ)",
    rp_lb['price'] >= lb_put - tol_3se * rp_lb['std_error'],
    f"Prix={rp_lb['price']:.4f}  LB={lb_put:.4f}"
)

# ── 5.3 Monotonie en S₀ : Call monte, Put descend ───────────────────────────
delta_S    = 5.0
mkt_hi     = Market(S0 + delta_S, SIGMA, R, 0.0, None)

mc_c_lo = MonteCarloModel(N_PROP, MARKET,  CALL_EU, PRICING_DATE, seed=12)
mc_c_hi = MonteCarloModel(N_PROP, mkt_hi,  CALL_EU, PRICING_DATE, seed=12)
mc_p_lo = MonteCarloModel(N_PROP, MARKET,  PUT_EU,  PRICING_DATE, seed=12)
mc_p_hi = MonteCarloModel(N_PROP, mkt_hi,  PUT_EU,  PRICING_DATE, seed=12)

pc_lo = mc_c_lo.price_european_vectorized(antithetic=True)['price']
pc_hi = mc_c_hi.price_european_vectorized(antithetic=True)['price']
pp_lo = mc_p_lo.price_european_vectorized(antithetic=True)['price']
pp_hi = mc_p_hi.price_european_vectorized(antithetic=True)['price']

check(
    f"5.4  Monotonie Call : prix(S+{delta_S}) > prix(S)  (delta > 0)",
    pc_hi > pc_lo,
    f"Call(S={S0+delta_S})={pc_hi:.4f}  Call(S={S0})={pc_lo:.4f}  Δ={pc_hi-pc_lo:+.4f}"
)
check(
    f"5.5  Monotonie Put : prix(S+{delta_S}) < prix(S)  (delta < 0)",
    pp_hi < pp_lo,
    f"Put(S={S0+delta_S})={pp_hi:.4f}  Put(S={S0})={pp_lo:.4f}  Δ={pp_hi-pp_lo:+.4f}"
)

# ── 5.4 Vega positif : prix monte avec σ ────────────────────────────────────
dsig       = 0.05
mkt_vol_hi = Market(S0, SIGMA + dsig, R, 0.0, None)

mc_vhi = MonteCarloModel(N_PROP, mkt_vol_hi, CALL_EU, PRICING_DATE, seed=13)
mc_vlo = MonteCarloModel(N_PROP, MARKET,     CALL_EU, PRICING_DATE, seed=13)
pv_hi  = mc_vhi.price_european_vectorized(antithetic=True)['price']
pv_lo  = mc_vlo.price_european_vectorized(antithetic=True)['price']

check(
    f"5.6  Vega Call : prix(σ+{dsig:.0%}) > prix(σ)  (vega > 0)",
    pv_hi > pv_lo,
    f"Call(σ={SIGMA+dsig:.0%})={pv_hi:.4f}  Call(σ={SIGMA:.0%})={pv_lo:.4f}  Δ={pv_hi-pv_lo:+.4f}"
)

# ── 5.5 American Call sans dividende ≈ European Call ────────────────────────
#        Résultat fondamental : l'exercice anticipé d'un call sans dividende
#        n'est jamais optimal → AM Call = EU Call
mc_ac = MonteCarloModel(N_PROP, MARKET, CALL_AM, PRICING_DATE, seed=14)
mc_ec = MonteCarloModel(N_PROP, MARKET, CALL_EU, PRICING_DATE, seed=14)
r_ac  = mc_ac.price_american_longstaff_schwartz_vectorized(
    num_steps=80, poly_degree=3, poly_basis=BasisType.LAGUERRE, antithetic=True
)
r_ec  = mc_ec.price_european_vectorized(antithetic=True)

early_prem = r_ac['price'] - r_ec['price']
total_se   = r_ac['std_error'] + r_ec['std_error']
print(f"\n  AM Call={r_ac['price']:.4f}  EU Call={r_ec['price']:.4f}  "
      f"Prime={early_prem:+.4f}  3×SE={3*total_se:.4f}")

check(
    "5.7  AM Call ≈ EU Call sans dividende  (|prime| < 3×SE)",
    abs(early_prem) < 3 * total_se,
    f"Prime={early_prem:+.4f}  3×SE={3*total_se:.4f}"
)

# ── 5.6 AM Put > EU Put (prime d'exercice anticipé strictement positive) ────
mc_ap = MonteCarloModel(N_PROP, MARKET, PUT_AM, PRICING_DATE, seed=14)
r_ap2 = mc_ap.price_american_longstaff_schwartz_vectorized(
    num_steps=80, poly_degree=3, poly_basis=BasisType.LAGUERRE, antithetic=True
)
check(
    "5.8  AM Put > EU Put  (prime d'exercice anticipé > 0)",
    r_ap2['price'] > r_p_eu['price'],
    f"AM Put={r_ap2['price']:.4f}  EU Put={r_p_eu['price']:.4f}  "
    f"Prime={r_ap2['price']-r_p_eu['price']:+.4f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  BILAN
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{'═'*66}")
print(f"  BILAN BATTERIE : {_PASS}/{total} tests passés", end="")
if _FAIL == 0:
    print("  ✓  TOUS LES TESTS SONT VERTS")
else:
    print(f"  — {_FAIL} test(s) en échec")
print("═" * 66)

# ── Graphique bilan ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(['Passed', 'Failed'], [_PASS, _FAIL],
              color=['#4caf50', '#f44336'], edgecolor='white', width=0.5)
for bar, val in zip(bars, [_PASS, _FAIL]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            str(val), ha='center', va='bottom', fontweight='bold', fontsize=14)
ax.set_ylim(0, max(_PASS + _FAIL, 1) * 1.25)
ax.set_ylabel('Nombre de tests')
ax.set_title(f'Batterie MC — bilan: {_PASS}/{_PASS+_FAIL} tests OK')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = os.path.join(PLOTS_DIR, 'test_battery_summary.png')
plt.savefig(out, dpi=150)
plt.close()
print(f"\n\u2713 Graphique sauvegard\u00e9 : {out}")