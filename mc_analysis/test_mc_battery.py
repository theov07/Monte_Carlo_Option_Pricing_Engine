"""
test_mc_battery.py -- In-depth test battery for Monte Carlo option pricing.

Run:
    python test_mc_battery.py

Difference from verify_mc.py (end-to-end tests):
  -> Here we test ALGORITHMIC COMPONENTS in isolation and stress
     the statistical properties with greater rigour.

5 suites:
  Suite 1 -- exercise_decision  (isolated from the backward loop)
  Suite 2 -- Discounted column  (constant EU / non-constant AM)
  Suite 3 -- Inter-seed stability  (std, z-scores, determinism, coverage)
  Suite 4 -- Antithetic variance reduction
  Suite 5 -- Economic properties  (vega, monotonicity, bounds, AM call)
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
from src.brownian_motion import BrownianMotion
from src.regression import Regression, BasisType

# ══════════════════════════════════════════════════════════════════════════════
#  Common parameters
# ══════════════════════════════════════════════════════════════════════════════
PRICING_DATE = date(2025, 1, 2)
MAT_DATE     = date(2026, 1, 2)   # T = 1 year exact (2025 not a leap year)
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
#  Display infrastructure
# ══════════════════════════════════════════════════════════════════════════════
_PASS = _FAIL = 0

def check(name: str, condition: bool, detail: str = "") -> bool:
    global _PASS, _FAIL
    if condition:
        _PASS += 1
        print(f"  [OK]  {name}")
    else:
        _FAIL += 1
        print(f"  [!!]  {name}  <- FAILED")
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
#  Helper: step-by-step backward induction capturing the discounted column
#
#  At each step j, we capture:
#    V[j] = mean(cash_flow_j) * e^{-r*j*dt}
#
#  Interpretation:
#    cash_flow_j = best reachable value from j onward (value at t=j)
#    * e^{-r*j*dt} = discount from j back to t=0
#
#  Theoretical properties:
#    European: V[j] = constant = price_EU  (discounted process = martingale)
#    American: V[j] INCREASES as j -> 0  (supermartingale, early exercise)
# ══════════════════════════════════════════════════════════════════════════════

def _backward_column_means(S_paths: np.ndarray,
                            option: OptionTrade,
                            r: float, T: float, num_steps: int,
                            is_american: bool) -> np.ndarray:
    """
    Returns V[j] = mean(cash_flow_j) * e^{-r*j*dt} for j = 0, ..., num_steps.
    Keeps ONLY ONE cash_flow column in memory per step (no matrix).
    """
    dt = T / num_steps
    df = np.exp(-r * dt)

    cash_flow = _payoff(S_paths[:, -1], option)        # value at maturity T
    V = np.empty(num_steps + 1)
    V[num_steps] = float(np.mean(cash_flow)) * np.exp(-r * T)

    reg = Regression(degree=3, basis=BasisType.LAGUERRE) if is_american else None

    for j in range(num_steps - 1, -1, -1):
        if is_american:
            # Exercise decision: keeps the same cash_flow column
            continuation = cash_flow * df
            intrinsic    = _payoff(S_paths[:, j], option)
            cash_flow    = reg.exercise_decision(S_paths[:, j], intrinsic, continuation)
        else:
            # European: discount one step, no exercise decision
            cash_flow = cash_flow * df

        # Discount from t_j -> t=0: obtains value in today's money
        V[j] = float(np.mean(cash_flow)) * np.exp(-r * j * dt)

    return V


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 1 -- exercise_decision  (isolated from the backward loop)
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 1 : exercise_decision -- isolated unit tests")
print(f"  (Put K={K}, tests on synthetic control data)")

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

# ── 1.3 Continuation >> intrinsic → DO NOT exercise ──────────────────────────
S_mix     = rng.uniform(60, 100, N)
iv_small  = np.maximum(K - S_mix, 0)       # <= 40
cont_big  = np.full(N, 200.0)              # crushes any intrinsic value

reg_3 = Regression(degree=3, basis=BasisType.POWER)
res_13 = reg_3.exercise_decision(S_mix, iv_small, cont_big)

# For ITM paths, regression must predict ~= 200 >> intrinsic -> continuation
itm_mask_3 = iv_small > 0
check(
    "1.3  Continuation >> intrinsic -> no cash_flow > continuation",
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

# ── 1.5 Fit quality: synthetic data with known analytical solution ──────────
N5    = 8000
S5    = rng.uniform(70, 100, N5)
# 'True' linear continuation in S (known)
a, b  = -0.8, 85.0
y_true = a * S5 + b                         # valeurs entre 5 et 29
y_obs  = y_true + rng.normal(0, 0.3, N5)   # bruit faible

reg_5 = Regression(degree=1, basis=BasisType.POWER)   # deg 1 suffit ici
reg_5.fit(S5, y_obs)
y_pred = reg_5.predict(S5)
r2 = np.corrcoef(y_pred, y_true)[0, 1] ** 2

check(
    "1.5  Linear fit (deg 1) on synthetic data: R^2(pred, true) >= 0.999",
    r2 >= 0.999,
    f"R^2 = {r2:.6f}"
)

# ── 1.6 residual_std is computed correctly and consistent with actual residuals ──
reg_6 = Regression(degree=3, basis=BasisType.POWER)
reg_6.fit(S5, y_obs)
residuals = y_obs - reg_6._design_matrix(S5) @ reg_6._coeffs
std_residuals_manual = float(np.std(residuals))
check(
    "1.6  _residual_std consistent with manually computed residuals",
    abs(reg_6._residual_std - std_residuals_manual) < 1e-10,
    f"_residual_std={reg_6._residual_std:.6f}  manual={std_residuals_manual:.6f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 2 -- Discounted column: constant (EU) vs increasing (AM)
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 2 : Discounted column  V[j] = mean(CF_j) * e^{-rjdt}")
print()
print("  Theory:")
print("    European  -> CF_j = payoff(S_T) * df^{T-j}  =>  V[j] EXACTLY constant")
print("    American -> early exercise enriches CF_j   =>  V[j] increases as j->0")

NUM_PATHS_COL = 40_000
NUM_STEPS_COL = 60

bm_col = BrownianMotion(NUM_PATHS_COL, NUM_STEPS_COL, T, antithetic=False, seed=99)
S_col, _ = bm_col.generate_paths(S0, R, SIGMA, 0.0)

print(f"\n  Simulation : {NUM_PATHS_COL:,} paths × {NUM_STEPS_COL} steps")

# ── 2.1 European: V[j] constant (exact algebraic identity) ────────────────────
V_eu = _backward_column_means(S_col, PUT_EU, R, T, NUM_STEPS_COL, is_american=False)

V_eu_range = float(V_eu.max() - V_eu.min())
V_eu_mean  = float(np.mean(V_eu))
cv_eu      = V_eu_range / V_eu_mean          # should be < 1e-8 (float)

print(f"\n  EU Put  V_mean={V_eu_mean:.6f}  V_range={V_eu_range:.2e}  CV={cv_eu:.2e}")
print(f"  BS Put  = {BS_PUT:.6f}")

check(
    "2.1  EU: V[j] algebraically constant -- range/mean < 1e-8",
    cv_eu < 1e-8,
    f"range/mean = {cv_eu:.2e}  (float ~= 0)"
)
check(
    "2.2  EU: V_mean close to BS price (error < 1.5%)",
    abs(V_eu_mean - BS_PUT) / BS_PUT < 0.015,
    f"V_mean={V_eu_mean:.4f}  BS={BS_PUT:.4f}  err={abs(V_eu_mean-BS_PUT)/BS_PUT:.3%}"
)

# ── 2.2 American: V[j] increases from V[T] -> V[0] ─────────────────────────────
V_am = _backward_column_means(S_col, PUT_AM, R, T, NUM_STEPS_COL, is_american=True)

premium = V_am[0] - V_am[-1]
print(f"\n  AM Put  V[0]={V_am[0]:.4f}  V[T]={V_am[-1]:.4f}  ΔV = {premium:+.4f}")

check(
    "2.3  AM: V[0] > V[T]  (early exercise adds value)",
    V_am[0] > V_am[-1],
    f"V_am[0]={V_am[0]:.4f}  V_am[T]={V_am[-1]:.4f}"
)
check(
    "2.4  AM: V[0] > V_EU[0]  (American worth more than European)",
    V_am[0] > V_eu[0],
    f"V_am[0]={V_am[0]:.4f}  V_eu[0]={V_eu[0]:.4f}"
)
check(
    "2.5  AM: early exercise premium > 0.1  (non-trivial)",
    premium > 0.1,
    f"Premium = {premium:.4f}"
)

# ── 2.3 Global trend of V_am: decreasing in j (increasing going backward) ────────
j_arr = np.arange(NUM_STEPS_COL + 1, dtype=float)
slope_am, _, r_am, _, _ = stats.linregress(j_arr, V_am)
check(
    "2.6  AM: slope of V_am vs j is NEGATIVE (increases in backward pass)",
    slope_am < 0,
    f"Slope = {slope_am:.6f}  R={r_am:.3f}"
)

# ── 2.4 EU vs AM: same V[T] (identical terminal payoff, same paths) ──────────────
check(
    "2.7  EU and AM have the same V[T]  (same terminal payoff, same paths)",
    abs(V_am[-1] - V_eu[-1]) < 1e-8,
    f"|V_am[T] - V_eu[T]| = {abs(V_am[-1] - V_eu[-1]):.2e}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 3 -- Inter-seed stability: std, z-scores, determinism, coverage
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 3 : Inter-seed stability -- std, z-scores, determinism")

N_RUN    = 8_000    # paths per run
N_SEEDS  = 80       # number of distinct seeds

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
print(f"  std(prices)     = {std_prices:.4f}   <- inter-seed dispersion")
print(f"  mean(SE)        = {mean_se:.4f}   <- average SE predicted by 1 run")
print(f"  Ratio std/SE    = {ratio:.3f}   (expected ~= 1.0)")

# ── 3.1 inter-seed std ~= theoretical SE ──────────────────────────────────────
check(
    "3.1  std(prices_seeds) / mean(SE) in [0.7 ; 1.3]",
    0.7 <= ratio <= 1.3,
    f"Ratio = {ratio:.3f}"
)

# ── 3.2 Global bias < 0.5% ──────────────────────────────────────────────────────
bias = abs(np.mean(prices) - BS_CALL) / BS_CALL
check(
    "3.2  Relative mean bias < 0.5%",
    bias < 0.005,
    f"Bias = {bias:.4%}   mean={np.mean(prices):.4f}  BS={BS_CALL:.4f}"
)

# ── 3.3 z-scores = (price_i − BS) / SE_i ~ N(0,1) ──────────────────────────
z_scores = (prices - BS_CALL) / ses

ks_stat, p_ks = stats.kstest(z_scores, 'norm')
print(f"\n  z-scores :  mean={np.mean(z_scores):+.3f}  std={np.std(z_scores):.3f}")
print(f"  KS stat={ks_stat:.4f}   p-value={p_ks:.4f}   (H₀ : ~ N(0,1))")

check(
    "3.3  z-scores ~ N(0,1) : KS p-value >= 0.05",
    p_ks >= 0.05,
    f"KS p = {p_ks:.4f}"
)
check(
    "3.4  |mean(z_scores)| < 0.3  (no systematic bias)",
    abs(np.mean(z_scores)) < 0.3,
    f"mean(z) = {np.mean(z_scores):+.3f}"
)
check(
    "3.5  std(z_scores) in [0.7 ; 1.3]  (SE well calibrated)",
    0.7 <= np.std(z_scores) <= 1.3,
    f"std(z) = {np.std(z_scores):.3f}"
)

# ── 3.4 Empirical 95% CI coverage ─────────────────────────────────────────────────
z95       = _norm.ppf(0.975)
in_ic     = int(np.sum(np.abs(prices - BS_CALL) <= z95 * ses))
coverage  = in_ic / N_SEEDS
print(f"\n  95% CI coverage: {in_ic}/{N_SEEDS} = {coverage:.1%}  (expected ~= 95%)")

check(
    "3.6  95% CI coverage in [80% ; 100%]",
    0.80 <= coverage <= 1.00,
    f"Coverage = {coverage:.1%}"
)

# ── 3.5 Determinism: same seed -> IDENTICAL price at machine precision ──────────────
mc_a = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=314)
mc_b = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=314)
pa = mc_a.price_european_vectorized()['price']
pb = mc_b.price_european_vectorized()['price']

check(
    "3.7  Determinism: seed=314 run twice -> identical price within 1e-12",
    abs(pa - pb) < 1e-12,
    f"|price_a - price_b| = {abs(pa - pb):.2e}"
)

# ── 3.6 Different seeds -> different prices (real stochasticity) ─────────────────
mc_s0 = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=0)
mc_s1 = MonteCarloModel(15_000, MARKET, CALL_EU, PRICING_DATE, seed=1)
p0 = mc_s0.price_european_vectorized()['price']
p1 = mc_s1.price_european_vectorized()['price']

check(
    "3.8  Different seeds -> different prices  (seed=0 != seed=1)",
    abs(p0 - p1) > 1e-6,
    f"price(seed=0)={p0:.6f}  price(seed=1)={p1:.6f}  diff={abs(p0-p1):.2e}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 4 -- Variance reduction by antithetic variables
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 4 : Variance reduction -- antithetic vs plain")

# Same total number of random draws: num_simulations fixed
N_VAR = 40_000

mc_anti  = MonteCarloModel(N_VAR, MARKET, CALL_EU, PRICING_DATE, seed=0)
mc_brute = MonteCarloModel(N_VAR, MARKET, CALL_EU, PRICING_DATE, seed=0)
r_anti   = mc_anti.price_european_vectorized(antithetic=True)
r_brute  = mc_brute.price_european_vectorized(antithetic=False)

se_anti  = r_anti['std_error']
se_brute = r_brute['std_error']
ratio_se = se_anti / se_brute

print(f"\n  N={N_VAR:,} paths  (antithetic: N/2 independent pairs)")
print(f"  SE antithetic = {se_anti:.5f}")
print(f"  SE plain      = {se_brute:.5f}")
print(f"  SE_anti / SE_plain = {ratio_se:.3f}   (expected < 1)")

check(
    "4.1  SE antithetic < SE plain (variance reduction confirmed)",
    se_anti < se_brute,
    f"{se_anti:.5f} < {se_brute:.5f}"
)
check(
    "4.2  SE_anti / SE_plain < 0.90  (reduction >= 10%)",
    ratio_se < 0.90,
    f"Ratio = {ratio_se:.3f}"
)

# Verification on put too (negative correlation often stronger)
mc_anti_put  = MonteCarloModel(N_VAR, MARKET, PUT_EU, PRICING_DATE, seed=0)
mc_brute_put = MonteCarloModel(N_VAR, MARKET, PUT_EU, PRICING_DATE, seed=0)
r_ap = mc_anti_put.price_european_vectorized(antithetic=True)
r_bp = mc_brute_put.price_european_vectorized(antithetic=False)
check(
    "4.3  SE antithetic < SE plain -- PUT also",
    r_ap['std_error'] < r_bp['std_error'],
    f"SE_anti={r_ap['std_error']:.5f}  SE_plain={r_bp['std_error']:.5f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUITE 5 -- Economic properties
# ══════════════════════════════════════════════════════════════════════════════
section("SUITE 5 : Economic properties of MC pricing")

N_PROP = 40_000

# ── 5.1 Time value strictement positive pour option ATM ─────────────────────
mc_p_eu = MonteCarloModel(N_PROP, MARKET, PUT_EU, PRICING_DATE, seed=10)
r_p_eu  = mc_p_eu.price_european_vectorized(antithetic=True)
iv_atm  = max(K - S0, 0)   # = 0 pour put ATM

check(
    "5.1  ATM EU Put: price > intrinsic value (time value > 0)",
    r_p_eu['price'] > iv_atm,
    f"Price={r_p_eu['price']:.4f}  IV={iv_atm:.4f}"
)

# ── 5.2 Lower bound: C >= max(S - K*e^{-rT}, 0) ────────────────────────────────────
lb_call = max(S0 - K * math.exp(-R * T), 0)
lb_put  = max(K * math.exp(-R * T) - S0, 0)
tol_3se = 3.0

mc_c_lb = MonteCarloModel(N_PROP, MARKET, CALL_EU, PRICING_DATE, seed=11)
mc_p_lb = MonteCarloModel(N_PROP, MARKET, PUT_EU,  PRICING_DATE, seed=11)
rc_lb   = mc_c_lb.price_european_vectorized(antithetic=True)
rp_lb   = mc_p_lb.price_european_vectorized(antithetic=True)

check(
    f"5.2  EU Call >= lb = max(S-K*e^{{-rT}},0) = {lb_call:.4f}  (within 3*SE)",
    rc_lb['price'] >= lb_call - tol_3se * rc_lb['std_error'],
    f"Price={rc_lb['price']:.4f}  LB={lb_call:.4f}"
)
check(
    f"5.3  EU Put  >= lb = max(K*e^{{-rT}}-S,0) = {lb_put:.4f}  (within 3*SE)",
    rp_lb['price'] >= lb_put - tol_3se * rp_lb['std_error'],
    f"Price={rp_lb['price']:.4f}  LB={lb_put:.4f}"
)

# ── 5.3 Monotonicity in S0: Call increases, Put decreases ───────────────────────────
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
    f"5.4  Call monotonicity: price(S+{delta_S}) > price(S)  (delta > 0)",
    pc_hi > pc_lo,
    f"Call(S={S0+delta_S})={pc_hi:.4f}  Call(S={S0})={pc_lo:.4f}  Delta={pc_hi-pc_lo:+.4f}"
)
check(
    f"5.5  Put monotonicity: price(S+{delta_S}) < price(S)  (delta < 0)",
    pp_hi < pp_lo,
    f"Put(S={S0+delta_S})={pp_hi:.4f}  Put(S={S0})={pp_lo:.4f}  Delta={pp_hi-pp_lo:+.4f}"
)

# ── 5.4 Positive vega: price increases with sigma ─────────────────────────────────
dsig       = 0.05
mkt_vol_hi = Market(S0, SIGMA + dsig, R, 0.0, None)

mc_vhi = MonteCarloModel(N_PROP, mkt_vol_hi, CALL_EU, PRICING_DATE, seed=13)
mc_vlo = MonteCarloModel(N_PROP, MARKET,     CALL_EU, PRICING_DATE, seed=13)
pv_hi  = mc_vhi.price_european_vectorized(antithetic=True)['price']
pv_lo  = mc_vlo.price_european_vectorized(antithetic=True)['price']

check(
    f"5.6  Call vega: price(sigma+{dsig:.0%}) > price(sigma)  (vega > 0)",
    pv_hi > pv_lo,
    f"Call(sigma={SIGMA+dsig:.0%})={pv_hi:.4f}  Call(sigma={SIGMA:.0%})={pv_lo:.4f}  Delta={pv_hi-pv_lo:+.4f}"
)

# ── 5.5 American call without dividend ~= European call ─────────────────────────────
#        Fundamental result: early exercise of a call without dividend
#        is never optimal  ->  AM Call = EU Call
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
    "5.7  AM Call ~= EU Call without dividend  (|premium| < 3*SE)",
    abs(early_prem) < 3 * total_se,
    f"Premium={early_prem:+.4f}  3*SE={3*total_se:.4f}"
)

# ── 5.6 AM Put > EU Put (strictly positive early exercise premium) ────────────────
mc_ap = MonteCarloModel(N_PROP, MARKET, PUT_AM, PRICING_DATE, seed=14)
r_ap2 = mc_ap.price_american_longstaff_schwartz_vectorized(
    num_steps=80, poly_degree=3, poly_basis=BasisType.LAGUERRE, antithetic=True
)
check(
    "5.8  AM Put > EU Put  (early exercise premium > 0)",
    r_ap2['price'] > r_p_eu['price'],
    f"AM Put={r_ap2['price']:.4f}  EU Put={r_p_eu['price']:.4f}  "
    f"Premium={r_ap2['price']-r_p_eu['price']:+.4f}"
)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{chr(9552)*66}")
print(f"  BATTERY SUMMARY: {_PASS}/{total} tests passed", end="")
if _FAIL == 0:
    print("  -- ALL TESTS PASSED")
else:
    print(f"  -- {_FAIL} test(s) FAILED")
print(chr(9552) * 66)