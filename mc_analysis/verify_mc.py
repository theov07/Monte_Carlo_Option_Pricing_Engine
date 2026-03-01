"""
verify_mc.py -- Systematic verification tests for Monte Carlo option pricing.

Run:
    python verify_mc.py

Tests implemented (following course notes order):
  1. European scalar price (call + put) vs Black-Scholes          -> 3-sigma CI
  2. Convergence: log-log slope of SE vs N  ~= -0.5              -> +/-0.1
  3. Likelihood / normality test for Brownian increments          -> KS p > 0.05
  4. Martingale: mean(e^{-r*t}*S(t)) ~= S0 for all t             -> < 0.5 %
  5. European put-call parity                                      -> 3-sigma CI
  6. Inter-seed reproducibility: std_seeds ~= theoretical SE       -> ratio [0.7 ; 1.3]
  7. American put price > European put price                        -> strict
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
#  Common parameters
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
#  Display infrastructure
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
        print(f"  [!!]  {name}  <- FAILED")
    if detail:
        print(f"        {detail}")


def section(title: str) -> None:
    bar = "─" * 64
    print(f"\n{'═'*64}")
    print(f"  {title}")
    print(bar)


# ══════════════════════════════════════════════════════════════════════════════
#  TEST 1 -- European scalar price (path-by-path loop) vs Black-Scholes
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 1 : European Scalar Price vs Black-Scholes")
print(f"  BS references  ->  Call={BS_CALL:.4f}   Put={BS_PUT:.4f}")
print(f"  (S={S0}, K={K}, r={R:.0%}, sigma={SIGMA:.0%}, T={T}y)")

N_SCALAR = 20_000
mc_c = MonteCarloModel(N_SCALAR, MARKET, CALL_EU, PRICING_DATE, seed=42)
mc_p = MonteCarloModel(N_SCALAR, MARKET, PUT_EU,  PRICING_DATE, seed=42)

rc = mc_c.price_european(antithetic=True)
rp = mc_p.price_european(antithetic=True)

print(f"\n  MC Call (N={N_SCALAR:,}) = {rc['price']:.4f}  SE={rc['std_error']:.4f}")
print(f"  MC Put  (N={N_SCALAR:,}) = {rp['price']:.4f}  SE={rp['std_error']:.4f}")

TOL = 3.0   # number of standard deviations

check(
    "European scalar call within 3-sigma CI",
    abs(rc['price'] - BS_CALL) < TOL * rc['std_error'],
    f"|{rc['price']:.4f} - {BS_CALL:.4f}| = {abs(rc['price']-BS_CALL):.4f} "
    f"vs 3*SE = {TOL*rc['std_error']:.4f}"
)
check(
    "European scalar put within 3-sigma CI",
    abs(rp['price'] - BS_PUT) < TOL * rp['std_error'],
    f"|{rp['price']:.4f} - {BS_PUT:.4f}| = {abs(rp['price']-BS_PUT):.4f} "
    f"vs 3*SE = {TOL*rp['std_error']:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 2 -- Convergence: log-log slope of SE vs N ~= -0.5
#  Theory: SE = sigma_payoff / sqrt(N)  ->  log(SE) = const - 0.5*log(N)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 2 : Convergence  log(SE) vs log(N)  -- expected slope = -0.5")

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

# Linear regression on (log N, log SE)
log_n  = np.log(n_values)
log_se = np.log(se_list)
slope, intercept, r_value, _, _ = stats.linregress(log_n, log_se)

print(f"\n  Estimated slope = {slope:.4f}   (expected ~= -0.5)   R^2 = {r_value**2:.5f}")

check(
    "Slope log(SE)/log(N) in [-0.6 ; -0.4]",
    -0.6 <= slope <= -0.4,
    f"Slope = {slope:.4f}"
)
check(
    "R^2 of fit >= 0.99  (good linear fit)",
    r_value**2 >= 0.99,
    f"R^2 = {r_value**2:.5f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 3 -- Brownian increment normality
#  Theory: dW ~ N(0, dt)  ->  Z = dW / sqrt(dt) ~ N(0, 1)
#  Kolmogorov-Smirnov test: H0 = normal distribution
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 3 : Brownian Increment Normality  (KS test)")

N_BM  = 50_000
DT    = T / 100        # 100 pas de temps
bm    = BrownianMotion(N_BM, 100, T, antithetic=False, seed=123)
dW    = bm.generate_increments_vectorized()   # shape (N_BM, 100)
Z_all = (dW / np.sqrt(DT)).flatten()          # N_BM*100 realisations of N(0,1)

ks_stat, p_value = stats.kstest(Z_all, 'norm')

print(f"\n  N increments = {len(Z_all):,}")
print(f"  Mean         = {np.mean(Z_all):+.6f}   (expected 0)")
print(f"  Std dev      = {np.std(Z_all):.6f}    (expected 1)")
print(f"  KS statistic = {ks_stat:.6f}")
print(f"  KS p-value   = {p_value:.4f}   (threshold alpha = 0.01)")

check(
    "Mean of Z ~= 0  (|mu| < 0.005)",
    abs(np.mean(Z_all)) < 0.005,
    f"mu = {np.mean(Z_all):+.6f}"
)
check(
    "Std dev of Z ~= 1  (|sigma-1| < 0.01)",
    abs(np.std(Z_all) - 1) < 0.01,
    f"sigma = {np.std(Z_all):.6f}"
)
check(
    "KS normality test p-value >= 0.01",
    p_value >= 0.01,
    f"p = {p_value:.4f}"
)

# Verification by skewness and kurtosis
skew = stats.skew(Z_all)
kurt = stats.kurtosis(Z_all)   # excess kurtosis, should be ~= 0
print(f"\n  Skewness     = {skew:+.4f}   (expected 0)")
print(f"  Excess kurt  = {kurt:+.4f}   (expected 0)")
check(
    "Skewness ~= 0  (|skew| < 0.05)",
    abs(skew) < 0.05,
    f"Skewness = {skew:+.4f}"
)
check(
    "Excess kurtosis ~= 0  (|kurt| < 0.05)",
    abs(kurt) < 0.05,
    f"Kurtosis = {kurt:+.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 4 -- Martingale: mean(e^{-r*t_j}*S(t_j)) ~= S0 for all j
#  Fundamental property of GBM under risk-neutral measure Q:
#    E^Q[e^{-r*t}*S(t)] = S0  (discounted process = martingale)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 4 : Martingale -- mean(e^{-rt}*S(t)) = S0 for all t")

N_MART  = 30_000
N_STEPS = 50
bm_m = BrownianMotion(N_MART, N_STEPS, T, antithetic=False, seed=7)
S_paths, _ = bm_m.generate_paths(S0, R, SIGMA, 0.0)

dt_m   = T / N_STEPS
errors = []

print(f"\n  {'Pas j':>6}   {'t_j':>6}   {'mean(e^{{-rt}}·S)':>18}   {'Erreur rel.':>12}")
print(f"  {'─'*50}")

# Display some values at regular intervals
display_steps = list(range(0, N_STEPS + 1, N_STEPS // 5))
for j in range(N_STEPS + 1):
    t_j   = j * dt_m
    disc  = np.mean(np.exp(-R * t_j) * S_paths[:, j])
    err   = abs(disc - S0) / S0
    errors.append(err)
    if j in display_steps:
        print(f"  {j:>6}   {t_j:>6.3f}   {disc:>18.4f}   {err:>11.3%}")

max_err = max(errors)
print(f"\n  Max relative error (over all steps) = {max_err:.4%}")

check(
    "Martingale: max relative error < 0.5%",
    max_err < 0.005,
    f"Max error = {max_err:.4%}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 5 -- European put-call parity
#  C - P = S0 - K*e^{-rT}  (no dividend)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 5 : Put-Call Parity  C - P = S0 - K*e^{-rT}")

import math
theoretical = S0 - K * math.exp(-R * T)

N_PAR = 50_000
mc_cv = MonteCarloModel(N_PAR, MARKET, CALL_EU, PRICING_DATE, seed=99)
mc_pv = MonteCarloModel(N_PAR, MARKET, PUT_EU,  PRICING_DATE, seed=99)

rv_c = mc_cv.price_european_vectorized(antithetic=True)
rv_p = mc_pv.price_european_vectorized(antithetic=True)

cp_diff = rv_c['price'] - rv_p['price']
# Error on C-P ~= sqrt(SE_C^2 + SE_P^2)  (if independent)
se_diff = np.sqrt(rv_c['std_error']**2 + rv_p['std_error']**2)

print(f"\n  BS  C − P = {BS_CALL - BS_PUT:.4f}   (analytique exact)")
print(f"  MC  C     = {rv_c['price']:.4f}   SE={rv_c['std_error']:.4f}")
print(f"  MC  P     = {rv_p['price']:.4f}   SE={rv_p['std_error']:.4f}")
print(f"  MC  C − P = {cp_diff:.4f}   SE_diff={se_diff:.4f}")
print(f"  Theoretical = {theoretical:.4f}")
print(f"  Difference  = {abs(cp_diff - theoretical):.4f}  vs  3*SE = {3*se_diff:.4f}")

check(
    "Put-call parity within 3-sigma CI",
    abs(cp_diff - theoretical) < TOL * se_diff,
    f"|C-P - theo| = {abs(cp_diff - theoretical):.4f} < {TOL}*{se_diff:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 6 -- Inter-seed reproducibility
#  For fixed N, estimators from different seeds should satisfy
#  std(prices_seeds) ~= theoretical_SE = std(payoffs) / sqrt(N)
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 6 : Inter-seed reproducibility -- std_seeds ~= theoretical SE")

N_SEED     = 10_000   # N per run
N_REPEATS  = 50       # number of distinct seeds

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

print(f"\n  N = {N_SEED:,}   repetitions = {N_REPEATS}")
print(f"  Mean of estimators       = {np.mean(prices_seeds):.4f}   (BS = {BS_CALL:.4f})")
print(f"  Inter-seed std           = {std_seeds:.4f}")
print(f"  Theoretical SE (1 run)   = {se_single:.4f}")
print(f"  Ratio std_seeds / SE     = {ratio:.3f}   (expected ~= 1)")

check(
    "BS within [mean +/- 2*sigma] of estimators",
    abs(np.mean(prices_seeds) - BS_CALL) < 2 * std_seeds,
    f"|{np.mean(prices_seeds):.4f} - {BS_CALL:.4f}| vs 2*{std_seeds:.4f}"
)
check(
    "Ratio std_seeds / SE in [0.7 ; 1.3]",
    0.7 <= ratio <= 1.3,
    f"Ratio = {ratio:.3f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  TEST 7 -- American price > European price  (put with no dividend)
#  For a put without dividend, the American is always worth more than the European.
# ══════════════════════════════════════════════════════════════════════════════
section("TEST 7 : American Put Price >= European Put Price")

N_AM = 50_000
mc_am = MonteCarloModel(N_AM, MARKET, PUT_AM, PRICING_DATE, seed=42)
r_am  = mc_am.price_american_longstaff_schwartz_vectorized(
    num_steps=100, poly_degree=3, poly_basis=BasisType.LAGUERRE,
    antithetic=True
)

mc_eu = MonteCarloModel(N_AM, MARKET, PUT_EU, PRICING_DATE, seed=42)
r_eu  = mc_eu.price_european_vectorized(antithetic=True)

print(f"\n  American put (LS)      = {r_am['price']:.4f}   SE={r_am['std_error']:.4f}")
print(f"  European put (MC)      = {r_eu['price']:.4f}   SE={r_eu['std_error']:.4f}")
print(f"  European put (BS)      = {BS_PUT:.4f}")
print(f"  American premium MC    = {r_am['price'] - r_eu['price']:+.4f}")

check(
    "American price >= European price (MC)",
    r_am['price'] >= r_eu['price'],
    f"{r_am['price']:.4f} >= {r_eu['price']:.4f}"
)
check(
    "American price >= BS Put (analytical)",
    r_am['price'] >= BS_PUT - 2 * r_am['std_error'],
    f"{r_am['price']:.4f} >= {BS_PUT:.4f} - 2*{r_am['std_error']:.4f}"
)

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total = _PASS + _FAIL
print(f"\n{chr(9552)*64}")
print(f"  SUMMARY: {_PASS}/{total} tests passed", end="")
if _FAIL == 0:
    print("  -- ALL TESTS PASSED")
else:
    print(f"  -- {_FAIL} test(s) FAILED")
print(chr(9552)*64)