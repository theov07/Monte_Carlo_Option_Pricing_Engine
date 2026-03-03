"""
Monte Carlo vs Black-Scholes Comparison

Scenarios covered:
  1. MC convergence vs BS for an ATM Call
  2. Comparison across different strikes (moneyness)
  3. Comparison across different volatilities

OUTPUT: figures/compare_mc_bs.png
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from datetime import date
from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.monte_carlo_model import MonteCarloModel
from src.pricing.black_scholes import BlackScholes
import numpy as np
import matplotlib.pyplot as plt

PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
os.makedirs(PLOTS_DIR, exist_ok=True)


def _bs(market, option, pricing_date):
    """Shortcut: Black-Scholes price via src.black_scholes."""
    return BlackScholes(market, option, pricing_date).price()


def compare_methods():
    """Compare Monte Carlo and Black-Scholes"""
    
    print("=" * 80)
    print("COMPARISON: MONTE CARLO vs BLACK-SCHOLES")
    print("=" * 80)
    
    # Configuration
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    
    T = (maturity_date - pricing_date).days / 365.0
    
    print(f"\nMarket parameters:")
    print(f"  Spot (S0):      {S0}")
    print(f"  Strike (K):     {K}")
    print(f"  Rate (r):       {r*100}%")
    print(f"  Volatility (s): {sigma*100}%")
    print(f"  Maturity (T):   {T:.4f} yr ({int(T*365)} days)")
    
    # Black-Scholes (prix exact)
    mkt = Market(underlying=S0, vol=sigma, rate=r, div_a=0.0, ex_div_date=None)
    call_opt_bs = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
    put_opt_bs  = OptionTrade(mat=maturity_date, call_put='PUT',  ex='EUROPEAN', k=K)
    bs_call = _bs(mkt, call_opt_bs, pricing_date)
    bs_put  = _bs(mkt, put_opt_bs,  pricing_date)
    
    print("\n" + "-" * 80)
    print("BLACK-SCHOLES (Exact Analytical Price)")
    print("-" * 80)
    print(f"  Call: {bs_call:.6f}")
    print(f"  Put:  {bs_put:.6f}")
    
    # Monte Carlo with different numbers of simulations
    print("\n" + "-" * 80)
    print("MONTE CARLO (Approximation)")
    print("-" * 80)
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Call
    call_option = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=K
    )
    
    # Put
    put_option = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='EUROPEAN',
        k=K
    )
    
    print("\nCALL Option:")
    print(f"{'N Simulations':>15} | {'MC Price':>12} | {'Error vs BS':>15} | {'Error %':>12}")
    print("-" * 80)
    
    for n_sims in [1000, 5000, 10000, 50000, 100000, 500000]:
        mc_model = MonteCarloModel(n_sims, market, call_option, pricing_date, seed=42)
        result = mc_model.price_european()
        mc_price = result['price']
        error = abs(mc_price - bs_call)
        error_pct = (error / bs_call) * 100
        
        print(f"{n_sims:>15,} | {mc_price:>12.6f} | {error:>15.6f} | {error_pct:>11.4f}%")
    
    print("\nPUT Option:")
    print(f"{'N Simulations':>15} | {'MC Price':>12} | {'Error vs BS':>15} | {'Error %':>12}")
    print("-" * 80)
    
    for n_sims in [1000, 5000, 10000, 50000, 100000, 500000]:
        mc_model = MonteCarloModel(n_sims, market, put_option, pricing_date, seed=42)
        result = mc_model.price_european()
        mc_price = result['price']
        error = abs(mc_price - bs_put)
        error_pct = (error / bs_put) * 100
        
        print(f"{n_sims:>15,} | {mc_price:>12.6f} | {error:>15.6f} | {error_pct:>11.4f}%")


def test_different_moneyness():
    """Test across different moneyness levels"""
    
    print("\n\n" + "=" * 80)
    print("COMPARISON FOR DIFFERENT STRIKES (Moneyness)")
    print("=" * 80)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    T = (maturity_date - pricing_date).days / 365.0
    
    S0 = 100.0
    r = 0.05
    sigma = 0.20
    
    strikes = [80, 90, 100, 110, 120]
    n_sims = 100000
    
    print(f"\nSpot = {S0}, T = {T:.4f} yr, N = {n_sims:,} simulations")
    print("\n" + "-" * 80)
    print(f"{'Strike':>8} | {'Moneyness':>12} | {'BS Call':>10} | {'MC Call':>10} | {'Error':>10}")
    print("-" * 80)
    
    for K in strikes:
        moneyness = S0 / K
        market = Market(underlying=S0, vol=sigma, rate=r, div_a=0.0, ex_div_date=None)
        option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
        bs_price = _bs(market, option, pricing_date)
        mc_model = MonteCarloModel(n_sims, market, option, pricing_date, seed=42)
        mc_price = mc_model.price_european()['price']
        
        error = abs(mc_price - bs_price)
        
        if moneyness > 1.05:
            moneyness_label = "ITM"
        elif moneyness < 0.95:
            moneyness_label = "OTM"
        else:
            moneyness_label = "ATM"
        
        print(f"{K:>8.0f} | {moneyness_label:>12} | {bs_price:>10.6f} | {mc_price:>10.6f} | {error:>10.6f}")


def test_different_volatilities():
    """Test across different volatility levels"""
    
    print("\n\n" + "=" * 80)
    print("COMPARISON FOR DIFFERENT VOLATILITIES")
    print("=" * 80)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    T = (maturity_date - pricing_date).days / 365.0
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    
    volatilities = [0.10, 0.20, 0.30, 0.40, 0.50]
    n_sims = 100000
    
    print(f"\nSpot = {S0}, Strike = {K}, T = {T:.4f} yr, N = {n_sims:,} simulations")
    print("\n" + "-" * 80)
    print(f"{'Volatility':>12} | {'BS Call':>10} | {'MC Call':>10} | {'Error':>10} | {'Error %':>10}")
    print("-" * 80)
    
    for sigma in volatilities:
        market = Market(underlying=S0, vol=sigma, rate=r, div_a=0.0, ex_div_date=None)
        option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
        bs_price = _bs(market, option, pricing_date)
        mc_model = MonteCarloModel(n_sims, market, option, pricing_date, seed=42)
        mc_price = mc_model.price_european()['price']
        
        error = abs(mc_price - bs_price)
        error_pct = (error / bs_price) * 100
        
        print(f"{sigma*100:>11.0f}% | {bs_price:>10.6f} | {mc_price:>10.6f} | {error:>10.6f} | {error_pct:>9.4f}%")


def plot_summary():
    """Summary plot: convergence + moneyness + volatility."""
    pricing_date  = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    T_local       = (maturity_date - pricing_date).days / 365.0
    S0_l, K_l, r_l, sigma_l = 100.0, 100.0, 0.05, 0.20

    # Convergence
    n_list = [500, 1_000, 5_000, 10_000, 50_000, 100_000]
    mkt0 = Market(S0_l, sigma_l, r_l, 0.0, None)
    opt0 = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K_l)
    bs0  = _bs(mkt0, opt0, pricing_date)
    errors_n = []
    for n in n_list:
        p = MonteCarloModel(n, mkt0, opt0, pricing_date, seed=42).price_european_vectorized(antithetic=True)['price']
        errors_n.append(abs(p - bs0) / bs0 * 100)

    # Moneyness
    strikes   = [80, 90, 100, 110, 120]
    mon_label = ['ITM\nK=80', 'ITM\nK=90', 'ATM\nK=100', 'OTM\nK=110', 'OTM\nK=120']
    bs_mon, mc_mon = [], []
    for K in strikes:
        mkt_k = Market(S0_l, sigma_l, r_l, 0.0, None)
        opt_k = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K)
        bs_mon.append(_bs(mkt_k, opt_k, pricing_date))
        mc_mon.append(MonteCarloModel(50_000, mkt_k, opt_k, pricing_date, seed=42)
                      .price_european_vectorized(antithetic=True)['price'])

    # Volatility
    vols = [0.10, 0.15, 0.20, 0.30, 0.40]
    bs_vol, mc_vol = [], []
    for sv in vols:
        mkt_v = Market(S0_l, sv, r_l, 0.0, None)
        opt_v = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=K_l)
        bs_vol.append(_bs(mkt_v, opt_v, pricing_date))
        mc_vol.append(MonteCarloModel(50_000, mkt_v, opt_v, pricing_date, seed=42)
                      .price_european_vectorized(antithetic=True)['price'])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle('Monte Carlo vs Black-Scholes Comparison  --  European Option',
                 fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.loglog(n_list, errors_n, 'o-', color='steelblue', lw=2, ms=6)
    ref = [100 / n**0.5 for n in n_list]
    ax.loglog(n_list, ref, '--', color='gray', lw=1.5, label='Ref. 1/sqrt(N)')
    ax.set_xlabel('N (simulations)')
    ax.set_ylabel('Relative Error (%)')
    ax.set_title(f'ATM Call Convergence (sigma=20%)\nBS = {bs0:.4f}')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    x = range(len(strikes))
    w = 0.35
    ax.bar([i - w/2 for i in x], bs_mon, w, label='Black-Scholes', color='steelblue', alpha=0.85)
    ax.bar([i + w/2 for i in x], mc_mon, w, label='Monte Carlo',   color='tomato',    alpha=0.85)
    ax.set_xticks(list(x)); ax.set_xticklabels(mon_label, fontsize=8)
    ax.set_ylabel("Call Price")
    ax.set_title('Price by Moneyness\n(N = 50,000, sigma = 20%)')
    ax.legend(); ax.grid(axis='y', alpha=0.3)

    ax = axes[2]
    vol_pct = [v * 100 for v in vols]
    ax.plot(vol_pct, bs_vol, 'o-', color='steelblue', lw=2, ms=7, label='Black-Scholes')
    ax.plot(vol_pct, mc_vol, 's--', color='tomato',   lw=2, ms=7, label='Monte Carlo')
    ax.set_xlabel('Volatilité \u03c3 (%)')
    ax.set_ylabel('Prix Call ATM')
    ax.set_title('Prix Call ATM vs \u03c3\n(N = 50 000, K = ATM)')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'compare_mc_bs.png')
    plt.savefig(out, dpi=150)
    print(f"\n  Plot saved: {out}")
    plt.close()


if __name__ == "__main__":
    compare_methods()
    test_different_moneyness()
    test_different_volatilities()
    plot_summary()

    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("- Monte Carlo converges to Black-Scholes")
    print("- Accuracy increases with number of simulations")
    print("- Error decreases at rate 1/sqrt(N)")
    print("- Monte Carlo works for all payoffs (not just European)")
    print("=" * 80)
