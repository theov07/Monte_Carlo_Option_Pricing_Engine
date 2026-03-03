"""
Scalar vs Vectorized Comparison -- Monte Carlo Implementation

Measures the speedup of the vectorized version over the scalar version
for different N (number of paths).

OUTPUT: plots/scalar_vs_vectorized.png
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.monte_carlo_model import MonteCarloModel



PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'figures')
os.makedirs(PLOTS_DIR, exist_ok=True)


def _compare_inline(mc: MonteCarloModel, antithetic: bool) -> dict:
    """Compare scalar vs vectorized on the same mc object."""
    results = {}
    for label, method in [('scalar', mc.price_european),
                           ('vectorized', mc.price_european_vectorized)]:
        t0 = time.perf_counter()
        res = method(antithetic=antithetic)
        results[label] = {
            'price': res['price'],
            'std_error': res['std_error'],
            'time': time.perf_counter() - t0,
        }
    results['price_difference'] = abs(results['scalar']['price'] - results['vectorized']['price'])
    results['speedup'] = (results['scalar']['time'] / results['vectorized']['time']
                          if results['vectorized']['time'] > 0 else float('inf'))
    return results


def test_comparison():
    pricing_date  = date(2026, 1, 22)
    maturity_date = date(2026, 7, 22)

    market = Market(underlying=100.0, vol=0.2, rate=0.05, div_a=0.0, ex_div_date=None)
    option_call = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=100.0)
    option_put  = OptionTrade(mat=maturity_date, call_put='PUT',  ex='EUROPEAN', k=100.0)

    print("=" * 70)
    print("COMPARISON: SCALAR vs VECTORIZED Monte Carlo")
    print("=" * 70)

    for num_sims in [1000, 10000, 100000]:
        print(f"\n{'='*70}")
        print(f"Number of simulations: {num_sims:,}")
        print("=" * 70)
        for option, name in [(option_call, "CALL"), (option_put, "PUT")]:
            print(f"\n--- {name} Option (K=100, S0=100, sigma=20%, r=5%, T=6m) ---")
            for antithetic in [False, True]:
                anti_str = "WITH antithetic" if antithetic else "WITHOUT antithetic"
                print(f"\n  {anti_str}:")
                mc = MonteCarloModel(num_simulations=num_sims, market=market,
                                     option=option, pricing_date=pricing_date, seed=42)
                cmp = _compare_inline(mc, antithetic)
                print(f"    Scalar      : price = {cmp['scalar']['price']:.6f}  "
                      f"(+/-{cmp['scalar']['std_error']:.6f})  "
                      f"t = {cmp['scalar']['time']*1000:.1f} ms")
                print(f"    Vectorized  : price = {cmp['vectorized']['price']:.6f}  "
                      f"(+/-{cmp['vectorized']['std_error']:.6f})  "
                      f"t = {cmp['vectorized']['time']*1000:.1f} ms")
                print(f"    Difference  : {cmp['price_difference']:.2e}")
                print(f"    Speedup     : {cmp['speedup']:.1f}x")


def plot_speedup():
    """Speedup plot: vectorized / scalar as a function of N."""
    pricing_date  = date(2026, 1, 22)
    maturity_date = date(2026, 7, 22)
    market = Market(underlying=100.0, vol=0.2, rate=0.05, div_a=0.0, ex_div_date=None)
    option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=100.0)

    n_list   = [500, 1_000, 5_000, 10_000, 50_000]
    speedups_no_anti  = []
    speedups_anti     = []

    print("\n  Computing speedups...")
    for n in n_list:
        mc = MonteCarloModel(n, market, option, pricing_date, seed=42)
        for anti, lst in [(False, speedups_no_anti), (True, speedups_anti)]:
            cmp = _compare_inline(mc, anti)
            lst.append(cmp['speedup'])
        print(f"    N={n:>6,}  speedup(anti=False)={speedups_no_anti[-1]:.1f}×  "
              f"speedup(anti=True)={speedups_anti[-1]:.1f}×")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_list, speedups_no_anti, 'o-', color='steelblue', lw=2, ms=7,
            label='Without antithetic')
    ax.plot(n_list, speedups_anti,    's-', color='tomato',    lw=2, ms=7,
            label='With antithetic')
    ax.axhline(1.0, color='gray', ls='--', lw=1, label='No gain (1x)')
    ax.set_xscale('log')
    ax.set_xlabel('N  (number of paths)')
    ax.set_ylabel('Speedup  (t_scalar / t_vectorized)')
    ax.set_title('Speedup  -- Vectorized vs Scalar\n'
                 '(ATM Call, S0=100, K=100, sigma=20%, r=5%, T=6m)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'scalar_vs_vectorized.png')
    plt.savefig(out, dpi=150)
    print(f"\n✓ Saved: {out}")
    plt.close()


if __name__ == "__main__":
    test_comparison()
    plot_speedup()
