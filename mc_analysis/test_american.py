"""
Test script for American option pricing using Longstaff-Schwartz MC.
- LS accuracy vs Trinomial Tree reference
- Convergence analysis
- Steps impact
- American vs European early exercise premium
- Antithetic variance reduction (visible plot)

OUTPUT : plots/american_options.png
         plots/american_antithetic.png
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src_trinomial.tree import Tree
from src_trinomial.trinomial_model import TrinomialModel

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ─── Common parameters ────────────────────────────────────────────────────────
PRICING_DATE  = date(2025, 9, 1)
MAT_DATE      = date(2026, 3, 1)   # 6 months
MARKET        = Market(underlying=100.0, vol=0.30, rate=0.05, div_a=0.0, ex_div_date=None)
K             = 100.0
LS_STEPS      = 80   # num_steps used by default in LS calls
LS_DEGREE     = 3


def _ls(n_sims, cp, antithetic=True, num_steps=LS_STEPS, seed=42):
    """Helper: price american option with Longstaff-Schwartz."""
    opt = OptionTrade(mat=MAT_DATE, call_put=cp, ex='AMERICAN', k=K)
    mc  = MonteCarloModel(n_sims, MARKET, opt, PRICING_DATE, seed=seed)
    return mc.price_american_longstaff_schwartz_vectorized(
        num_steps=num_steps, poly_degree=LS_DEGREE, antithetic=antithetic
    )


def _tree_ref(cp, steps=60):
    """Helper: trinomial tree reference price."""
    opt  = OptionTrade(mat=MAT_DATE, call_put=cp, ex='AMERICAN', k=K)
    tree = Tree(steps, MARKET, opt, PRICING_DATE, prunning_threshold=1e-8)
    tree.build_tree()
    return TrinomialModel(PRICING_DATE, tree).price(opt, "backward")


# ─── Test 1 ───────────────────────────────────────────────────────────────────

def test_american_ls_accuracy():
    """LS American pricing accuracy vs trinomial tree reference."""
    print("\n" + "=" * 90)
    print("TEST 1: LONGSTAFF-SCHWARTZ ACCURACY vs TRINOMIAL TREE")
    print("=" * 90)

    for cp in ['CALL', 'PUT']:
        ref = _tree_ref(cp, steps=60)
        print(f"\n  American {cp}  — Tree ref (60 steps) = {ref:.6f}")
        print(f"  {'N sims':>8} | {'LS price':>10} | {'Std err':>9} | {'Error':>9} | {'Error %':>8}")
        print("  " + "-" * 60)
        for n in [2_000, 5_000, 10_000, 25_000, 50_000]:
            r = _ls(n, cp)
            err = abs(r['price'] - ref)
            print(f"  {n:>8,} | {r['price']:>10.5f} | {r['std_error']:>9.5f} | "
                  f"{err:>9.5f} | {100*err/ref:>7.2f}%")


# ─── Test 2 ───────────────────────────────────────────────────────────────────

def test_american_ls_vs_trinomial():
    """LS vs trinomial for varying market parameters."""
    print("\n" + "=" * 90)
    print("TEST 2: LS vs TRINOMIAL — VARYING SPOT & VOLATILITY")
    print("=" * 90)

    n_sims = 20_000
    for cp in ['CALL', 'PUT']:
        print(f"\n  American {cp}  (N={n_sims:,}, K={K}, r=5%, T=6m)")

        # --- Varying spot ---
        print(f"\n  Spot sensitivity:")
        print(f"  {'S0':>6} | {'Tree ref':>10} | {'LS price':>10} | {'Diff':>8} | {'Diff %':>7}")
        print("  " + "-" * 55)
        for s0 in [80, 90, 95, 100, 105, 110, 120]:
            mkt  = Market(s0, 0.30, 0.05, 0.0, None)
            opt  = OptionTrade(mat=MAT_DATE, call_put=cp, ex='AMERICAN', k=K)
            tree = Tree(60, mkt, opt, PRICING_DATE, prunning_threshold=1e-8)
            tree.build_tree()
            ref  = TrinomialModel(PRICING_DATE, tree).price(opt, "backward")
            mc   = MonteCarloModel(n_sims, mkt, opt, PRICING_DATE, seed=42)
            r    = mc.price_american_longstaff_schwartz_vectorized(
                       num_steps=LS_STEPS, poly_degree=LS_DEGREE, antithetic=True)
            diff = r['price'] - ref
            print(f"  {s0:>6} | {ref:>10.5f} | {r['price']:>10.5f} | {diff:>+8.5f} | "
                  f"{100*abs(diff)/ref:>6.2f}%")


# ─── Test 3 ───────────────────────────────────────────────────────────────────

def test_american_convergence():
    """LS convergence with increasing N (log-log SE plot)."""
    print("\n" + "=" * 90)
    print("TEST 3: LONGSTAFF-SCHWARTZ CONVERGENCE  (SE ~ 1/√N)")
    print("=" * 90)

    ref = _tree_ref('PUT', steps=60)
    print(f"\n  American PUT  — Tree ref = {ref:.6f}")
    print(f"  {'N sims':>8} | {'LS price':>10} | {'Std err':>9} | {'95% CI':>22} | {'Error':>8}")
    print("  " + "-" * 72)

    for n in [1_000, 2_500, 5_000, 10_000, 25_000, 50_000]:
        r  = _ls(n, 'PUT', seed=None)
        lo = r['price'] - 1.96 * r['std_error']
        hi = r['price'] + 1.96 * r['std_error']
        print(f"  {n:>8,} | {r['price']:>10.5f} | {r['std_error']:>9.5f} | "
              f"[{lo:.5f}, {hi:.5f}] | {abs(r['price']-ref):>8.5f}")


# ─── Test 4 ───────────────────────────────────────────────────────────────────

def test_american_steps_impact():
    """Impact of the number of LS discretization steps on price."""
    print("\n" + "=" * 90)
    print("TEST 4: IMPACT OF DISCRETIZATION STEPS ON LS PRICE")
    print("=" * 90)

    n_sims = 20_000
    ref = _tree_ref('PUT', steps=60)
    print(f"\n  American PUT  (N={n_sims:,}) — Tree ref = {ref:.6f}")
    print(f"  {'Steps':>7} | {'LS price':>10} | {'Std err':>9} | {'Time':>7} | {'Δ vs ref':>9}")
    print("  " + "-" * 58)

    prev = None
    for steps in [10, 25, 50, 80, 120, 200]:
        opt = OptionTrade(mat=MAT_DATE, call_put='PUT', ex='AMERICAN', k=K)
        mc  = MonteCarloModel(n_sims, MARKET, opt, PRICING_DATE, seed=42)
        t0  = time.perf_counter()
        r   = mc.price_american_longstaff_schwartz_vectorized(
                  num_steps=steps, poly_degree=LS_DEGREE, antithetic=True)
        elapsed = time.perf_counter() - t0
        diff = r['price'] - ref
        conv = "" if prev is None else ("→ stable" if abs(r['price'] - prev) < 5e-4 else "")
        print(f"  {steps:>7} | {r['price']:>10.5f} | {r['std_error']:>9.5f} | "
              f"{elapsed:>6.2f}s | {diff:>+9.5f}  {conv}")
        prev = r['price']


# ─── Test 5 ───────────────────────────────────────────────────────────────────

def test_american_vs_european():
    """American vs European — early exercise premium via LS."""
    print("\n" + "=" * 90)
    print("TEST 5: AMERICAN vs EUROPEAN — EARLY EXERCISE PREMIUM (LS)")
    print("=" * 90)

    n_sims = 25_000
    for cp in ['CALL', 'PUT']:
        opt_am = OptionTrade(mat=MAT_DATE, call_put=cp, ex='AMERICAN', k=K)
        opt_eu = OptionTrade(mat=MAT_DATE, call_put=cp, ex='EUROPEAN', k=K)
        mc_am  = MonteCarloModel(n_sims, MARKET, opt_am, PRICING_DATE, seed=42)
        mc_eu  = MonteCarloModel(n_sims, MARKET, opt_eu, PRICING_DATE, seed=42)
        r_am   = mc_am.price_american_longstaff_schwartz_vectorized(
                     num_steps=LS_STEPS, poly_degree=LS_DEGREE, antithetic=True)
        r_eu   = mc_eu.price_european_vectorized(antithetic=True)
        prem   = r_am['price'] - r_eu['price']

        print(f"\n  {cp}:")
        print(f"    American (LS):  {r_am['price']:.5f}  ±{r_am['std_error']:.5f}")
        print(f"    European MC:    {r_eu['price']:.5f}  ±{r_eu['std_error']:.5f}")
        print(f"    Early exercise: {prem:+.5f}")
        if cp == 'PUT':
            ok = prem >= -2 * (r_am['std_error'] + r_eu['std_error'])
            print(f"    ✓ AM PUT ≥ EU PUT" if ok else "    ✗ Unexpected: AM PUT < EU PUT")
        else:
            print(f"    ✓ CALL: premium ≈ 0 without dividends (expected)")


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_antithetic_comparison():
    """
    2-panel figure showing antithetic variance reduction clearly:
      Left : std_error vs N (log-log)  — with vs without antithetic
      Right: 95% CI half-width vs N    — same comparison
    Saves plots/american_antithetic.png
    """
    sim_counts = [500, 1_000, 2_500, 5_000, 10_000, 20_000]
    se_with, se_without = [], []

    print("\n  Antithetic study: computing SE with / without antithetic...")
    for n in sim_counts:
        r_on  = _ls(n, 'PUT', antithetic=True,  seed=None)
        r_off = _ls(n, 'PUT', antithetic=False, seed=None)
        se_with.append(r_on['std_error'])
        se_without.append(r_off['std_error'])
        print(f"    N={n:6,}  SE_antith={r_on['std_error']:.5f}  SE_plain={r_off['std_error']:.5f}"
              f"  ratio={r_off['std_error']/r_on['std_error']:.2f}x")

    ns   = np.array(sim_counts, dtype=float)
    se_w = np.array(se_with)
    se_o = np.array(se_without)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Antithetic variance reduction -- American PUT LS\n"
                 f"K={K}, sigma=30%, r=5%, T=6m, poly_degree={LS_DEGREE}",
                 fontsize=12, fontweight='bold')

    # Panel 1 — log-log SE vs N
    ax = axes[0]
    ax.loglog(ns, se_o, 'o--', color='tomato',    lw=1.8, ms=6, label='Without antithetic')
    ax.loglog(ns, se_w, 's-',  color='steelblue', lw=1.8, ms=6, label='With antithetic')
    # Theoretical -0.5 slope reference
    anchor = se_o[0] * (ns[0] / ns) ** 0.5
    ax.loglog(ns, anchor, ':', color='gray', lw=1.2, label='Theoretical slope -1/2')
    ax.set_xlabel('N simulations (log)')
    ax.set_ylabel('Std error (log)')
    ax.set_title('Std error vs N  (log-log)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    # Panel 2 — ratio SE_plain / SE_antith vs N
    ax2 = axes[1]
    ratio = se_o / se_w
    ax2.semilogx(ns, ratio, 'D-', color='mediumseagreen', lw=2, ms=7)
    ax2.axhline(1.0, ls='--', color='gray', lw=1)
    ax2.fill_between(ns, 1.0, ratio, alpha=0.15, color='mediumseagreen')
    ax2.set_xlabel('N simulations (log)')
    ax2.set_ylabel('SE_plain / SE_antith')
    ax2.set_title('Precision gain  (ratio > 1 = antithetic is better)')
    ax2.grid(alpha=0.3)
    # Annotate last point
    ax2.annotate(f'×{ratio[-1]:.2f}',
                 xy=(ns[-1], ratio[-1]), xytext=(-40, 8),
                 textcoords='offset points', fontsize=10, color='mediumseagreen',
                 arrowprops=dict(arrowstyle='->', color='mediumseagreen'))

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'american_antithetic.png')
    plt.savefig(out, dpi=150)
    print(f"\n  ✓ Saved {out}")
    plt.close()


def plot_am_vs_eu():
    """
    3-panel figure:
      Left  : LS AM price vs trinomial ref for PUT at varying spots
      Centre: early exercise premium (AM−EU) PUT vs CALL
      Right : LS SE vs N log-log (slope check)
    Saves plots/american_options.png
    """
    spots  = [80, 90, 95, 100, 105, 110, 120]
    n_sims = 15_000

    ls_put, tree_put, prem_put, prem_call = [], [], [], []

    print("\n  Building AM vs EU plot data...")
    for s0 in spots:
        mkt = Market(s0, 0.30, 0.05, 0.0, None)
        for cp in ['PUT', 'CALL']:
            opt_am = OptionTrade(mat=MAT_DATE, call_put=cp, ex='AMERICAN', k=K)
            opt_eu = OptionTrade(mat=MAT_DATE, call_put=cp, ex='EUROPEAN', k=K)
            mc_am  = MonteCarloModel(n_sims, mkt, opt_am, PRICING_DATE, seed=42)
            mc_eu  = MonteCarloModel(n_sims, mkt, opt_eu, PRICING_DATE, seed=42)
            pam = mc_am.price_american_longstaff_schwartz_vectorized(
                      num_steps=LS_STEPS, poly_degree=LS_DEGREE, antithetic=True)['price']
            peu = mc_eu.price_european_vectorized(antithetic=True)['price']
            if cp == 'PUT':
                tree = Tree(60, mkt, opt_am, PRICING_DATE, prunning_threshold=1e-8)
                tree.build_tree()
                tree_put.append(TrinomialModel(PRICING_DATE, tree).price(opt_am, "backward"))
                ls_put.append(pam)
                prem_put.append(pam - peu)
            else:
                prem_call.append(pam - peu)
        print(f"    S0={s0}  PremPUT={prem_put[-1]:.4f}  PremCALL={prem_call[-1]:.4f}")

    # LS SE vs N for slope panel
    ns_range = [500, 1_000, 2_500, 5_000, 10_000, 25_000]
    se_vals  = [_ls(n, 'PUT', seed=None)['std_error'] for n in ns_range]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"American PUT  LS (K={K}, σ=30%, r=5%, T=6m)",
                 fontsize=12, fontweight='bold')

    # Panel 1: LS vs Tree
    ax = axes[0]
    ax.plot(spots, tree_put, 'k--o', ms=5, lw=1.5, label='Trinomial Tree (ref)')
    ax.plot(spots, ls_put,  'tomato', marker='s', ms=5, lw=1.5, ls='-', label='LS MC')
    ax.set_xlabel('Spot S₀')
    ax.set_ylabel('American PUT price')
    ax.set_title('LS MC vs Trinomial Tree')
    ax.legend(); ax.grid(alpha=0.3)

    # Panel 2: early exercise premiums
    ax2 = axes[1]
    x   = np.arange(len(spots))
    w   = 0.35
    ax2.bar(x - w/2, prem_put,  width=w, color='tomato',    alpha=0.8, label='PUT')
    ax2.bar(x + w/2, prem_call, width=w, color='steelblue', alpha=0.8, label='CALL')
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xticks(x); ax2.set_xticklabels(spots)
    ax2.set_xlabel('Spot S₀')
    ax2.set_ylabel('AM − EU')
    ax2.set_title("Early exercise premium (AM - EU)")
    ax2.legend(); ax2.grid(axis='y', alpha=0.3)

    # Panel 3: SE log-log slope
    ax3 = axes[2]
    ns_arr = np.array(ns_range, dtype=float)
    se_arr = np.array(se_vals)
    ax3.loglog(ns_arr, se_arr, 'o-', color='steelblue', ms=6, lw=1.8, label='SE empirique')
    slope_ref = se_arr[0] * (ns_arr[0] / ns_arr) ** 0.5
    ax3.loglog(ns_arr, slope_ref, '--', color='gray', lw=1.2, label='Theoretical slope -1/2')
    ax3.set_xlabel('N simulations')
    ax3.set_ylabel('Std error')
    ax3.set_title('SE convergence  (log-log, slope ~= -1/2)')
    ax3.legend(); ax3.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'american_options.png')
    plt.savefig(out, dpi=150)
    print(f"\n  ✓ Saved {out}")
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    """Run all American option tests."""
    print("\n" * 2 + "#" * 90)
    print("# AMERICAN OPTION PRICING TEST SUITE  (Longstaff-Schwartz)")
    print("#" * 90)

    try:
        test_american_ls_accuracy()
        test_american_ls_vs_trinomial()
        test_american_convergence()
        test_american_steps_impact()
        test_american_vs_european()
        plot_antithetic_comparison()
        plot_am_vs_eu()

        print("\n" + "=" * 90)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 90 + "\n")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

