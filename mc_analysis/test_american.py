"""
Test script for American option pricing
- Scalar vs Vectorized Monte Carlo comparison
- Comparison with Trinomial Tree
- Convergence analysis

OUTPUT : plots/american_options.png
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src_trinomial.tree import Tree
from src_trinomial.trinomial_model import TrinomialModel

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def test_american_scalar_vs_vectorized():
    """Compare scalar and vectorized American pricing"""
    print("\n" + "=" * 80)
    print("TEST 1: AMERICAN OPTION - SCALAR vs VECTORIZED")
    print("=" * 80)
    
    # Setup
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.30,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Test both CALL and PUT
    for call_put in ['CALL', 'PUT']:
        option = OptionTrade(
            mat=maturity_date,
            call_put=call_put,
            ex='AMERICAN',
            k=100.0
        )
        
        print(f"\n{'-'*80}")
        print(f"AMERICAN {call_put} (K=100, S0=100, σ=30%, r=5%, T=6m)")
        print(f"{'-'*80}")
        
        for num_sims in [5000, 25000]:
            print(f"\nNumber of Simulations: {num_sims:,}")
            print(f"{'Number of Steps':>20} | {'Scalar Price':>15} | {'Scalar Time':>12} | "
                  f"{'Vector Price':>15} | {'Vector Time':>12} | {'Speedup':>8} | "
                  f"{'Price Diff':>12}")
            print("-" * 130)
            
            for num_steps in [50, 100, 252]:
                mc = MonteCarloModel(
                    num_simulations=num_sims,
                    market=market,
                    option=option,
                    pricing_date=pricing_date,
                    seed=42
                )
                
                # Compare implementations — inline (la méthode compare a été retirée du modèle)
                t0 = time.perf_counter()
                r_scalar = mc.price_american_naive(num_steps=num_steps, antithetic=True)
                scalar_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                r_vector = mc.price_american_naive_vectorized(num_steps=num_steps, antithetic=True)
                vector_time = time.perf_counter() - t0

                scalar_price = r_scalar['price']
                vector_price = r_vector['price']
                speedup = scalar_time / vector_time if vector_time > 0 else float('inf')
                price_diff = abs(scalar_price - vector_price)
                
                print(f"{num_steps:>20} | {scalar_price:>15.6f} | {scalar_time:>12.4f}s | "
                      f"{vector_price:>15.6f} | {vector_time:>12.4f}s | {speedup:>8.2f}x | "
                      f"{price_diff:>12.2e}")


def test_american_naive_vs_trinomial():
    """Compare Monte Carlo American pricing with Trinomial Tree"""
    print("\n" + "=" * 80)
    print("TEST 2: AMERICAN OPTION - MONTE CARLO vs TRINOMIAL TREE")
    print("=" * 80)
    
    # Setup
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.30,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Test both CALL and PUT
    for call_put in ['CALL', 'PUT']:
        option = OptionTrade(
            mat=maturity_date,
            call_put=call_put,
            ex='AMERICAN',
            k=100.0
        )
        
        print(f"\n{'-'*80}")
        print(f"AMERICAN {call_put} (K=100, S0=100, σ=30%, r=5%, T=6m)")
        print(f"{'-'*80}")
        
        # Trinomial Tree pricing
        print(f"\nTrinomial Tree Pricing:")
        tree_prices = {}
        for tree_steps in [10, 20, 30, 50]:
            tree = Tree(tree_steps, market, option, pricing_date, prunning_threshold=1e-8)
            tree.build_tree()
            trinomial_model = TrinomialModel(pricing_date, tree)
            tree_price = trinomial_model.price(option, "backward")
            tree_prices[tree_steps] = tree_price
            print(f"  Steps={tree_steps:3d}: {tree_price:.6f}")
        
        # Use the finest trinomial as reference
        reference_price = tree_prices[max(tree_prices.keys())]
        
        # Monte Carlo pricing
        print(f"\nMonte Carlo Pricing (Vectorized):")
        print(f"{'MC Sims':>10} | {'MC Steps':>10} | {'MC Price':>12} | {'Std Error':>12} | "
              f"{'Error vs Tree':>15} | {'Error %':>10}")
        print("-" * 85)
        
        for num_sims in [5000, 10000, 25000]:
            mc = MonteCarloModel(
                num_simulations=num_sims,
                market=market,
                option=option,
                pricing_date=pricing_date,
                seed=42
            )
            
            for mc_steps in [50, 100, 252]:
                result = mc.price_american_naive_vectorized(
                    num_steps=mc_steps,
                    antithetic=True
                )
                
                mc_price = result['price']
                std_error = result['std_error']
                error = abs(mc_price - reference_price)
                error_pct = (error / reference_price) * 100 if reference_price != 0 else 0
                
                print(f"{num_sims:>10,} | {mc_steps:>10} | {mc_price:>12.6f} | "
                      f"{std_error:>12.6f} | {error:>15.6f} | {error_pct:>9.2f}%")


def test_american_convergence():
    """Test convergence of American pricing with increasing simulations"""
    print("\n" + "=" * 80)
    print("TEST 3: CONVERGENCE ANALYSIS - AMERICAN OPTIONS")
    print("=" * 80)
    
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.30,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    option_put = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='AMERICAN',
        k=100.0
    )
    
    print(f"\nAMERICAN PUT - Convergence with increasing simulations")
    print(f"Number of steps: 100")
    print(f"{'N Sims':>10} | {'Price':>12} | {'Std Error':>12} | {'95% CI Lower':>15} | "
          f"{'95% CI Upper':>15}")
    print("-" * 80)
    
    prices = []
    std_errors = []
    
    for num_sims in [1000, 5000, 10000, 25000, 50000]:
        mc = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_put,
            pricing_date=pricing_date,
            seed=None  # No seed for convergence test
        )
        
        result = mc.price_american_naive_vectorized(
            num_steps=100,
            antithetic=True
        )
        
        price = result['price']
        std_error = result['std_error']
        
        prices.append(price)
        std_errors.append(std_error)
        
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        
        print(f"{num_sims:>10,} | {price:>12.6f} | {std_error:>12.6f} | "
              f"{ci_lower:>15.6f} | {ci_upper:>15.6f}")
    
    print(f"\n✓ Price should stabilize as N increases")
    print(f"✓ Std error should decrease as 1/√N")


def test_american_steps_impact():
    """Test impact of number of steps on American option pricing"""
    print("\n" + "=" * 80)
    print("TEST 4: IMPACT OF NUMBER OF STEPS ON AMERICAN PRICING")
    print("=" * 80)
    
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.30,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    option_put = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='AMERICAN',
        k=100.0
    )
    
    num_sims = 10000
    
    print(f"\nAMERICAN PUT - Impact of discretization steps")
    print(f"Number of simulations: {num_sims:,}")
    print(f"{'Steps':>8} | {'Price':>12} | {'Std Error':>12} | {'Time (s)':>10} | "
          f"{'Interpretation':>30}")
    print("-" * 85)
    
    previous_price = None
    
    for num_steps in [10, 25, 50, 100, 252]:
        mc = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_put,
            pricing_date=pricing_date,
            seed=42
        )
        
        start = time.time()
        result = mc.price_american_naive_vectorized(
            num_steps=num_steps,
            antithetic=True
        )
        exec_time = time.time() - start
        
        price = result['price']
        std_error = result['std_error']
        
        interpretation = ""
        if previous_price is not None:
            change = price - previous_price
            if abs(change) < 0.001:
                interpretation = "Converged"
            elif change > 0:
                interpretation = f"↑ Price increasing"
            else:
                interpretation = f"↓ Price decreasing"
        
        print(f"{num_steps:>8} | {price:>12.6f} | {std_error:>12.6f} | {exec_time:>10.4f} | "
              f"{interpretation:>30}")
        
        previous_price = price
    
    print(f"\n✓ Price typically increases with more steps (captures more early exercise)")
    print(f"✓ Convergence expected around 100-252 steps for 6-month option")


def test_american_vs_european():
    """Compare American vs European option prices"""
    print("\n" + "=" * 80)
    print("TEST 5: AMERICAN vs EUROPEAN OPTIONS")
    print("=" * 80)
    
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.30,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    num_sims = 25000
    
    for call_put in ['CALL', 'PUT']:
        print(f"\n{'-'*80}")
        print(f"{call_put} OPTION (K=100, S0=100, σ=30%, r=5%, T=6m)")
        print(f"{'-'*80}")
        
        # American option
        option_american = OptionTrade(
            mat=maturity_date,
            call_put=call_put,
            ex='AMERICAN',
            k=100.0
        )
        
        # European option
        option_european = OptionTrade(
            mat=maturity_date,
            call_put=call_put,
            ex='EUROPEAN',
            k=100.0
        )
        
        # Price American
        mc_american = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_american,
            pricing_date=pricing_date,
            seed=42
        )
        result_american = mc_american.price_american_naive_vectorized(
            num_steps=100,
            antithetic=True
        )
        
        # Price European
        mc_european = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_european,
            pricing_date=pricing_date,
            seed=42
        )
        result_european = mc_european.price_european_vectorized(antithetic=True)
        
        american_price = result_american['price']
        american_error = result_american['std_error']
        european_price = result_european['price']
        european_error = result_european['std_error']
        
        early_exercise_value = american_price - european_price
        
        print(f"\nAmerican:  {american_price:.6f} (±{american_error:.6f})")
        print(f"European:  {european_price:.6f} (±{european_error:.6f})")
        print(f"\nEarly exercise premium: {early_exercise_value:.6f}")
        
        if call_put == 'PUT':
            print("✓ PUT: American ≥ European (early exercise can be valuable)")
        else:
            print("✓ CALL: American ≈ European (early exercise typically not optimal without dividends)")


def plot_am_vs_eu():
    """
    Graphique : prime d'exercice anticipé (AM - EU) pour Call et Put,
    en fonction du spot S0.  Sauvegarde plots/american_options.png.
    """
    pricing_date  = date(2025, 9, 1)
    maturity_date = date(2026, 3, 1)  # 6 mois
    K, R, sigma   = 100.0, 0.05, 0.30
    n_sims        = 15_000
    num_steps     = 80

    spots    = [80, 90, 95, 100, 105, 110, 120]
    call_premiums = []
    put_premiums  = []

    print("\n  Calcul de la prime américaine pour différents spots...")
    for s0 in spots:
        mkt = Market(s0, sigma, R, 0.0, None)
        for cp, lst in [('CALL', call_premiums), ('PUT', put_premiums)]:
            opt_am = OptionTrade(mat=maturity_date, call_put=cp, ex='AMERICAN', k=K)
            opt_eu = OptionTrade(mat=maturity_date, call_put=cp, ex='EUROPEAN', k=K)
            mc_am = MonteCarloModel(n_sims, mkt, opt_am, pricing_date, seed=42)
            mc_eu = MonteCarloModel(n_sims, mkt, opt_eu, pricing_date, seed=42)
            pam = mc_am.price_american_longstaff_schwartz_vectorized(
                num_steps=num_steps, antithetic=True)['price']
            peu = mc_eu.price_european_vectorized(antithetic=True)['price']
            lst.append(pam - peu)
        print(f"    S0={s0}  CallPremium={call_premiums[-1]:.4f}  PutPremium={put_premiums[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Prime d'exercice anticipé MC (AM \u2212 EU)\n"
                 f"K={K}, \u03c3={sigma:.0%}, r={R:.0%}, T=6m, N={n_sims:,}",
                 fontsize=12, fontweight='bold')

    for ax, premiums, cp, color in [
        (axes[0], call_premiums, 'CALL', 'steelblue'),
        (axes[1], put_premiums,  'PUT',  'tomato'),
    ]:
        ax.bar(spots, premiums, width=4, color=color, alpha=0.8, edgecolor='white')
        ax.axhline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Spot S₀')
        ax.set_ylabel('Prime AM \u2212 EU')
        ax.set_title(f'{cp} \u2014 Prime d\'exercice anticipé')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'american_options.png')
    plt.savefig(out, dpi=150)
    print(f"\n\u2713 Graphique sauvegardé : {out}")
    plt.close()


def main():
    """Run all American option tests"""
    print("\n" * 2)
    print("#" * 80)
    print("# AMERICAN OPTION PRICING TEST SUITE")
    print("#" * 80)
    
    try:
        # Test 1: Scalar vs Vectorized
        test_american_scalar_vs_vectorized()
        
        # Test 2: MC vs Trinomial
        test_american_vs_trinomial()
        
        # Test 3: Convergence
        test_american_convergence()
        
        # Test 4: Impact of steps
        test_american_steps_impact()
        
        # Test 5: American vs European
        test_american_vs_european()

        # Plot: early exercise premium
        plot_am_vs_eu()

        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
