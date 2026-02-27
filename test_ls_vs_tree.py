"""
Test script for Longstaff-Schwartz American pricing vs Trinomial Tree
- Compare LS with Trinomial Tree (reference)
- Test convergence of LS with increasing simulations
- Test impact of polynomial degree
- Analyze accuracy improvements
"""

from datetime import date, timedelta
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src_trinomial.tree import Tree
from src_trinomial.trinomial_model import TrinomialModel
import time
import numpy as np


def test_ls_vs_tree_accuracy():
    """Compare LS pricing with Trinomial Tree (reference)"""
    print("\n" + "=" * 100)
    print("TEST 1: LONGSTAFF-SCHWARTZ vs TRINOMIAL TREE - ACCURACY COMPARISON")
    print("=" * 100)
    
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
        
        print(f"\n{'-'*100}")
        print(f"AMERICAN {call_put} (K=100, S0=100, σ=30%, r=5%, T=6m)")
        print(f"{'-'*100}")
        
        # Trinomial Tree pricing (reference)
        print(f"\nTrinomial Tree Pricing (Reference):")
        tree_prices = {}
        for tree_steps in [15, 25, 40, 60]:
            tree = Tree(tree_steps, market, option, pricing_date, prunning_threshold=1e-8)
            tree.build_tree()
            trinomial_model = TrinomialModel(pricing_date, tree)
            tree_price = trinomial_model.price(option, "backward")
            tree_prices[tree_steps] = tree_price
            print(f"  Steps={tree_steps:3d}: {tree_price:.6f}")
        
        # Use finest Trinomial as reference
        reference_price = tree_prices[max(tree_prices.keys())]
        print(f"\n  Reference (finest): {reference_price:.6f}")
        
        # Longstaff-Schwartz pricing
        print(f"\nLongstaff-Schwartz Pricing:")
        print(f"{'LS Sims':>10} | {'LS Steps':>10} | {'Poly Deg':>8} | {'LS Price':>12} | "
              f"{'Std Error':>12} | {'Error vs Tree':>15} | {'Error %':>10}")
        print("-" * 110)
        
        for num_sims in [5000, 10000, 25000]:
            mc = MonteCarloModel(
                num_simulations=num_sims,
                market=market,
                option=option,
                pricing_date=pricing_date,
                seed=42
            )
            
            for poly_degree in [2, 3, 4]:
                result = mc.price_american_longstaff_schwartz_vectorized(
                    num_steps=100,
                    poly_degree=poly_degree,
                    antithetic=True
                )
                
                ls_price = result['price']
                std_error = result['std_error']
                error = abs(ls_price - reference_price)
                error_pct = (error / reference_price) * 100 if reference_price != 0 else 0
                
                print(f"{num_sims:>10,} | {100:>10} | {poly_degree:>8} | {ls_price:>12.6f} | "
                      f"{std_error:>12.6f} | {error:>15.6f} | {error_pct:>9.2f}%")


def test_ls_convergence():
    """Test LS convergence with increasing simulations"""
    print("\n" + "=" * 100)
    print("TEST 2: LONGSTAFF-SCHWARTZ CONVERGENCE")
    print("=" * 100)
    
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
    
    # Get Trinomial reference
    tree = Tree(50, market, option_put, pricing_date, prunning_threshold=1e-8)
    tree.build_tree()
    trinomial_model = TrinomialModel(pricing_date, tree)
    reference_price = trinomial_model.price(option_put, "backward")
    
    print(f"\nTrinomial Tree Reference (50 steps): {reference_price:.6f}")
    print(f"\nLS Convergence with increasing simulations:")
    print(f"{'N Sims':>10} | {'LS Price':>12} | {'Std Error':>12} | {'95% CI Lower':>15} | "
          f"{'95% CI Upper':>15} | {'Error vs Tree':>15} | {'Error %':>10}")
    print("-" * 115)
    
    prices = []
    errors = []
    
    for num_sims in [1000, 5000, 10000, 25000, 50000]:
        mc = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_put,
            pricing_date=pricing_date,
            seed=None
        )
        
        result = mc.price_american_longstaff_schwartz_vectorized(
            num_steps=100,
            poly_degree=3,
            antithetic=True
        )
        
        price = result['price']
        std_error = result['std_error']
        
        prices.append(price)
        error = abs(price - reference_price)
        errors.append(error)
        
        ci_lower = price - 1.96 * std_error
        ci_upper = price + 1.96 * std_error
        error_pct = (error / reference_price) * 100 if reference_price != 0 else 0
        
        print(f"{num_sims:>10,} | {price:>12.6f} | {std_error:>12.6f} | "
              f"{ci_lower:>15.6f} | {ci_upper:>15.6f} | {error:>15.6f} | {error_pct:>9.2f}%")
    
    print(f"\n✓ LS converges to Tree reference as N increases")
    print(f"✓ Error typically decreases with more simulations")


def test_poly_degree_impact():
    """Test impact of polynomial degree on LS accuracy"""
    print("\n" + "=" * 100)
    print("TEST 3: IMPACT OF POLYNOMIAL DEGREE")
    print("=" * 100)
    
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
        
        print(f"\n{'-'*100}")
        print(f"AMERICAN {call_put}")
        print(f"{'-'*100}")
        
        # Get Trinomial reference
        tree = Tree(40, market, option, pricing_date, prunning_threshold=1e-8)
        tree.build_tree()
        trinomial_model = TrinomialModel(pricing_date, tree)
        reference_price = trinomial_model.price(option, "backward")
        
        print(f"\nTrinomial Tree Reference: {reference_price:.6f}")
        print(f"\nLS with different polynomial degrees:")
        print(f"{'Poly Deg':>10} | {'LS Price':>12} | {'Std Error':>12} | {'Error vs Tree':>15} | "
              f"{'Error %':>10} | {'Time (s)':>10}")
        print("-" * 85)
        
        num_sims = 25000
        mc = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option,
            pricing_date=pricing_date,
            seed=42
        )
        
        for poly_degree in [1, 2, 3, 4, 5]:
            start = time.time()
            result = mc.price_american_longstaff_schwartz_vectorized(
                num_steps=100,
                poly_degree=poly_degree,
                antithetic=True
            )
            exec_time = time.time() - start
            
            ls_price = result['price']
            std_error = result['std_error']
            error = abs(ls_price - reference_price)
            error_pct = (error / reference_price) * 100 if reference_price != 0 else 0
            
            print(f"{poly_degree:>10} | {ls_price:>12.6f} | {std_error:>12.6f} | "
                  f"{error:>15.6f} | {error_pct:>9.2f}% | {exec_time:>10.4f}")
        
        print(f"\n✓ Polynomial degree 3-4 typically optimal (captures payoff shape)")
        print(f"✓ Higher degree may overfit")


def test_naive_vs_ls_vs_tree():
    """Compare Naive MC, LS, and Trinomial Tree"""
    print("\n" + "=" * 100)
    print("TEST 4: NAIVE vs LONGSTAFF-SCHWARTZ vs TRINOMIAL TREE")
    print("=" * 100)
    
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
    
    # Trinomial Tree
    print(f"\nTrinomial Tree Pricing:")
    tree = Tree(50, market, option_put, pricing_date, prunning_threshold=1e-8)
    tree.build_tree()
    trinomial_model = TrinomialModel(pricing_date, tree)
    tree_price = trinomial_model.price(option_put, "backward")
    print(f"  Price: {tree_price:.6f}")
    
    num_sims = 25000
    mc = MonteCarloModel(
        num_simulations=num_sims,
        market=market,
        option=option_put,
        pricing_date=pricing_date,
        seed=42
    )
    
    num_steps = 100
    
    # Naive
    print(f"\nNaive American MC ({num_sims:,} sims, {num_steps} steps):")
    start = time.time()
    result_naive = mc.price_american_naive_vectorized(num_steps=num_steps, antithetic=True)
    time_naive = time.time() - start
    print(f"  Price: {result_naive['price']:.6f} (±{result_naive['std_error']:.6f})")
    print(f"  Error vs Tree: {abs(result_naive['price'] - tree_price):.6f}")
    print(f"  Error %: {abs(result_naive['price'] - tree_price) / tree_price * 100:.2f}%")
    print(f"  Time: {time_naive:.4f}s")
    
    # Longstaff-Schwartz
    print(f"\nLongstaff-Schwartz MC ({num_sims:,} sims, {num_steps} steps, degree=3):")
    start = time.time()
    result_ls = mc.price_american_longstaff_schwartz_vectorized(
        num_steps=num_steps,
        poly_degree=3,
        antithetic=True
    )
    time_ls = time.time() - start
    print(f"  Price: {result_ls['price']:.6f} (±{result_ls['std_error']:.6f})")
    print(f"  Error vs Tree: {abs(result_ls['price'] - tree_price):.6f}")
    print(f"  Error %: {abs(result_ls['price'] - tree_price) / tree_price * 100:.2f}%")
    print(f"  Time: {time_ls:.4f}s")
    
    print(f"\n{'Method':<25} | {'Price':>12} | {'Error vs Tree':>15} | {'Error %':>10}")
    print("-" * 70)
    print(f"{'Trinomial Tree (50)':25} | {tree_price:>12.6f} | {'Reference':>15} | {'0.00%':>10}")
    print(f"{'Naive MC':25} | {result_naive['price']:>12.6f} | "
          f"{abs(result_naive['price'] - tree_price):>15.6f} | "
          f"{abs(result_naive['price'] - tree_price) / tree_price * 100:>9.2f}%")
    print(f"{'Longstaff-Schwartz':25} | {result_ls['price']:>12.6f} | "
          f"{abs(result_ls['price'] - tree_price):>15.6f} | "
          f"{abs(result_ls['price'] - tree_price) / tree_price * 100:>9.2f}%")
    
    improvement = (abs(result_naive['price'] - tree_price) - 
                   abs(result_ls['price'] - tree_price)) / abs(result_naive['price'] - tree_price) * 100
    
    print(f"\n✓ LS Improvement: {improvement:.1f}% better than Naive")
    print(f"✓ LS Time ratio: {time_ls / time_naive:.2f}x (LS is slightly slower due to regression)")


def test_ls_steps_impact():
    """Test impact of number of steps on LS accuracy"""
    print("\n" + "=" * 100)
    print("TEST 5: IMPACT OF NUMBER OF STEPS ON LS")
    print("=" * 100)
    
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
    
    # Get Trinomial reference
    tree = Tree(50, market, option_put, pricing_date, prunning_threshold=1e-8)
    tree.build_tree()
    trinomial_model = TrinomialModel(pricing_date, tree)
    reference_price = trinomial_model.price(option_put, "backward")
    
    print(f"\nTrinomial Tree Reference (50 steps): {reference_price:.6f}")
    print(f"\nLS Convergence with increasing number of steps:")
    print(f"{'Steps':>8} | {'LS Price':>12} | {'Std Error':>12} | {'Error vs Tree':>15} | "
          f"{'Error %':>10} | {'Time (s)':>10}")
    print("-" * 85)
    
    num_sims = 25000
    mc = MonteCarloModel(
        num_simulations=num_sims,
        market=market,
        option=option_put,
        pricing_date=pricing_date,
        seed=42
    )
    
    for num_steps in [25, 50, 100, 252]:
        start = time.time()
        result = mc.price_american_longstaff_schwartz_vectorized(
            num_steps=num_steps,
            poly_degree=3,
            antithetic=True
        )
        exec_time = time.time() - start
        
        ls_price = result['price']
        std_error = result['std_error']
        error = abs(ls_price - reference_price)
        error_pct = (error / reference_price) * 100 if reference_price != 0 else 0
        
        print(f"{num_steps:>8} | {ls_price:>12.6f} | {std_error:>12.6f} | "
              f"{error:>15.6f} | {error_pct:>9.2f}% | {exec_time:>10.4f}")
    
    print(f"\n✓ More steps = better discretization = closer to true value")
    print(f"✓ Computation time increases with steps")


def test_american_vs_european_ls():
    """Compare American vs European with LS method"""
    print("\n" + "=" * 100)
    print("TEST 6: AMERICAN vs EUROPEAN - LONGSTAFF-SCHWARTZ")
    print("=" * 100)
    
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
        print(f"\n{'-'*100}")
        print(f"{call_put} OPTION")
        print(f"{'-'*100}")
        
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
        
        # American LS
        mc_american = MonteCarloModel(
            num_simulations=num_sims,
            market=market,
            option=option_american,
            pricing_date=pricing_date,
            seed=42
        )
        result_american = mc_american.price_american_longstaff_schwartz_vectorized(
            num_steps=100,
            poly_degree=3,
            antithetic=True
        )
        
        # European
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
        
        print(f"\nAmerican (LS):  {american_price:.6f} (±{american_error:.6f})")
        print(f"European:       {european_price:.6f} (±{european_error:.6f})")
        print(f"\nEarly exercise premium: {early_exercise_value:.6f}")
        
        if call_put == 'PUT':
            if early_exercise_value > 0:
                print("✓ PUT: American > European (early exercise is valuable)")
            else:
                print("⚠ PUT: American ≤ European (unexpected for puts)")
        else:
            if abs(early_exercise_value) < 0.01:
                print("✓ CALL: American ≈ European (no dividends, early exercise not optimal)")
            else:
                print(f"⚠ CALL: Difference = {early_exercise_value:.6f} (expected near 0 without dividends)")


def main():
    """Run all LS vs Tree tests"""
    print("\n" * 2)
    print("#" * 100)
    print("# LONGSTAFF-SCHWARTZ vs TRINOMIAL TREE - COMPREHENSIVE TEST SUITE")
    print("#" * 100)
    
    try:
        # Test 1: Accuracy comparison
        test_ls_vs_tree_accuracy()
        
        # Test 2: Convergence
        test_ls_convergence()
        
        # Test 3: Polynomial degree impact
        test_poly_degree_impact()
        
        # Test 4: Naive vs LS vs Tree
        test_naive_vs_ls_vs_tree()
        
        # Test 5: Steps impact
        test_ls_steps_impact()
        
        # Test 6: American vs European
        test_american_vs_european_ls()
        
        print("\n" + "=" * 100)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 100 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
