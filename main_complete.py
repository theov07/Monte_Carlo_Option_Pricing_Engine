"""
Complete Monte Carlo American Pricing Test
- European option pricing (scalar vs vectorized)
- American option pricing (scalar vs vectorized)
- Comparison with Black-Scholes
- Comparison with Trinomial Tree
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from scipy.stats import norm

from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from src_trinomial.tree import Tree
from src_trinomial.trinomial_model import TrinomialModel
from PriceurBS import Call, Put


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes price"""
    if T <= 0:
        if option_type.lower() == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price


def test_european_options():
    """Test European option pricing - Scalar vs Vectorized"""
    print("\n" + "="*80)
    print("STEP 1: EUROPEAN OPTIONS - SCALAR VS VECTORIZED")
    print("="*80)
    
    # Setup
    pricing_date = date(2025, 1, 15)
    maturity_date = date(2026, 1, 15)  # 1 year
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    
    T = (maturity_date - pricing_date).days / 365.0
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Black-Scholes reference
    bs_call = black_scholes_price(S0, K, T, r, sigma, 'call')
    bs_put = black_scholes_price(S0, K, T, r, sigma, 'put')
    
    print(f"\nBlack-Scholes Reference (Analytical):")
    print(f"  Call: {bs_call:.6f}")
    print(f"  Put:  {bs_put:.6f}")
    
    # Test with different number of simulations
    num_sims_list = [1000, 10000, 100000]
    
    for num_sims in num_sims_list:
        print(f"\n{'-'*80}")
        print(f"Number of Simulations: {num_sims:,}")
        print(f"{'-'*80}")
        
        # CALL option
        call_option = OptionTrade(
            mat=maturity_date,
            call_put='CALL',
            ex='EUROPEAN',
            k=K
        )
        
        mc = MonteCarloModel(num_sims, market, call_option, pricing_date, seed=42)
        comparison = mc.compare_scalar_vs_vectorized(antithetic=True)
        
        print(f"\nCALL OPTION (with antithetic):")
        print(f"  Scalar:        Price={comparison['scalar']['price']:.6f} | "
              f"StdErr={comparison['scalar']['std_error']:.6f} | "
              f"Time={comparison['scalar']['time']:.4f}s")
        print(f"  Vectorized:    Price={comparison['vectorized']['price']:.6f} | "
              f"StdErr={comparison['vectorized']['std_error']:.6f} | "
              f"Time={comparison['vectorized']['time']:.4f}s")
        print(f"  Speedup:       {comparison['speedup']:.2f}x")
        print(f"  Price diff:    {comparison['price_difference']:.2e}")
        print(f"  Error vs BS:   {abs(comparison['vectorized']['price'] - bs_call):.6f}")
        
        # PUT option
        put_option = OptionTrade(
            mat=maturity_date,
            call_put='PUT',
            ex='EUROPEAN',
            k=K
        )
        
        mc = MonteCarloModel(num_sims, market, put_option, pricing_date, seed=42)
        comparison = mc.compare_scalar_vs_vectorized(antithetic=True)
        
        print(f"\nPUT OPTION (with antithetic):")
        print(f"  Scalar:        Price={comparison['scalar']['price']:.6f} | "
              f"StdErr={comparison['scalar']['std_error']:.6f} | "
              f"Time={comparison['scalar']['time']:.4f}s")
        print(f"  Vectorized:    Price={comparison['vectorized']['price']:.6f} | "
              f"StdErr={comparison['vectorized']['std_error']:.6f} | "
              f"Time={comparison['vectorized']['time']:.4f}s")
        print(f"  Speedup:       {comparison['speedup']:.2f}x")
        print(f"  Price diff:    {comparison['price_difference']:.2e}")
        print(f"  Error vs BS:   {abs(comparison['vectorized']['price'] - bs_put):.6f}")


def test_american_options():
    """Test American option pricing - Scalar vs Vectorized"""
    print("\n" + "="*80)
    print("STEP 2: AMERICAN OPTIONS - SCALAR VS VECTORIZED (NAIVE)")
    print("="*80)
    
    # Setup
    pricing_date = date(2025, 1, 15)
    maturity_date = date(2025, 7, 15)  # 6 months
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    
    T = (maturity_date - pricing_date).days / 365.0
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Test with different number of steps and simulations
    num_sims_list = [1000, 10000]
    num_steps = 126  # ~daily steps for 6 months
    
    for num_sims in num_sims_list:
        print(f"\n{'-'*80}")
        print(f"Number of Simulations: {num_sims:,} | Number of Steps: {num_steps}")
        print(f"{'-'*80}")
        
        # AMERICAN PUT (usually more valuable than European)
        put_option = OptionTrade(
            mat=maturity_date,
            call_put='PUT',
            ex='AMERICAN',
            k=K
        )
        
        mc = MonteCarloModel(num_sims, market, put_option, pricing_date, seed=42)
        
        # Scalar American
        print(f"\nAMERICAN PUT (Scalar):")
        start = time.time()
        result_scalar = mc.price_american_naive(num_steps=num_steps, antithetic=True)
        time_scalar = time.time() - start
        print(f"  Price:    {result_scalar['price']:.6f}")
        print(f"  StdErr:   {result_scalar['std_error']:.6f}")
        print(f"  Time:     {time_scalar:.4f}s")
        
        # Vectorized American
        print(f"\nAMERICAN PUT (Vectorized):")
        start = time.time()
        result_vector = mc.price_american_naive_vectorized(num_steps=num_steps, antithetic=True)
        time_vector = time.time() - start
        print(f"  Price:    {result_vector['price']:.6f}")
        print(f"  StdErr:   {result_vector['std_error']:.6f}")
        print(f"  Time:     {time_vector:.4f}s")
        
        # Comparison
        print(f"\nComparison:")
        speedup = time_scalar / time_vector if time_vector > 0 else float('inf')
        price_diff = abs(result_scalar['price'] - result_vector['price'])
        print(f"  Speedup:           {speedup:.2f}x")
        print(f"  Price difference:  {price_diff:.2e}")
        
        # European PUT for reference
        european_put = OptionTrade(
            mat=maturity_date,
            call_put='PUT',
            ex='EUROPEAN',
            k=K
        )
        mc_euro = MonteCarloModel(num_sims, market, european_put, pricing_date, seed=42)
        result_euro = mc_euro.price_european_vectorized(antithetic=True)
        
        print(f"\nComparison with European PUT:")
        print(f"  European PUT:  {result_euro['price']:.6f}")
        print(f"  American PUT:  {result_vector['price']:.6f}")
        print(f"  Early exercise value: {result_vector['price'] - result_euro['price']:.6f}")


def test_convergence_american():
    """Test convergence of American pricing with more simulations"""
    print("\n" + "="*80)
    print("STEP 3: CONVERGENCE STUDY - AMERICAN OPTION")
    print("="*80)
    
    pricing_date = date(2025, 1, 15)
    maturity_date = date(2025, 7, 15)  # 6 months
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    put_option = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='AMERICAN',
        k=K
    )
    
    # Test with increasing simulations
    num_sims_list = [100, 500, 1000, 5000, 10000]
    num_steps = 126
    
    prices = []
    std_errs = []
    
    print(f"\nConvergence with increasing simulations (num_steps={num_steps}):")
    print(f"{'N Sims':>10} | {'Price':>12} | {'Std Error':>12} | {'95% CI':>20}")
    print("-" * 60)
    
    for num_sims in num_sims_list:
        mc = MonteCarloModel(num_sims, market, put_option, pricing_date, seed=None)
        result = mc.price_american_naive_vectorized(num_steps=num_steps, antithetic=True)
        
        prices.append(result['price'])
        std_errs.append(result['std_error'])
        
        ci_lower = result['price'] - 1.96 * result['std_error']
        ci_upper = result['price'] + 1.96 * result['std_error']
        
        print(f"{num_sims:>10,} | {result['price']:>12.6f} | "
              f"{result['std_error']:>12.6f} | [{ci_lower:.6f}, {ci_upper:.6f}]")
    
    print("\n✓ Price should converge as N increases")
    print(f"✓ Standard error should decrease as 1/√N")


def test_vs_trinomial():
    """Compare American MC with Trinomial Tree"""
    print("\n" + "="*80)
    print("STEP 4: COMPARISON - AMERICAN MC vs TRINOMIAL TREE")
    print("="*80)
    
    pricing_date = date(2025, 9, 1)
    maturity_date = date(2026, 9, 1)  # 1 year
    
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.30
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # American PUT
    put_option = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='AMERICAN',
        k=K
    )
    
    print(f"\nParameters: S0={S0}, K={K}, T=1yr, r={r}, σ={sigma}")
    print(f"{'-'*80}")
    
    # Trinomial Tree (reference)
    print("\nTrinomial Tree Pricing:")
    for num_steps in [20, 50]:
        tree = Tree(num_steps, market, put_option, pricing_date)
        tree.build_tree()
        trinomial_model = TrinomialModel(pricing_date, tree)
        tree_price = trinomial_model.price(put_option, "backward")
        print(f"  Steps={num_steps:3d}: Price = {tree_price:.6f}")
    
    # Monte Carlo
    print("\nMonte Carlo Pricing:")
    for num_sims in [10000, 50000]:
        mc = MonteCarloModel(num_sims, market, put_option, pricing_date, seed=42)
        result = mc.price_american_naive_vectorized(num_steps=100, antithetic=True)
        print(f"  Sims={num_sims:5d}: Price = {result['price']:.6f} (±{result['std_error']:.6f})")


def main():
    """Run all tests"""
    print("\n" * 2)
    print("#" * 80)
    print("# COMPLETE MONTE CARLO AMERICAN PRICING TEST SUITE")
    print("#" * 80)
    
    # Test 1: European options
    test_european_options()
    
    # Test 2: American options
    test_american_options()
    
    # Test 3: Convergence
    test_convergence_american()
    
    # Test 4: Comparison with Trinomial
    test_vs_trinomial()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
