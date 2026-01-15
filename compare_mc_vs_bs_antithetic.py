"""
Compare Monte Carlo Pricing:
- MC sans antithetic variates
- MC avec antithetic variates
- Black-Scholes (référence analytique)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
from PriceurBS import Call, Put


def compare_pricing(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, num_sims=10000):
    """
    Compare MC (avec/sans antithetic) vs Black-Scholes
    
    Parameters:
    -----------
    S0 : float - Spot price
    K : float - Strike price
    T : float - Time to maturity (years)
    r : float - Risk-free rate
    sigma : float - Volatility
    num_sims : int - Number of MC simulations
    """
    
    # Dates
    pricing_date = date(2025, 1, 15)
    mat_date = pricing_date + timedelta(days=int(T * 365))
    
    # Market parameters
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Test CALL
    print("=" * 80)
    print(f"EUROPEAN CALL OPTION")
    print(f"S0={S0}, K={K}, T={T}yr, r={r}, σ={sigma}")
    print("=" * 80)
    
    # Black-Scholes
    bs_call = Call(S0, K, r, T, sigma)
    bs_price_call = bs_call.price()
    print(f"\n{'Method':<30} {'Price':>12} {'Std Error':>12} {'Error %':>12}")
    print("-" * 80)
    print(f"{'Black-Scholes (ref)':30} {bs_price_call:12.6f} {'':>12} {'':>12}")
    
    # MC sans antithetic
    option_call = OptionTrade(mat_date, 'CALL', 'EUROPEAN', K)
    mc_model_no_anti = MonteCarloModel(num_sims, market, option_call, pricing_date, seed=42)
    result_no_anti = mc_model_no_anti.price_european(antithetic=False)
    mc_price_no_anti = result_no_anti['price']
    mc_std_no_anti = result_no_anti['std_error']
    error_no_anti = abs(mc_price_no_anti - bs_price_call) / bs_price_call * 100
    
    print(f"{'MC sans antithetic':30} {mc_price_no_anti:12.6f} {mc_std_no_anti:12.6f} {error_no_anti:11.4f}%")
    
    # MC avec antithetic
    mc_model_with_anti = MonteCarloModel(num_sims, market, option_call, pricing_date, seed=42)
    result_with_anti = mc_model_with_anti.price_european(antithetic=True)
    mc_price_with_anti = result_with_anti['price']
    mc_std_with_anti = result_with_anti['std_error']
    error_with_anti = abs(mc_price_with_anti - bs_price_call) / bs_price_call * 100
    
    print(f"{'MC avec antithetic':30} {mc_price_with_anti:12.6f} {mc_std_with_anti:12.6f} {error_with_anti:11.4f}%")
    
    # Test PUT
    print("\n" + "=" * 80)
    print(f"EUROPEAN PUT OPTION")
    print(f"S0={S0}, K={K}, T={T}yr, r={r}, σ={sigma}")
    print("=" * 80)
    
    # Black-Scholes
    bs_put = Put(S0, K, r, T, sigma)
    bs_price_put = bs_put.price()
    print(f"\n{'Method':<30} {'Price':>12} {'Std Error':>12} {'Error %':>12}")
    print("-" * 80)
    print(f"{'Black-Scholes (ref)':30} {bs_price_put:12.6f} {'':>12} {'':>12}")
    
    # MC sans antithetic
    option_put = OptionTrade(mat_date, 'PUT', 'EUROPEAN', K)
    mc_model_put_no_anti = MonteCarloModel(num_sims, market, option_put, pricing_date, seed=42)
    result_put_no_anti = mc_model_put_no_anti.price_european(antithetic=False)
    mc_price_put_no_anti = result_put_no_anti['price']
    mc_std_put_no_anti = result_put_no_anti['std_error']
    error_put_no_anti = abs(mc_price_put_no_anti - bs_price_put) / bs_price_put * 100
    
    print(f"{'MC sans antithetic':30} {mc_price_put_no_anti:12.6f} {mc_std_put_no_anti:12.6f} {error_put_no_anti:11.4f}%")
    
    # MC avec antithetic
    mc_model_put_with_anti = MonteCarloModel(num_sims, market, option_put, pricing_date, seed=42)
    result_put_with_anti = mc_model_put_with_anti.price_european(antithetic=True)
    mc_price_put_with_anti = result_put_with_anti['price']
    mc_std_put_with_anti = result_put_with_anti['std_error']
    error_put_with_anti = abs(mc_price_put_with_anti - bs_price_put) / bs_price_put * 100
    
    print(f"{'MC avec antithetic':30} {mc_price_put_with_anti:12.6f} {mc_std_put_with_anti:12.6f} {error_put_with_anti:11.4f}%")
    
    return {
        'call': {
            'bs': bs_price_call,
            'mc_no_anti': mc_price_no_anti,
            'mc_no_anti_std': mc_std_no_anti,
            'mc_with_anti': mc_price_with_anti,
            'mc_with_anti_std': mc_std_with_anti,
        },
        'put': {
            'bs': bs_price_put,
            'mc_no_anti': mc_price_put_no_anti,
            'mc_no_anti_std': mc_std_put_no_anti,
            'mc_with_anti': mc_price_put_with_anti,
            'mc_with_anti_std': mc_std_put_with_anti,
        }
    }


def convergence_study(S0=100, K=100, T=1.0, r=0.05, sigma=0.2):
    """
    Étudier la convergence en fonction du nombre de simulations
    """
    print("\n" + "=" * 80)
    print("ÉTUDE DE CONVERGENCE")
    print("=" * 80)
    
    pricing_date = date(2025, 1, 15)
    mat_date = pricing_date + timedelta(days=int(T * 365))
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    # Référence BS
    bs_call = Call(S0, K, r, T, sigma)
    bs_price = bs_call.price()
    
    # Test avec différents nombres de simulations
    num_sims_list = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    results_no_anti = []
    results_with_anti = []
    
    print(f"\n{'Num Sims':>10} {'MC no anti':>15} {'Error %':>12} {'MC antithetic':>15} {'Error %':>12}")
    print("-" * 80)
    
    for num_sims in num_sims_list:
        option = OptionTrade(mat_date, 'CALL', 'EUROPEAN', K)
        
        # Sans antithetic
        mc_no_anti = MonteCarloModel(num_sims, market, option, pricing_date, seed=42)
        res_no_anti = mc_no_anti.price_european(antithetic=False)
        price_no_anti = res_no_anti['price']
        error_no_anti = abs(price_no_anti - bs_price) / bs_price * 100
        results_no_anti.append(error_no_anti)
        
        # Avec antithetic
        mc_with_anti = MonteCarloModel(num_sims, market, option, pricing_date, seed=42)
        res_with_anti = mc_with_anti.price_european(antithetic=True)
        price_with_anti = res_with_anti['price']
        error_with_anti = abs(price_with_anti - bs_price) / bs_price * 100
        results_with_anti.append(error_with_anti)
        
        print(f"{num_sims:10d} {price_no_anti:15.6f} {error_no_anti:11.4f}% {price_with_anti:15.6f} {error_with_anti:11.4f}%")
    
    # Plot convergence
    plt.figure(figsize=(12, 6))
    plt.loglog(num_sims_list, results_no_anti, 'o-', label='MC sans antithetic', linewidth=2, markersize=8)
    plt.loglog(num_sims_list, results_with_anti, 's-', label='MC avec antithetic', linewidth=2, markersize=8)
    
    # Référence 1/sqrt(N)
    ref_convergence = np.array([100 / np.sqrt(n) for n in num_sims_list])
    plt.loglog(num_sims_list, ref_convergence, '--', label='Référence 1/√N', linewidth=2, color='gray', alpha=0.7)
    
    plt.xlabel('Nombre de simulations (N)', fontsize=12)
    plt.ylabel('Erreur relative (%)', fontsize=12)
    plt.title('Convergence Monte Carlo - CALL ATM', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('convergence_antithetic.png', dpi=150)
    print("\n✓ Graphique sauvegardé: convergence_antithetic.png")
    plt.show()


def variance_reduction_analysis(S0=100, K=100, T=1.0, r=0.05, sigma=0.2, num_sims=10000):
    """
    Analyser la réduction de variance avec antithetic variates
    """
    print("\n" + "=" * 80)
    print("ANALYSE DE RÉDUCTION DE VARIANCE")
    print("=" * 80)
    
    pricing_date = date(2025, 1, 15)
    mat_date = pricing_date + timedelta(days=int(T * 365))
    
    market = Market(
        underlying=S0,
        vol=sigma,
        rate=r,
        div_a=0.0,
        ex_div_date=None
    )
    
    option = OptionTrade(mat_date, 'CALL', 'EUROPEAN', K)
    
    # Sans antithetic
    mc_no_anti = MonteCarloModel(num_sims, market, option, pricing_date, seed=42)
    res_no_anti = mc_no_anti.price_european(antithetic=False)
    payoffs_no_anti = np.array(res_no_anti['payoffs'])
    
    # Avec antithetic
    mc_with_anti = MonteCarloModel(num_sims, market, option, pricing_date, seed=42)
    res_with_anti = mc_with_anti.price_european(antithetic=True)
    payoffs_with_anti = np.array(res_with_anti['payoffs'])
    
    var_no_anti = np.var(payoffs_no_anti)
    var_with_anti = np.var(payoffs_with_anti)
    reduction = (1 - var_with_anti / var_no_anti) * 100
    
    print(f"\nVariance sans antithetic: {var_no_anti:.8f}")
    print(f"Variance avec antithetic: {var_with_anti:.8f}")
    print(f"Réduction de variance:    {reduction:.2f}%")
    print(f"Facteur d'amélioration:   {var_no_anti / var_with_anti:.2f}x")
    
    # Histogramme
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(payoffs_no_anti, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title(f'Distribution payoffs\nSans antithetic (σ²={var_no_anti:.6f})', fontsize=12)
    axes[0].set_xlabel('Payoff actualisé')
    axes[0].set_ylabel('Fréquence')
    axes[0].grid(alpha=0.3)
    
    axes[1].hist(payoffs_with_anti, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title(f'Distribution payoffs\nAvec antithetic (σ²={var_with_anti:.6f})', fontsize=12)
    axes[1].set_xlabel('Payoff actualisé')
    axes[1].set_ylabel('Fréquence')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('variance_reduction.png', dpi=150)
    print("\n✓ Graphique sauvegardé: variance_reduction.png")
    plt.show()


if __name__ == '__main__':
    # Comparaison simple
    results = compare_pricing(S0=100, K=0.1, T=1.0, r=0.05, sigma=0.2, num_sims=10000)
    
    # Étude de convergence
    convergence_study(S0=100, K=0.1, T=1.0, r=0.05, sigma=0.2)
    
    # Analyse variance
    variance_reduction_analysis(S0=100, K=0.1, T=1.0, r=0.05, sigma=0.2, num_sims=10000)