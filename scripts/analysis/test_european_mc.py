"""
Test European Option Pricing using Monte Carlo
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from datetime import date, timedelta
from src.instruments.market import Market
from src.instruments.option_trade import OptionTrade
from src.pricing.monte_carlo_model import MonteCarloModel
import numpy as np


def test_european_call():
    """Test pricing of a European Call option"""
    
    # Market parameters
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)  # 6 months
    
    market = Market(
        underlying=100.0,      # S0 = 100
        vol=0.20,              # 20% volatility
        rate=0.05,             # 5% risk-free rate
        div_a=0.0,             # No dividend
        ex_div_date=None
    )
    
    # European Call option
    option = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=100.0                # Strike = 100 (ATM)
    )
    
    # Monte Carlo simulation
    print("=" * 60)
    print("European Call Option Pricing - Monte Carlo")
    print("=" * 60)
    print(f"Underlying:    {market.underlying}")
    print(f"Strike:        {option.strike}")
    print(f"Volatility:    {market.vol * 100}%")
    print(f"Rate:          {market.rate * 100}%")
    print(f"Time to Mat:   {(maturity_date - pricing_date).days / 365.0:.4f} years")
    print("-" * 60)
    
    # Test with different number of simulations
    for n_sims in [1000, 10000, 100000]:
        mc_model = MonteCarloModel(
            num_simulations=n_sims,
            market=market,
            option=option,
            pricing_date=pricing_date,
            seed=42
        )
        
        result = mc_model.price_european()
        
        print(f"\nSimulations: {n_sims:,}")
        print(f"  MC Price:    {result['price']:.6f}")
        print(f"  Std Error:   {result['std_error']:.6f}")
        print(f"  95% CI:      [{result['price'] - 1.96*result['std_error']:.6f}, "
              f"{result['price'] + 1.96*result['std_error']:.6f}]")


def test_european_put():
    """Test pricing of a European Put option"""
    
    # Market parameters
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)  # 6 months
    
    market = Market(
        underlying=100.0,
        vol=0.20,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    # European Put option
    option = OptionTrade(
        mat=maturity_date,
        call_put='PUT',
        ex='EUROPEAN',
        k=100.0                # Strike = 100 (ATM)
    )
    
    print("\n" + "=" * 60)
    print("European Put Option Pricing - Monte Carlo")
    print("=" * 60)
    print(f"Underlying:    {market.underlying}")
    print(f"Strike:        {option.strike}")
    print(f"Volatility:    {market.vol * 100}%")
    print(f"Rate:          {market.rate * 100}%")
    print(f"Time to Mat:   {(maturity_date - pricing_date).days / 365.0:.4f} years")
    print("-" * 60)
    
    mc_model = MonteCarloModel(
        num_simulations=100000,
        market=market,
        option=option,
        pricing_date=pricing_date,
        seed=42
    )
    
    result = mc_model.price_european()
    
    print(f"\nSimulations: {mc_model.num_simulations:,}")
    print(f"  MC Price:    {result['price']:.6f}")
    print(f"  Std Error:   {result['std_error']:.6f}")


def test_call_put_parity():
    """Verify Put-Call Parity: C - P = S0 - K*exp(-rT)"""
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    
    market = Market(
        underlying=100.0,
        vol=0.20,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    strike = 100.0
    T = (maturity_date - pricing_date).days / 365.0
    
    # Call option
    call_option = OptionTrade(mat=maturity_date, call_put='CALL', ex='EUROPEAN', k=strike)
    mc_call = MonteCarloModel(100000, market, call_option, pricing_date, seed=42)
    call_price = mc_call.price_european()['price']
    
    # Put option
    put_option = OptionTrade(mat=maturity_date, call_put='PUT', ex='EUROPEAN', k=strike)
    mc_put = MonteCarloModel(100000, market, put_option, pricing_date, seed=42)
    put_price = mc_put.price_european()['price']
    
    # Put-Call Parity
    lhs = call_price - put_price
    rhs = market.underlying - strike * np.exp(-market.rate * T)
    
    print("\n" + "=" * 60)
    print("Put-Call Parity Verification")
    print("=" * 60)
    print(f"Call Price:           {call_price:.6f}")
    print(f"Put Price:            {put_price:.6f}")
    print(f"C - P:                {lhs:.6f}")
    print(f"S0 - K*exp(-rT):      {rhs:.6f}")
    print(f"Difference:           {abs(lhs - rhs):.6f}")
    print("-" * 60)


if __name__ == "__main__":
    test_european_call()
    test_european_put()
    test_call_put_parity()
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
