import random
import numpy as np
from datetime import date
from .market import Market
from .option_trade import OptionTrade


class MonteCarloModel:
    def __init__(self, num_simulations: int, market: Market, option: OptionTrade, 
                 pricing_date: date, seed=None):
        """
        Monte Carlo Model for European Option Pricing
        
        Parameters:
        -----------
        num_simulations : int
            Number of Monte Carlo paths (N)
        market : Market
            Market parameters (underlying, vol, rate, div)
        option : OptionTrade
            Option parameters (strike, maturity, call/put)
        pricing_date : date
            Valuation date
        seed : int, optional
            Random seed for reproducibility
        """
        self.num_simulations = num_simulations
        self.market = market
        self.option = option
        self.pricing_date = pricing_date
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def price_european(self, antithetic=True) -> dict:
        """
        Price a European option using Monte Carlo simulation
        
        Process:
        1. Repeat N times:
           a. Draw a random number between 0 and 1
           b. Convert it to a normal draw
           c. Convert it to a Brownian draw
           d. Deduce the value of the underlying at T
           e. Calculate the value of the option at T
           f. Discount to today, this is one result
        2. Average the N results: this is the price
        
        Returns:
        --------
        dict: {'price': float, 'std_error': float, 'payoffs': list}
        """
        # Calculate time to maturity in years
        T = (self.option.mat_date - self.pricing_date).days / 365.0
        
        if T <= 0:
            # Option has expired
            return {
                'price': 0.0,
                'std_error': 0.0,
                'payoffs': []
            }
        
        # Market parameters
        S0 = self.market.underlying
        sigma = self.market.vol
        r = self.market.rate
        
        # Adjust for dividends (simple approach: reduce drift)
        # If there's a dividend before maturity, adjust the forward
        q = 0.0  # dividend yield (simplified)
        if self.market.ex_div_date and self.market.ex_div_date < self.option.mat_date:
            # Approximate dividend yield
            q = self.market.div_a / S0 if S0 > 0 else 0.0
        
        discounted_payoffs = []
        if antithetic:
            num_paths = self.num_simulations // 2
        else:
            num_paths = self.num_simulations
        for i in range(num_paths):
            # Step 1: Draw a random number between 0 and 1
            
            u = np.random.uniform(0, 1)
            
            # Step 2: Convert it to a normal draw N(0,1) using inverse transform
            # Box-Muller transform is used internally by numpy
            Z = np.random.standard_normal()
            
            # Step 3: Convert it to a Brownian draw W(T)
            # W(T) ~ N(0, T)
            W_T = Z * np.sqrt(T)
            
            # Step 4: Deduce the value of the underlying at T
            # Using Black-Scholes formula: S(T) = S0 * exp((r - q - 0.5*sigma^2)*T + sigma*W(T))
            S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)
            
            if antithetic:
                # Generate antithetic path
                S_T_antithetic = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * (-W_T))
                
                # Step 5: Calculate the value of the option at T (payoff)
                payoff_antithetic = self.option.pay_off(S_T_antithetic)
                
                # Step 6: Discount to today
                discounted_payoff_antithetic = payoff_antithetic * np.exp(-r * T)
                discounted_payoffs.append(discounted_payoff_antithetic)

            # Step 5: Calculate the value of the option at T (payoff)
            payoff = self.option.pay_off(S_T)
            
            # Step 6: Discount to today
            discounted_payoff = payoff * np.exp(-r * T)
            discounted_payoffs.append(discounted_payoff)
        
        # Step 7: Average the N results - this is the price
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)
        
        return {
            'price': price,
            'std_error': std_error,
            'payoffs': discounted_payoffs
        }
    
    def run_simulation(self, model_input):
        """Legacy method for backward compatibility"""
        results = []
        for _ in range(self.num_simulations):
            result = self._simulate(model_input)
            results.append(result)
        return results

    def _simulate(self, model_input):
        """Legacy method for backward compatibility"""
        import random
        simulated_value = model_input + random.uniform(-1, 1)
        return simulated_value
    
    def calculate_average(self, results):
        return sum(results) / len(results) if results else 0
    
    def calculate_variance(self, results):
        avg = self.calculate_average(results)
        return sum((x - avg) ** 2 for x in results) / len(results) if results else 0
    