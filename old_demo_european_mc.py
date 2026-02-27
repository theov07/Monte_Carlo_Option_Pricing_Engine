"""
Démonstration détaillée du pricing Monte Carlo pour options européennes
Step-by-Step European MC Option Pricing

Ce fichier illustre en détail les 6 étapes du pricing Monte Carlo :
1. Tirer un nombre aléatoire entre 0 et 1
2. Le convertir en tirage normal
3. Le convertir en tirage brownien
4. En déduire la valeur du sous-jacent à T
5. Calculer la valeur de l'option à T
6. Actualiser à aujourd'hui
"""

from datetime import date
from src.market import Market
from src.option_trade import OptionTrade
from src.monte_carlo_model import MonteCarloModel
import numpy as np


def demonstrate_one_path():
    """Démonstration détaillée d'une seule simulation Monte Carlo"""
    
    print("=" * 70)
    print("DÉMONSTRATION D'UNE SIMULATION MONTE CARLO - OPTION EUROPÉENNE")
    print("=" * 70)
    
    # Paramètres
    S0 = 100.0          # Prix initial du sous-jacent
    K = 100.0           # Strike
    r = 0.05            # Taux sans risque (5%)
    sigma = 0.20        # Volatilité (20%)
    T = 0.5             # Maturité (6 mois)
    
    print(f"\nParamètres:")
    print(f"  S0 (prix initial):     {S0}")
    print(f"  K (strike):            {K}")
    print(f"  r (taux):              {r * 100}%")
    print(f"  σ (volatilité):        {sigma * 100}%")
    print(f"  T (maturité):          {T} ans")
    
    print("\n" + "-" * 70)
    print("ÉTAPES DE LA SIMULATION")
    print("-" * 70)
    
    # Étape 1 : Tirer un nombre aléatoire entre 0 et 1
    np.random.seed(42)
    u = np.random.uniform(0, 1)
    print(f"\n1. Tirer un nombre aléatoire U ~ Uniform(0,1)")
    print(f"   u = {u:.6f}")
    
    # Étape 2 : Convertir en tirage normal N(0,1)
    Z = np.random.standard_normal()
    print(f"\n2. Convertir en tirage normal Z ~ N(0,1)")
    print(f"   Z = {Z:.6f}")
    
    # Étape 3 : Convertir en tirage brownien W(T) ~ N(0,T)
    W_T = Z * np.sqrt(T)
    print(f"\n3. Convertir en tirage brownien W(T) ~ N(0,T)")
    print(f"   W(T) = Z × √T = {Z:.6f} × {np.sqrt(T):.6f}")
    print(f"   W(T) = {W_T:.6f}")
    
    # Étape 4 : Calculer le prix du sous-jacent à maturité
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * W_T
    S_T = S0 * np.exp(drift + diffusion)
    
    print(f"\n4. Calculer S(T) avec la formule de Black-Scholes")
    print(f"   S(T) = S0 × exp((r - σ²/2)T + σW(T))")
    print(f"   Drift    = (r - σ²/2)T = ({r} - {sigma**2/2:.6f}) × {T} = {drift:.6f}")
    print(f"   Diffusion = σW(T) = {sigma} × {W_T:.6f} = {diffusion:.6f}")
    print(f"   S(T) = {S0} × exp({drift:.6f} + {diffusion:.6f})")
    print(f"   S(T) = {S0} × exp({drift + diffusion:.6f})")
    print(f"   S(T) = {S_T:.6f}")
    
    # Étape 5 : Calculer le payoff de l'option
    payoff_call = max(S_T - K, 0)
    payoff_put = max(K - S_T, 0)
    
    print(f"\n5. Calculer le payoff à maturité")
    print(f"   Call payoff = max(S(T) - K, 0) = max({S_T:.2f} - {K}, 0) = {payoff_call:.6f}")
    print(f"   Put payoff  = max(K - S(T), 0) = max({K} - {S_T:.2f}, 0) = {payoff_put:.6f}")
    
    # Étape 6 : Actualiser à aujourd'hui
    discount_factor = np.exp(-r * T)
    pv_call = payoff_call * discount_factor
    pv_put = payoff_put * discount_factor
    
    print(f"\n6. Actualiser à aujourd'hui")
    print(f"   Facteur d'actualisation = exp(-rT) = exp(-{r} × {T}) = {discount_factor:.6f}")
    print(f"   Valeur actualisée Call = {payoff_call:.6f} × {discount_factor:.6f} = {pv_call:.6f}")
    print(f"   Valeur actualisée Put  = {payoff_put:.6f} × {discount_factor:.6f} = {pv_put:.6f}")
    
    print("\n" + "=" * 70)
    print("Ceci est le résultat d'UNE SEULE simulation.")
    print("Le prix de l'option = moyenne de NOMBREUSES simulations (typiquement 10,000+)")
    print("=" * 70)
    
    return pv_call, pv_put


def compare_mc_convergence():
    """Montre la convergence du prix Monte Carlo avec le nombre de simulations"""
    
    print("\n\n" + "=" * 70)
    print("CONVERGENCE DU PRIX MONTE CARLO")
    print("=" * 70)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    
    market = Market(
        underlying=100.0,
        vol=0.20,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    option = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=100.0
    )
    
    print(f"\nOption: European Call")
    print(f"S0 = {market.underlying}, K = {option.strike}, σ = {market.vol}, r = {market.rate}")
    print("\n" + "-" * 70)
    print(f"{'N Simulations':>15} | {'Prix MC':>12} | {'Erreur Std':>12} | {'IC 95%':>25}")
    print("-" * 70)
    
    simulation_counts = [100, 500, 1000, 5000, 10000, 50000, 100000]
    
    for n_sims in simulation_counts:
        mc_model = MonteCarloModel(
            num_simulations=n_sims,
            market=market,
            option=option,
            pricing_date=pricing_date,
            seed=42
        )
        
        result = mc_model.price_european()
        price = result['price']
        std_err = result['std_error']
        ci_lower = price - 1.96 * std_err
        ci_upper = price + 1.96 * std_err
        
        print(f"{n_sims:>15,} | {price:>12.6f} | {std_err:>12.6f} | [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    print("-" * 70)
    print("\nObservation : L'erreur standard diminue en 1/√N")
    print("Pour réduire l'erreur de moitié, il faut 4× plus de simulations")


def show_price_distribution():
    """Affiche la distribution des prix du sous-jacent à maturité"""
    
    print("\n\n" + "=" * 70)
    print("DISTRIBUTION DES PRIX À MATURITÉ (Monte Carlo)")
    print("=" * 70)
    
    pricing_date = date(2024, 1, 1)
    maturity_date = date(2024, 7, 1)
    
    market = Market(
        underlying=100.0,
        vol=0.20,
        rate=0.05,
        div_a=0.0,
        ex_div_date=None
    )
    
    option = OptionTrade(
        mat=maturity_date,
        call_put='CALL',
        ex='EUROPEAN',
        k=100.0
    )
    
    mc_model = MonteCarloModel(
        num_simulations=10000,
        market=market,
        option=option,
        pricing_date=pricing_date,
        seed=42
    )
    
    # Simuler les trajectoires
    T = (maturity_date - pricing_date).days / 365.0
    S0 = market.underlying
    sigma = market.vol
    r = market.rate
    
    np.random.seed(42)
    Z = np.random.standard_normal(10000)
    S_T = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    
    print(f"\nStatistiques de S(T) sur {len(S_T)} simulations:")
    print(f"  Moyenne:      {np.mean(S_T):.2f}")
    print(f"  Médiane:      {np.median(S_T):.2f}")
    print(f"  Écart-type:   {np.std(S_T):.2f}")
    print(f"  Min:          {np.min(S_T):.2f}")
    print(f"  Max:          {np.max(S_T):.2f}")
    
    # Percentiles
    percentiles = [5, 25, 50, 75, 95]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(S_T, p)
        print(f"  {p}%:  {val:.2f}")
    
    # Prix de l'option
    result = mc_model.price_european()
    print(f"\nPrix de l'option Call (K={option.strike}):")
    print(f"  Prix MC:      {result['price']:.6f}")
    print(f"  Erreur std:   {result['std_error']:.6f}")


if __name__ == "__main__":
    # Démonstration d'une seule trajectoire
    demonstrate_one_path()
    
    # Convergence avec le nombre de simulations
    compare_mc_convergence()
    
    # Distribution des prix
    show_price_distribution()
    
    print("\n" + "=" * 70)
    print("FIN DE LA DÉMONSTRATION")
    print("=" * 70)
