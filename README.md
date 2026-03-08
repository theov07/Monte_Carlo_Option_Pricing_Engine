# Monte Carlo Option Pricing Engine

Advanced options pricing application using Monte Carlo simulation with the Longstaff-Schwartz algorithm, antithetic variates, and finite-difference Greeks — with real-time interactive visualization.

**Academic project — Paris Dauphine University — Master 2 Financial Engineering — Major in Quantitative Finance**

---

## 🚀 Live Application

### [→ Launch Streamlit App](https://monte-carlo-option-pricing-engine.streamlit.app/)

Interactive web application with real-time pricing, Greeks calculation, GBM path visualization, and convergence analysis.

---

## Features

### Pricing Models

| Model | Scope | Notes |
|---|---|---|
| **Black-Scholes** | European Call & Put | Closed-form with discrete dividend (escrow approximation) |
| **Monte Carlo — Vectorized** | European Call, Put, Binary | Antithetic variates, fully vectorized with NumPy |
| **Longstaff-Schwartz** | American Call & Put | Least-squares MC with polynomial basis regression |
| **Trinomial Tree** *(benchmark)* | European & American | CRR-extended recombining tree for validation |

**Dividend handling:** discrete cash dividend with ex-dividend date support — spot adjusted by the present value of the dividend: $S_\text{eff} = S_0 - D \cdot e^{-r t_\text{div}}$ (escrow approximation).

### Greeks & Sensitivity Analysis

- **Complete Greeks suite:** Delta (Δ), Gamma (Γ), Vega (ν), Theta (Θ), Rho (ρ)
- **Method:** central finite differences with **Common Random Numbers (CRN)** — same seed for base and bumped pricings, dramatically reducing estimator variance
- **Standard errors:** propagated analytically from elementary pricers
- **95% confidence intervals** on each Greek

### Visualizations

- **GBM paths:** interactive display of simulated trajectories with terminal spot distribution
- **Payoff diagram:** option payoff at maturity across spot range
- **Convergence study:** MC price and standard error vs number of paths $N$ (log/log scale confirming the $O(1/\sqrt{N})$ rate)
- **Strike profile:** price vs strike across the moneyness range
- **MC vs. Black-Scholes comparison:** real-time benchmarking with relative error

---

## Mathematical Background

### Geometric Brownian Motion

The underlying follows:

$$dS_t = r S_t \, dt + \sigma S_t \, dW_t$$

Simulated via exact discretization:

$$S_{t+\Delta t} = S_t \exp\!\left(\left(r - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t}\, Z\right), \quad Z \sim \mathcal{N}(0,1)$$

### Antithetic Variates

For each standard normal draw $Z$, a paired draw $-Z$ is used. The estimator:

$$\hat{P} = \frac{1}{2N}\sum_{i=1}^{N}\left[f(S^{(i)}) + f(S^{(-i)})\right]$$

reduces variance when the payoff is monotone in $Z$, which holds for vanilla options.

### Longstaff-Schwartz (American Options)

Backward induction with least-squares regression of the **continuation value** on a polynomial basis of the current spot:

$$\hat{C}(S_t) = \sum_{k=0}^{d} \beta_k \psi_k(S_t)$$

Two basis families are supported:
- **Power:** $\psi_k(x) = x^k$ (standard monomials)
- **Laguerre:** $\psi_k(x) = e^{-x/2} L_k(x)$ — basis used in the original Longstaff-Schwartz (2001) paper

Early exercise is applied at each step:

$$V(S, t) = \max\!\left(\text{Intrinsic}(S),\; \hat{C}(S_t)\right)$$

### Monte Carlo Greeks (Finite Differences + CRN)

Central difference schemes with the **same seed** for base and bumped pricings:

| Greek | Formula |
|---|---|
| Delta | $\Delta = \dfrac{V(S_0+h) - V(S_0-h)}{2h}$ |
| Gamma | $\Gamma = \dfrac{V(S_0+h) - 2V(S_0) + V(S_0-h)}{h^2}$ |
| Vega | $\nu = \dfrac{V(\sigma+h) - V(\sigma-h)}{2h}$ |
| Theta | $\Theta = -\dfrac{V(t+1\text{d}) - V(t)}{1/365}$ |
| Rho | $\rho = \dfrac{V(r+h) - V(r-h)}{2h}$ |

Standard errors are propagated from elementary pricer SEs: e.g. $\text{SE}(\Delta) = \sqrt{\text{SE}_+^2 + \text{SE}_-^2} / (2h)$.

---

## Code Architecture

```
src/
├── instruments/
│   ├── market.py           # Market parameters (spot, vol, rate, dividend)
│   └── option_trade.py     # Option contract (strike, maturity, call/put, exercise)
├── models/
│   └── brownian_motion.py  # GBM path simulation (scalar and vectorized)
├── pricing/
│   ├── black_scholes.py    # Analytical BS pricer (European + discrete dividend)
│   ├── monte_carlo_model.py# MC engine: European vectorized, LS American
│   ├── greeks.py           # MCGreeks + GreeksConfig + AllGreeks
│   ├── regression.py       # Least-squares regression, BasisType (POWER / LAGUERRE)
│   └── pricing_result.py   # PricingResult dataclass (price, SE, CI, elapsed time)
├── studies/
│   └── convergence.py      # ConvergenceStudy: MC SE vs N analysis
└── benchmarks/
    └── trinomial_tree/     # CRR trinomial tree (reference benchmark)
        ├── market.py
        ├── option_trade.py
        ├── node.py
        ├── tree.py
        ├── trinomial_model.py
        └── tree_pricing.py

app.py                      # Streamlit interactive application
main.py                     # Quick CLI pricing script
scripts/
├── analysis/               # Comparison and validation scripts
└── plots/                  # Figure generation scripts
figures/                    # Output PNGs
docs/
├── report_short.pdf        # Short project report
└── report_full.pdf         # Full project report
requirements.txt
```

---

## Installation & Usage

### Prerequisites

- Python 3.11+
- Git

### Local Installation

```bash
# Clone the repository
git clone https://github.com/theov07/Monte_Carlo_Option_Pricing_Engine.git
cd Monte-Carlo-Option-Pricing

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Launch the Streamlit App

```bash
streamlit run app.py
```

The application opens automatically at [http://localhost:8501](http://localhost:8501).

### Quick CLI Pricing

```bash
python main.py
```

---

## Application Guide

### 1. Configure Parameters (Left Sidebar)

| Section | Parameters |
|---|---|
| **Market** | Spot price $S_0$, Volatility $\sigma$, Risk-free rate $r$, Dividend $D$ |
| **Option** | Strike $K$, Maturity (months), Call/Put, European/American |
| **Monte Carlo** | Number of paths $N$, Antithetic variates toggle, Seed, Steps (LS) |

### 2. Tabs Overview

| Tab | Content |
|---|---|
| 💰 **Pricing** | MC + BS prices, metric cards, results table, payoff diagram |
| 📉 **Paths** | GBM path simulation with terminal spot distribution |
| 🔢 **Greeks** | Δ Γ ν Θ ρ with SE, CI, bar chart, MC vs BS comparison |
| 📊 **Convergence** | Price and SE vs $N$ (8 levels), $O(1/\sqrt{N})$ validation |
| 🌊 **Strike Profile** | Price as a function of strike across moneyness range |
| ℹ️ **About** | Architecture, references |

---

## Example Configurations

### European Call — ATM

| Parameter | Value |
|---|---|
| Spot $S_0$ | 100 |
| Strike $K$ | 100 |
| Volatility $\sigma$ | 20% |
| Risk-free rate $r$ | 5% |
| Maturity $T$ | 1 year |
| Paths $N$ | 100 000 |

Expected MC price: ~10.45 (matches Black-Scholes)

### American Put — ITM with Dividend

| Parameter | Value |
|---|---|
| Spot $S_0$ | 100 |
| Strike $K$ | 110 |
| Volatility $\sigma$ | 25% |
| Dividend $D$ | 3.0 at 6 months |
| Maturity $T$ | 1 year |
| Method | Longstaff-Schwartz |

Early exercise premium is visible vs European BS price.

---

## Performance

| Paths $N$ | European MC | LS American |
|---|---|---|
| 10 000 | ~20 ms | ~100 ms |
| 50 000 | ~80 ms | ~500 ms |
| 100 000 | ~150 ms | ~1 s |

*(Measured on Apple M-series — results vary by hardware)*

---

## Dependencies

```
numpy>=1.26,<2.0
scipy>=1.11
matplotlib>=3.8
pandas>=2.1
streamlit>=1.35
```

Install:

```bash
pip install -r requirements.txt
```

---

## References

- Longstaff, F. A. & Schwartz, E. S. (2001). *Valuing American Options by Simulation: A Simple Least-Squares Approach.* The Review of Financial Studies, 14(1), 113–147.
- Glasserman, P. (2004). *Monte Carlo Methods in Financial Engineering.* Springer.
- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
- Cox, J., Ross, S. & Rubinstein, M. (1979). *Option Pricing: A Simplified Approach.* Journal of Financial Economics, 7(3), 229–263.

---

**Author:** VERDELHAN Théo & RENAULT Léo

**Institution:** Paris Dauphine University — Master 2 Financial Engineering — Major in Quantitative Finance
