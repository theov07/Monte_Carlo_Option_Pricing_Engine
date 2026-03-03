"""
ConvergenceStudy -- Monte Carlo convergence analysis.

Reproduces the "Step 4" analysis:
    std_error(MC) ~ sigma_payoff / sqrt(N)  ->  log/log slope of -1/2
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergencePoint:
    """A single point on the convergence curve."""
    n_paths: int
    price_mean: float       # mean over n_repeat independent runs
    price_std: float        # standard deviation across n_repeat runs
    se_mean: float          # mean of individual MC std_error estimates
    elapsed_s: float        # total wall-clock time for this N


class ConvergenceStudy:
    """
    Studies Monte Carlo convergence as a function of N (number of paths).

    Parameters
    ----------
    mc_model : MonteCarloModel
        Pre-configured model (market, option, pricing_date, seed).
    method : str
        Pricing method to test:
        - 'european'        -> price_european_vectorized
        - 'european_scalar' -> price_european (scalar version)
        - 'american_ls'     -> price_american_longstaff_schwartz_vectorized
    ls_kwargs : dict
        Additional kwargs passed to the LS method (num_steps, poly_basis, ...)
    antithetic : bool
        Use antithetic variates for variance reduction.
    """

    def __init__(self, mc_model,
                 method: str = 'european',
                 ls_kwargs: dict | None = None,
                 antithetic: bool = True):
        self.mc_model  = mc_model
        self.method    = method
        self.ls_kwargs = ls_kwargs or {}
        self.antithetic = antithetic
        self.results: list[ConvergencePoint] = []

    # ------------------------------------------------------------------
    # MC callable factory
    # ------------------------------------------------------------------

    def _make_pricer(self, n: int) -> Callable[[], dict]:
        """Returns a zero-argument callable that prices with n paths."""
        mc = self.mc_model
        # Temporarily override the number of simulations
        old_n = mc.num_simulations
        mc.num_simulations = n

        if self.method == 'european':
            def pricer():
                return mc.price_european_vectorized(antithetic=self.antithetic)
        elif self.method == 'european_scalar':
            def pricer():
                return mc.price_european(antithetic=self.antithetic)
        elif self.method == 'american_ls':
            kwargs = {**self.ls_kwargs, 'antithetic': self.antithetic}
            def pricer():
                return mc.price_american_longstaff_schwartz_vectorized(**kwargs)
        else:
            raise ValueError(f"Unknown method: '{self.method}'. "
                             "Choose from: european, european_scalar, american_ls")

        # Wrapper that restores num_simulations after each call
        def wrapped():
            mc.num_simulations = n
            result = pricer()
            mc.num_simulations = old_n
            return result

        return wrapped

    # ------------------------------------------------------------------
    # Run the study
    # ------------------------------------------------------------------

    def run(self, n_list: list[int], n_repeat: int = 10,
            seed_start: int = 0) -> ConvergenceStudy:
        """
        Runs n_repeat simulations for each N in n_list.

        Parameters
        ----------
        n_list     : list of path counts to test
        n_repeat   : number of independent repetitions to estimate estimator variance
        seed_start : first seed (uses seed_start, seed_start+1, ...)
        """
        self.results = []
        original_seed = self.mc_model.seed

        for n in n_list:
            prices    = []
            se_list   = []
            t0 = time.perf_counter()

            for rep in range(n_repeat):
                self.mc_model.seed = seed_start + rep
                pricer = self._make_pricer(n)
                res    = pricer()
                prices.append(res['price'])
                se_list.append(res.get('std_error', 0.0))

            elapsed = time.perf_counter() - t0
            self.results.append(ConvergencePoint(
                n_paths    = n,
                price_mean = float(np.mean(prices)),
                price_std  = float(np.std(prices, ddof=1)),
                se_mean    = float(np.mean(se_list)),
                elapsed_s  = elapsed,
            ))

        self.mc_model.seed = original_seed  # restore the original seed
        return self

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_table(self, reference: float | None = None) -> None:
        """
        Prints a convergence table.

        Columns: N | mean_price | std_price | se_mean | bias | t(s)
        """
        if not self.results:
            print("No results -- call run() first.")
            return

        header = f"{'N':>10}  {'mean_price':>10}  {'std_price':>10}  {'se_mean':>10}"
        if reference is not None:
            header += f"  {'bias%':>8}"
        header += f"  {'t(s)':>6}"
        print(header)
        print("-" * len(header))

        for pt in self.results:
            line = (f"{pt.n_paths:>10,}  {pt.price_mean:>10.4f}  "
                    f"{pt.price_std:>10.5f}  {pt.se_mean:>10.5f}")
            if reference is not None:
                bias = (pt.price_mean - reference) / reference * 100
                line += f"  {bias:>+7.3f}%"
            line += f"  {pt.elapsed_s:>6.2f}"
            print(line)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    def plot(self, reference: float | None = None,
             title: str | None = None) -> None:
        """
        Plots two sub-panels:
        1. Mean price +/- 1 sigma vs N (with optional reference dashed line)
        2. std_price and se_mean vs N on log/log scale -- theoretical slope -1/2
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Install with: pip install matplotlib")
            return

        if not self.results:
            print("No results -- call run() first.")
            return

        ns      = np.array([pt.n_paths    for pt in self.results])
        means   = np.array([pt.price_mean for pt in self.results])
        stds    = np.array([pt.price_std  for pt in self.results])
        se_mean = np.array([pt.se_mean    for pt in self.results])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Subplot 1 : price convergence ---
        ax = axes[0]
        ax.fill_between(ns, means - stds, means + stds, alpha=0.25, label='+/-1 sigma (across runs)')
        ax.plot(ns, means, 'o-', label='MC mean price')
        if reference is not None:
            ax.axhline(reference, color='red', linestyle='--', label=f'Reference = {reference:.4f}')
        ax.set_xscale('log')
        ax.set_xlabel('N (number of paths)')
        ax.set_ylabel('Price')
        ax.set_title('Price Convergence')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        # --- Subplot 2 : std vs N (log/log) ---
        ax = axes[1]
        ax.loglog(ns, stds,    'o-', label='std (across runs)')
        ax.loglog(ns, se_mean, 's--', label='mean se (within each run)')

        # Theoretical slope -1/2
        c = stds[0] * ns[0] ** 0.5
        ax.loglog(ns, c / np.sqrt(ns), 'k:', label='slope -1/2 (theory)')

        ax.set_xlabel('N (number of paths)')
        ax.set_ylabel('Standard error')
        ax.set_title('Standard Error Decay')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        if title:
            fig.suptitle(title, fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Converts results to a pandas DataFrame (requires pandas)."""
        try:
            import pandas as pd
        except ImportError as err:
            raise ImportError("pandas not installed. Install with: pip install pandas") from err

        rows = []
        for pt in self.results:
            rows.append({
                'n_paths':    pt.n_paths,
                'price_mean': pt.price_mean,
                'price_std':  pt.price_std,
                'se_mean':    pt.se_mean,
                'elapsed_s':  pt.elapsed_s,
            })
        return pd.DataFrame(rows)
