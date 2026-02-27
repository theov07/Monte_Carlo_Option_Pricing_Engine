"""
ConvergenceStudy — Étude de la convergence Monte-Carlo.

Reproduit l'analyse "Step 4" du cours :
    std_error(MC) ≈ σ_payoff / √N  →  droite log/log de pente -½
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable

import numpy as np


@dataclass
class ConvergencePoint:
    """Un point de la courbe de convergence."""
    n_paths: int
    price_mean: float       # moyenne sur n_repeat tirages
    price_std: float        # écart-type entre les n_repeat tirages
    se_mean: float          # moyenne des std_error MC individuelles
    elapsed_s: float        # temps total pour ce N


class ConvergenceStudy:
    """
    Étudie la convergence en N (nombre de trajectoires) d'une méthode MC.

    Parameters
    ----------
    mc_model : MonteCarloModel
        Modèle déjà configuré (market, option, pricing_date, seed).
    method : str
        Méthode à tester parmi :
        - 'european'        → price_european_vectorized
        - 'european_scalar' → price_european (version scalaire)
        - 'american_ls'     → price_american_longstaff_schwartz_vectorized
    ls_kwargs : dict
        Kwargs supplémentaires passés à la méthode LS (num_steps, poly_basis, …)
    antithetic : bool
        Utiliser les variables antithétiques.
    """

    def __init__(self, mc_model,
                 method: str = 'european',
                 ls_kwargs: Optional[dict] = None,
                 antithetic: bool = True):
        self.mc_model  = mc_model
        self.method    = method
        self.ls_kwargs = ls_kwargs or {}
        self.antithetic = antithetic
        self.results: List[ConvergencePoint] = []

    # ------------------------------------------------------------------
    # Construction de l'appelant MC selon la méthode
    # ------------------------------------------------------------------

    def _make_pricer(self, n: int) -> Callable[[], dict]:
        """Retourne un callable sans argument qui prixe avec n trajectoires."""
        mc = self.mc_model
        # Changer temporairement le nombre de simulations
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
            raise ValueError(f"Méthode inconnue : '{self.method}'. "
                             "Choisir parmi: european, european_scalar, american_ls")

        # Wrapper qui restaure num_simulations après chaque appel
        def wrapped():
            mc.num_simulations = n
            result = pricer()
            mc.num_simulations = old_n
            return result

        return wrapped

    # ------------------------------------------------------------------
    # Exécution de l'étude
    # ------------------------------------------------------------------

    def run(self, n_list: List[int], n_repeat: int = 10,
            seed_start: int = 0) -> "ConvergenceStudy":
        """
        Lance n_repeat simulations pour chaque N dans n_list.

        Parameters
        ----------
        n_list     : liste de nombres de trajectoires à tester
        n_repeat   : nombre de répétitions pour estimer la variance des estimateurs
        seed_start : premier seed (utilise seed_start, seed_start+1, …)
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

        self.mc_model.seed = original_seed  # restaure le seed initial
        return self

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def print_table(self, reference: Optional[float] = None) -> None:
        """
        Affiche un tableau de convergence.

        Colonnes : N | mean_price | std_price | se_mean | bias | t(s)
        """
        if not self.results:
            print("Aucun résultat — appeler run() d'abord.")
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
    # Tracé
    # ------------------------------------------------------------------

    def plot(self, reference: Optional[float] = None,
             title: Optional[str] = None) -> None:
        """
        Trace deux sous-graphes :
        1. Prix moyen ± 1σ en fonction de N (avec la référence en pointillé)
        2. std_price et se_mean vs N en échelle log/log  →  pente théorique -½
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib non installé. Installer avec: pip install matplotlib")
            return

        if not self.results:
            print("Aucun résultat — appeler run() d'abord.")
            return

        ns      = np.array([pt.n_paths    for pt in self.results])
        means   = np.array([pt.price_mean for pt in self.results])
        stds    = np.array([pt.price_std  for pt in self.results])
        se_mean = np.array([pt.se_mean    for pt in self.results])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # --- Subplot 1 : convergence du prix ---
        ax = axes[0]
        ax.fill_between(ns, means - stds, means + stds, alpha=0.25, label='±1σ (entre répétitions)')
        ax.plot(ns, means, 'o-', label='Prix moyen MC')
        if reference is not None:
            ax.axhline(reference, color='red', linestyle='--', label=f'Référence = {reference:.4f}')
        ax.set_xscale('log')
        ax.set_xlabel('N (nombre de trajectoires)')
        ax.set_ylabel('Prix')
        ax.set_title('Convergence du prix')
        ax.legend()
        ax.grid(True, which='both', alpha=0.3)

        # --- Subplot 2 : std vs N (log/log) ---
        ax = axes[1]
        ax.loglog(ns, stds,    'o-', label='std (entre répétitions)')
        ax.loglog(ns, se_mean, 's--', label='se moyen (dans chaque run)')

        # Pente théorique -½
        c = stds[0] * ns[0] ** 0.5
        ax.loglog(ns, c / np.sqrt(ns), 'k:', label='pente -½ (théorie)')

        ax.set_xlabel('N (nombre de trajectoires)')
        ax.set_ylabel('Erreur standard')
        ax.set_title('Décroissance de l\'erreur standard')
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
        """Convertit les résultats en pandas DataFrame (si pandas disponible)."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas non installé. Installer avec: pip install pandas")

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
