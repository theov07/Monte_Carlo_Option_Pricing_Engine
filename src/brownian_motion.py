import numpy as np


class BrownianMotion:
    """
    Génère des mouvements Browniens pour la simulation Monte Carlo.

    Utilise np.random.default_rng (objet Generator) pour la reproductibilité
    sans effet de bord sur l'état global du générateur NumPy.

    Supporte les dividendes discrets via les paramètres div/jdiv de generate_paths
    (formule cours 5-Feb) :
        j < jdiv  : GBM standard
        j = jdiv  : GBM + soustraction du dividende (saut ex-div)
        j > jdiv  : GBM depuis S(jdiv) après la baisse

    Formule Black-Scholes (GBM) :
        S(t+dt) = S(t) * exp((r - q - 0.5*σ²)*dt + σ*dW),  dW ~ N(0, dt)
    """

    def __init__(self, num_paths: int, num_steps: int, T: float,
                 antithetic: bool = True, seed=None):
        """
        Parameters
        ----------
        num_paths  : nombre de paths Monte Carlo
        num_steps  : nombre de pas de temps entre 0 et T
        T          : maturité en années
        antithetic : si True, génère les paths antithétiques (-dW)
        seed       : graine pour np.random.default_rng (int, SeedSequence, …)
        """
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.T = T
        self.antithetic = antithetic
        self.dt = T / num_steps if num_steps > 0 else T
        # Objet Generator : état local, pas d'effet de bord sur np.random global
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Génération des incréments Browniens
    # ------------------------------------------------------------------

    def generate_increments_scalar(self) -> tuple:
        """
        Génère UN incrément dW ~ N(0, dt) via le Generator.

        Returns
        -------
        dW      : float, incrément principal
        dW_anti : float (-dW) si antithetic=True, sinon None
        """
        Z = self._rng.standard_normal()
        dW = Z * np.sqrt(self.dt)
        return dW, -dW if self.antithetic else None

    def generate_increments_vectorized(self) -> np.ndarray:
        """
        Génère la matrice complète des incréments : shape (num_paths, num_steps).
        """
        return self._rng.standard_normal((self.num_paths, self.num_steps)) * np.sqrt(self.dt)

    # ------------------------------------------------------------------
    # Génération des paths de prix (GBM + dividende discret optionnel)
    # ------------------------------------------------------------------

    def generate_paths(self, S0: float, r: float, sigma: float,
                       q: float = 0.0,
                       div: float = 0.0, jdiv: int = None) -> tuple:
        """
        Paths complets S(t) pour tous les pas de temps.

        Dividende discret (formule cours 5-Feb) :
          j < jdiv  : S(j*dt) = S((j-1)*dt) * exp(drift*dt + σ*dW)
          j = jdiv  : idem, PUIS S(jdiv*dt) -= div  (saut ex-dividende)
          j > jdiv  : GBM depuis S(jdiv) après la baisse

        Parameters
        ----------
        S0, r, sigma, q : paramètres Black-Scholes habituels
        div  : montant absolu du dividende discret (0 = pas de dividende)
        jdiv : index du step ex-div (1-basé, 1 ≤ jdiv ≤ num_steps, None = absent)

        Returns
        -------
        S_paths      : ndarray (num_paths, num_steps+1)
        S_paths_anti : ndarray (num_paths, num_steps+1) ou None
        """
        dW = self.generate_increments_vectorized()
        drift = (r - q - 0.5 * sigma ** 2) * self.dt

        S = np.zeros((self.num_paths, self.num_steps + 1))
        S[:, 0] = S0
        for step in range(self.num_steps):
            S[:, step + 1] = S[:, step] * np.exp(drift + sigma * dW[:, step])
            if jdiv is not None and step + 1 == jdiv and div > 0:
                S[:, step + 1] = np.maximum(S[:, step + 1] - div, 0.0)

        if not self.antithetic:
            return S, None

        S_anti = np.zeros((self.num_paths, self.num_steps + 1))
        S_anti[:, 0] = S0
        for step in range(self.num_steps):
            S_anti[:, step + 1] = S_anti[:, step] * np.exp(drift + sigma * (-dW[:, step]))
            if jdiv is not None and step + 1 == jdiv and div > 0:
                S_anti[:, step + 1] = np.maximum(S_anti[:, step + 1] - div, 0.0)

        return S, S_anti

    def generate_terminal_prices(self, S0: float, r: float, sigma: float,
                                 q: float = 0.0) -> tuple:
        """
        Génère uniquement S(T) en un seul saut (pricing européen sans dividende discret).

        Note : pour les options avec dividende discret, utiliser generate_paths
               avec suffisamment de steps pour capturer le saut ex-div.

        Returns
        -------
        S_T      : ndarray (num_paths,)
        S_T_anti : ndarray (num_paths,) ou None
        """
        Z = self._rng.standard_normal(self.num_paths)
        W_T = Z * np.sqrt(self.T)
        drift = (r - q - 0.5 * sigma ** 2) * self.T
        S_T = S0 * np.exp(drift + sigma * W_T)
        if not self.antithetic:
            return S_T, None
        return S_T, S0 * np.exp(drift - sigma * W_T)
