from .tree import Tree
from .option_trade import OptionTrade
from .market import Market
from .node import Node
from datetime import date
import numpy as np  


class TrinomialModel:
    def __init__(self, pricing_date: date, tree: Tree) -> None:
        """Trinomial Model to launch pricing and greeks calculations"""
        self.tree = tree
        self.pricing_date = pricing_date

    def price(self, option: OptionTrade, method="backward") -> float:
        """User can decide wether to use recursive pricing or backward induction"""
        if method == "recursive":
            df = np.exp(-self.tree.market.rate * self.tree.delta_t)
            return self.tree.root.priceRecursive(option, df)
        elif method == "backward":
            return self.tree.price_backward_induction()
        else:
            raise ValueError("method must be 'recursive' or 'backward'")

    # Greeks calculations

    def _ensure_first_step_priced(self, option: OptionTrade) -> Node:
        """Make sure the three children of the root are priced"""
        root = self.tree.root
        df = np.exp(-self.tree.market.rate * self.tree.delta_t)     # df : discount factor 
        for child in (root.next_up_node, root.next_mid_node, root.next_down_node):
            if child is not None and child.option_value is None:
                child.priceRecursive(option, df)
        return root

    def delta(self, option: OptionTrade) -> float:
        """∂V/∂S Local delta from the first step (no reprice)."""
        root = self._ensure_first_step_priced(option)
        Su, Sm, Sd = (root.next_up_node.underlying_i,
                      root.next_mid_node.underlying_i,
                      root.next_down_node.underlying_i)
        Vu, Vm, Vd = (root.next_up_node.option_value,
                      root.next_mid_node.option_value,
                      root.next_down_node.option_value)
        return (Vu - Vd) / (Su - Sd)

    def gamma(self, option: OptionTrade) -> float:
        """∂²V/∂S² Local gamma from the first step on a (possibly) non-uniform grid (no reprice)."""
        root = self._ensure_first_step_priced(option)
        Su, Sm, Sd = (root.next_up_node.underlying_i,
                      root.next_mid_node.underlying_i,
                      root.next_down_node.underlying_i)
        Vu, Vm, Vd = (root.next_up_node.option_value,
                      root.next_mid_node.option_value,
                      root.next_down_node.option_value)
        h1, h2 = (Su - Sm), (Sm - Sd)
        denom = Su - Sd
        return 2.0 * ((Vu - Vm) / h1 - (Vm - Vd) / h2) / denom

    # used only for vol-based Greeks, it requires repricing bc changing vol changes the tree
    def _reprice(self, option: OptionTrade, S=None, sigma=None) -> float:
        """Reprice the option with a bumped market parameter (underlying or vol)"""
        mkt = self.tree.market
        bumped = Market(
            underlying = mkt.underlying if S is None else S,
            vol        = max(1e-8, mkt.vol if sigma is None else sigma),
            rate       = mkt.rate,
            div_a      = mkt.div_a,
            ex_div_date= mkt.ex_div_date,
        )
        T = Tree(self.tree.nb_step, bumped, self.tree.option, self.pricing_date)
        T.build_tree()
        return TrinomialModel(self.pricing_date, T).price(option)

    def vega(self, option: OptionTrade, h_abs: float = 1e-4) -> float:
        """∂V/∂σ using a short central diff (2 reprices total)."""
        s0 = self.tree.market.vol; h = max(1e-8, h_abs)
        up = self._reprice(option, sigma=s0 + h)
        dn = self._reprice(option, sigma=s0 - h)
        return (up - dn) / (2.0 * h) / 100      # divided by 100 to get per 1% vol

    def vomma(self, option: OptionTrade, h_abs: float = 1e-4) -> float:
        """∂²V/∂σ² using the same evaluations (price0 is already known)."""
        s0 = self.tree.market.vol; h = max(1e-8, h_abs)
        up = self._reprice(option, sigma=s0 + h)
        mi = self._reprice(option, sigma=s0)
        dn = self._reprice(option, sigma=s0 - h)
        return (up - 2.0 * mi + dn) / (h * h) / 10000   # per 1% vol squared

    def vanna(self, option: OptionTrade, hS_rel: float = 1e-4, hV_abs: float = 1e-4) -> float:
        """∂²V/(∂S ∂σ); only when you need it."""
        S0 = self.tree.market.underlying
        hS = max(1e-8, hS_rel * max(1.0, S0))
        s0 = self.tree.market.vol
        hV = max(1e-8, hV_abs)
        f_pp = self._reprice(option, S=S0 + hS, sigma=s0 + hV)
        f_pm = self._reprice(option, S=S0 + hS, sigma=s0 - hV)
        f_mp = self._reprice(option, S=S0 - hS, sigma=s0 + hV)
        f_mm = self._reprice(option, S=S0 - hS, sigma=s0 - hV)
        return ((f_pp - f_pm - f_mp + f_mm) / (4.0 * hS * hV) ) / 100 # per 1% vol