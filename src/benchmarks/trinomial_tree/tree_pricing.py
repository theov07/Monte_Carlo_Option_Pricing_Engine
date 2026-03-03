import time
from datetime import date

from .market import Market
from .option_trade import OptionTrade
from .tree import Tree
from .trinomial_model import TrinomialModel


def tree_pricing(market: Market, option: OptionTrade, pricing_date: date, n_steps: int = 500):
    pruning_threshold = 1e-8
    tree = Tree(n_steps, market, option, pricing_date, pruning_threshold)
    mod = TrinomialModel(pricing_date, tree)
    print("Tree created with root underlying:", tree.root.underlying_i)
    
    start = time.time()
    tree.build_tree()
    end = time.time()

    print("Tree build time:", end - start, "seconds")
    
    start = time.time()
    print("Option price via backward induction = ", mod.price(option, "backward"))
    end = time.time()
    print("Backward induction time:", end - start, "seconds")

    # tree.plot_tree()
    # Greeks

    print("Delta =", mod.delta(option))
    print("Gamma =", mod.gamma(option))
    print("Delta hedge", 1000*mod.delta(option))
    if n_steps <= 950:
        
        print("Vega  =", mod.vega(option))
        print("Vomma =", mod.vomma(option))
        print("Vanna =", mod.vanna(option))