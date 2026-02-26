import time 
from datetime import date

import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import numpy as np

from src.tree import Tree
from src.option_trade import OptionTrade
from src.market import Market
from src.trinomial_model import TrinomialModel
from PriceurBS import Call, Put

def main():
    # Paramètres du marché
    underlying = 102.45
    vol = 0.28
    rate = 0.04
    div = 3
    market = Market(underlying, vol, rate, div, date(2026, 6, 9))

    # Paramètres de l'option
    strike = 91
    maturity = date(2026, 8, 27)
    type_option = 'american'
    direction = 'put'
    option = OptionTrade(k = strike, mat = maturity, ex = type_option, call_put = direction)
    # Paramètres de l'arbre
    nb_step = 15 #impossible d'aller au delà de 994 sinon erreur récursion 
    date_valuation = date(2025, 10, 29)
    prunning_threshold = 1e-8 #1e-15 par exemple 
    tree = Tree(nb_step, market, option, date_valuation, prunning_threshold)
    mod = TrinomialModel(date_valuation, tree)
    print("Tree created with root underlying:", tree.root.underlying_i)


    print("Différence date", maturity - date_valuation)

    
    
    start = time.time()
    tree.build_tree()
    end = time.time()

    print("temps d'exécution création arbre:", end - start, "secondes")
    
    start = time.time()
    #print("L'option via pricing recursif vaut = ", mod.price(option, "recursive"))
    print("L'option via backward induction vaut = ", mod.price(option, "backward"))
    end = time.time()
    print("temps d'exécution priceur récursif:", end - start, "secondes")

    tree.plot_tree()
    # Greeks

    print("Delta =", mod.delta(option))
    print("Gamma =", mod.gamma(option))
    print("Delta hedge", 1000*mod.delta(option))
    if nb_step <= 950: #car utilise price recursive qui est limité à 950 
        
        print("Vega  =", mod.vega(option))
        print("Vomma =", mod.vomma(option))
        print("Vanna =", mod.vanna(option))
    

if __name__ == "__main__":
    main()