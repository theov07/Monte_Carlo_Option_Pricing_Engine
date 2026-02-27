from .option_trade import OptionTrade
from .market import Market
from .node import Node
from datetime import date
import numpy as np  
import plotly.graph_objects as go #for visualization

class Tree:
    def __init__(self, nb_step: int, market: Market, option: OptionTrade, pricing_date: date, prunning_threshold: float = 1e-8) -> None:
        """Trinomial tree for option pricing"""
        self.nb_step = nb_step #number of time steps, related to the depth of the tree and so the precision of the pricing
        self.delta_t = (option.mat_date - pricing_date).days / 365 / nb_step #time step size in years
        self.market = market
        self.option = option
        self.alpha = np.exp(market.vol * np.sqrt(3*self.delta_t))  # up/down factor bewteen next/mid and mid/down nodes
        self.root = Node(market.underlying) #creation of the root node with the current underlying price
        self.root.p = 1  # initial probability at root
        self.pricing_date = pricing_date
        self.prunning_threshold = prunning_threshold #threshold for pruning nodes with low probability

    def _place_dividende(self) -> int | None:
        """Return the step where the dividend is paid, None if no dividend"""
        if self.market.div_a and self.market.div_a > 0:
            total_days = (self.option.mat_date - self.pricing_date).days
            div_days = (self.market.ex_div_date - self.pricing_date).days
            if 0 <= div_days <= total_days:
                div_step = int(div_days / total_days * self.nb_step)
                return div_step
        return None
    
    def _calculate_proba(self, node: Node, div_t: float) -> None:
            """Calculate transition probabilities for a given node"""
            esp_next_node = node.underlying_i * np.exp(self.market.rate * self.delta_t) - div_t #div_t is the dividend at this step
            var_next_node = node.underlying_i**2 * np.exp(2*self.market.rate*self.delta_t) * (np.exp(self.market.vol**2 * self.delta_t) -1)
            node.p_down = (
                (node.next_mid_node.underlying_i**-2 * (var_next_node + esp_next_node**2) - 1 - (self.alpha + 1) * (node.next_mid_node.underlying_i**-1 * esp_next_node -1))
                / ((1 - self.alpha) * (self.alpha**-2 -1)) 
            )
            node.p_up = ((esp_next_node/node.next_mid_node.underlying_i -1) + (1 - 1/self.alpha) * node.p_down) / (self.alpha - 1) 
            node.p_mid = 1 - node.p_up - node.p_down
            
            #update the probabilities of next nodes
            node.next_up_node.p += node.p * node.p_up
            node.next_mid_node.p += node.p * node.p_mid
            node.next_down_node.p += node.p * node.p_down
            if not (0 <= node.p_up <= 1 and 0 <= node.p_mid <= 1 and 0 <= node.p_down <= 1):
                raise ValueError(f"Probabilités invalides au noeud avec underlying {node.underlying_i:.2f}: p_up={node.p_up}, p_mid={node.p_mid}, p_down={node.p_down}")

    def _build_next_level(self, current_trunk_node: Node, div_t: float) -> None:
            """Build the next level of the tree from the current trunk node i.e up/down nodes"""
            current_node = current_trunk_node
            current_node_up_node_next_mid_candidate = current_node.next_up_node
            while current_node.up_node is not None: #Moove Up, create the upper part of the level
                
                current_node_up_node_next_mid_candidate.up_node = Node(current_node_up_node_next_mid_candidate.underlying_i *self.alpha)
                current_node_up_node_next_mid_candidate.up_node.down_node = current_node_up_node_next_mid_candidate
                
                upper_bound = current_node_up_node_next_mid_candidate.underlying_i * (1 + self.alpha) / 2
                lower_bound = current_node_up_node_next_mid_candidate.underlying_i * (1 + 1 / self.alpha) / 2
                theorical_next_mid_underlying_i = current_node.up_node.underlying_i * np.exp(self.market.rate * self.delta_t) - div_t

                if (theorical_next_mid_underlying_i <= upper_bound and theorical_next_mid_underlying_i >= lower_bound) or theorical_next_mid_underlying_i < 0:
                    current_node.up_node.next_mid_node = current_node_up_node_next_mid_candidate
                    current_node.up_node.next_up_node = current_node_up_node_next_mid_candidate.up_node
                    current_node.up_node.next_down_node = current_node_up_node_next_mid_candidate.down_node

                    if (self.prunning_threshold is not None) and (current_node.up_node.p < self.prunning_threshold and current_node.up_node.up_node is None):
                        #Monomial if the probability is too low and no up child
                        current_node.up_node.p_mid = 1.0
                        current_node.next_mid_node.p += current_node.up_node.p
                        current_node.up_node.next_up_node = None
                        current_node.up_node.next_down_node = None

                        current_node_up_node_next_mid_candidate.up_node = None

                    else:
                        self._calculate_proba(current_node.up_node, div_t)

                    current_node = current_node.up_node
                current_node_up_node_next_mid_candidate = current_node_up_node_next_mid_candidate.up_node

            current_node = current_trunk_node
            current_node_down_node_next_mid_candidate = current_node.next_down_node

            while current_node.down_node is not None: #Moove Down, create the lower part of the level

                current_node_down_node_next_mid_candidate.down_node = Node(current_node_down_node_next_mid_candidate.underlying_i / self.alpha)
                current_node_down_node_next_mid_candidate.down_node.up_node = current_node_down_node_next_mid_candidate

                upper_bound = current_node_down_node_next_mid_candidate.underlying_i * (1 + self.alpha) / 2
                lower_bound = current_node_down_node_next_mid_candidate.underlying_i * (1 + 1 / self.alpha) / 2
                theorical_next_mid_underlying_i = current_node.down_node.underlying_i * np.exp(self.market.rate * self.delta_t) - div_t

                if (theorical_next_mid_underlying_i <= upper_bound and theorical_next_mid_underlying_i >= lower_bound) or theorical_next_mid_underlying_i < 0:
                    current_node.down_node.next_mid_node = current_node_down_node_next_mid_candidate
                    current_node.down_node.next_up_node = current_node_down_node_next_mid_candidate.up_node
                    current_node.down_node.next_down_node = current_node_down_node_next_mid_candidate.down_node
                    
                    if (self.prunning_threshold is not None) and (current_node.down_node.p < self.prunning_threshold and current_node.down_node.down_node is None):
                        #Monomial if the probability is too low and no down child
                        current_node.down_node.p_mid = 1.0
                        current_node.next_mid_node.p += current_node.down_node.p
                        current_node.down_node.next_up_node = None
                        current_node.down_node.next_down_node = None

                        current_node_down_node_next_mid_candidate.down_node = None

                    else:
                        self._calculate_proba(current_node.down_node, div_t)

                    current_node = current_node.down_node
                current_node_down_node_next_mid_candidate = current_node_down_node_next_mid_candidate.down_node

    def build_tree(self) -> None:
        """Build the trinomial tree with the given number of steps"""
        current_trunk_node = self.root
        div_step = self._place_dividende()
        for i in range(self.nb_step):
            div_t = self.market.div_a if div_step is not None and i == div_step else 0
            #instantiate next nodes of the current trunk node
            current_trunk_node.next_mid_node = Node(current_trunk_node.underlying_i * np.exp(self.market.rate * self.delta_t) - div_t)
            current_trunk_node.next_up_node = Node(current_trunk_node.next_mid_node.underlying_i * self.alpha)
            current_trunk_node.next_down_node = Node(current_trunk_node.next_mid_node.underlying_i / self.alpha)
            self._calculate_proba(current_trunk_node, div_t)
            #create the links between up/down nodes
            current_trunk_node.next_mid_node.up_node = current_trunk_node.next_up_node
            current_trunk_node.next_mid_node.up_node.down_node = current_trunk_node.next_mid_node

            current_trunk_node.next_mid_node.down_node = current_trunk_node.next_down_node
            current_trunk_node.next_mid_node.down_node.up_node = current_trunk_node.next_mid_node

            self._build_next_level(current_trunk_node, div_t)
            #advance to the next trunk node
            current_trunk_node = current_trunk_node.next_mid_node

    def price_backward_induction(self) -> float:
        """
        Backward-induction pricing without recursion:
        start from the terminal nodes at maturity and work backward to the root.
        """
        df = np.exp(-self.market.rate * self.delta_t)

        #collect all nodes level by level
        levels = []
        current_level = [self.root]
        for _ in range(self.nb_step + 1):
            levels.append(current_level)
            next_level = []
            for node in current_level:
                for child in [node.next_up_node, node.next_mid_node, node.next_down_node]:
                    if child and child not in next_level:
                        next_level.append(child)
            current_level = next_level

        #final nodes payoffs
        for node in levels[-1]:
            node.option_value = self.option.pay_off(node.underlying_i)

        #backward induction
        for level in reversed(levels[:-1]):
            for node in level:
                continuation = df * (
                    (node.p_up or 0) * (node.next_up_node.option_value if node.next_up_node else 0) +
                    (node.p_mid or 0) * (node.next_mid_node.option_value if node.next_mid_node else 0) +
                    (node.p_down or 0) * (node.next_down_node.option_value if node.next_down_node else 0)
                )
                if self.option.is_american():
                    exercise = self.option.pay_off(node.underlying_i)
                    node.option_value = max(exercise, continuation)
                else:
                    node.option_value = continuation

        return self.root.option_value
    
    def plot_tree(self) -> None:
        """
        Affiche le trinomial tree de manière interactive avec Plotly.
        Chaque nœud montre le prix du sous-jacent et permet d'afficher les
        probabilités de transition et la valeur de l'option au survol.
        """
        def format_if_float(value, fmt=".2f"):
            return f"{value:{fmt}}" if isinstance(value, float) else str(value)

        # --- collecte des noeuds niveau par niveau ---
        levels = []
        current_level = [self.root]

        for _ in range(self.nb_step + 1):
            levels.append(current_level)
            next_level = []
            for node in current_level:
                for child in [node.next_up_node, node.next_mid_node, node.next_down_node]:
                    if child is not None and child not in next_level:
                        next_level.append(child)
            current_level = next_level

        # --- données pour plotly ---
        xs, ys, texts, colors = [], [], [], []
        edges_x, edges_y = [], []

        S0 = self.root.underlying_i
        all_prices = [n.underlying_i for level in levels for n in level]
        min_p, max_p = min(all_prices), max(all_prices)

        # pour chaque niveau
        for i, level_nodes in enumerate(levels):
            for node in sorted(level_nodes, key=lambda n: n.underlying_i, reverse=True):
                x, y = i, node.underlying_i
                xs.append(x)
                ys.append(y)

                # couleur selon prix relatif
                rel = (node.underlying_i - S0) / (max_p - min_p)
                colors.append(rel)

                # info-bulle (tooltip)
                texts.append(
                    f"<b>Step:</b> {i}<br>"
                    f"<b>Underlying:</b> {format_if_float(node.underlying_i, '.2f')}<br>"
                    f"<b>p_up:</b> {format_if_float(node.p_up, '.3f')}<br>"
                    f"<b>p_mid:</b> {format_if_float(node.p_mid, '.3f')}<br>"
                    f"<b>p_down:</b> {format_if_float(node.p_down, '.3f')}<br>"
                    f"<b>Option value:</b> {format_if_float(node.option_value, '.4f')}"
                )


                # arêtes
                for child in [node.next_up_node, node.next_mid_node, node.next_down_node]:
                    if child is not None:
                        edges_x += [x, i + 1, None]
                        edges_y += [y, child.underlying_i, None]

        # --- création du graphe ---
        fig = go.Figure()

        # arêtes (liens)
        fig.add_trace(
            go.Scatter(
                x=edges_x,
                y=edges_y,
                mode="lines",
                line=dict(color="gray", width=0.6),
                hoverinfo="none"
            )
        )

        # nœuds
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=colors,
                    colorscale="RdYlGn",
                    cmin=-0.5,
                    cmax=0.5,
                    showscale=True,
                    colorbar=dict(title="Relative to S₀"),
                    line=dict(color="black", width=0.3)
                ),
                text=[f"{n.underlying_i:.2f}" for level in levels for n in level],
                textposition="top center",
                hovertemplate="%{text}<br>%{customdata}",
                customdata=texts,
                hoverinfo="text"
            )
        )

        fig.update_layout(
            title=f"Trinomial Tree — Interactive Visualization ({self.nb_step} steps)",
            xaxis_title="Time Steps",
            yaxis_title="Underlying Price",
            template="plotly_white",
            showlegend=False,
            height=600
        )

        fig.show()