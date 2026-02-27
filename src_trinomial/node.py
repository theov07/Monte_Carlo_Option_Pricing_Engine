from .option_trade import OptionTrade
class Node:
    def __init__(self, underlying_i: float) -> None:
        """A node in the trinomial tree"""
        #node price
        self.underlying_i = underlying_i 
        self.next_mid_node = None
        self.next_up_node = None 
        self.next_down_node = None 

        self.up_node = None
        self.down_node = None 

        self.p_up = None
        self.p_mid = None
        self.p_down = None

        self.option_value = None
        #initialization of the probability to reach this node
        self.p = 0 

    def priceRecursive(self, option: OptionTrade, df: float) -> float:
        """Recursive pricing of the option at this node"""
        #if price already computed, return it
        if self.option_value is not None:
            return self.option_value
        
        # if leaf -> payoff
        if self.next_mid_node is None:
            self.option_value = option.pay_off(self.underlying_i)
            return self.option_value

        # else -> discounted expected value, df for discount factor
        continuation_value = df * sum([
            self.p_up * self.next_up_node.priceRecursive(option, df) if self.next_up_node else 0,
            self.p_mid * self.next_mid_node.priceRecursive(option, df) if self.next_mid_node else 0,
            self.p_down * self.next_down_node.priceRecursive(option, df) if self.next_down_node else 0
        ])

        if option.is_american():
            # compare immediate exercise vs continuation in case of american option
            immediate_value = option.pay_off(self.underlying_i)
            self.option_value = max(immediate_value, continuation_value)
        else:
            self.option_value = continuation_value

        return self.option_value

class TrunkNode(Node):
    def __init__(self, underlying_i: float) -> None:
        """
        A trunk node in the trinomial tree, with a link to the previous trunk node
        Useful for the backward induction pricing method
        """
        super().__init__(underlying_i)
        self.previous_trunk_node = None