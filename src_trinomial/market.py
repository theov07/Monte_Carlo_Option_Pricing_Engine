from datetime import date
class Market:
    def __init__(self, underlying: float, vol: float, rate: float, div_a: float, ex_div_date: date) -> None:
        """Market parameters for option pricing"""
        self.underlying = underlying
        self.vol = vol
        self.rate = rate
        self.div_a = div_a #dividend amount 
        self.ex_div_date = ex_div_date