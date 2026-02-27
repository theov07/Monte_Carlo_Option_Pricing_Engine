from datetime import date
class OptionTrade:
    CALL_LABEL: str = 'CALL'
    PUT_LABEL: str = 'PUT'
    AMER_LABEL: str = 'AMERICAN'
    EURO_LABEL: str = 'EUROPEAN'
    BINARY_LABEL: str = 'BINARY'
    
    def __init__(self, mat: date, call_put: str, ex: str, k: float) -> None:
        """Option trade parameters"""
        self.mat_date: date = mat
        #upper to avoid case issues
        self.opt_type: str = call_put.upper() 
        self.exercise: str = ex.upper()
        self.strike: float = k

    def is_american(self) -> bool:
        return self.exercise == OptionTrade.AMER_LABEL

    def is_a_call(self) -> bool:
        return self.opt_type == OptionTrade.CALL_LABEL
    
    def is_a_put(self) -> bool:
        return self.opt_type == OptionTrade.PUT_LABEL

    def is_binary(self) -> bool:
        return self.opt_type == OptionTrade.BINARY_LABEL

    def pay_off(self, spot_price: float) -> float:
        if self.is_a_call():
            return max(spot_price - self.strike, 0.0)
        elif self.is_a_put():
            return max(self.strike - spot_price, 0.0)
        #one touch binary call
        elif self.is_binary(): 
            if spot_price >= self.strike:
                return 1
            return 0.0
        else:
            return 0.0