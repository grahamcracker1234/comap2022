import pandas as pd
from dataclasses import dataclass

@dataclass
class Currency:
    usd: float = 0
    btc: float = 0
    gold: float = 0
    
@dataclass
class Action(Currency):pass

@dataclass
class Hold(Action):pass

@dataclass
class Buy(Action):
    def __init__(self, btc_value, usd) -> None:
        self.btc = usd * (1 - commissionRate.btc) / btc_value
        self.usd = usd
        
@dataclass
class Sell(Action):
    def __init__(self, btc_value, btc) -> None:
        self.btc = btc
        self.usd = (btc_value * btc) * (1 - commissionRate.btc)
        
@dataclass
class Hold(Action): pass

def nullAction(day, streamHistory, wallet) -> Action:
    if day == 0: return Buy(streamHistory.BTC.values[-1], wallet.usd)
    if day == 1825: return Sell(streamHistory.BTC.values[-1], wallet.btc)
    
def nullActionGold(day, streamHistory, wallet) -> Action:
    if day == 0: return Buy(streamHistory.BTC.values[-1], wallet.usd)
    if day == 1825: return Sell(streamHistory.BTC.values[-1], wallet.btc)
    
    
data = pd.read_csv("data/full.csv")
commissionRate = Currency(1, 0.02, 0.01)
wallet = Currency(1000, 0, 0)

for day in data.index:
    streamHistory = data[:day+1]
    print("-" * 50)
    print(f"Day {day}")
    print(wallet)
    # print(streamHistory.tail(), end="\n\n")
    action = nullAction(day, streamHistory, wallet)
    if isinstance(action, Buy):
        wallet.usd -= action.usd
        wallet.btc += action.btc
    elif isinstance(action, Sell):
        wallet.usd += action.usd
        wallet.btc -= action.btc
        
print(f"Total USD: ${round(wallet.usd, 2)}")