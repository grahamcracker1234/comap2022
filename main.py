import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class Currency:
    usd: float = 0
    btc: float = 0
    gold: float = 0
    
@dataclass
class Action(Currency): pass

@dataclass
class Buy(Action):
    def __init__(self, currency_rate: Currency, commission_rate: Currency, currency_to_buy_in_usd: Currency) -> None:
        # Nan currency rate represent a closed market
        if not np.isnan(currency_rate.btc):
            self.btc = currency_to_buy_in_usd.btc * (1 - commission_rate.btc) / currency_rate.btc
            self.usd += currency_to_buy_in_usd.btc
        if not np.isnan(currency_rate.gold):
            self.gold = currency_to_buy_in_usd.gold * (1 - commission_rate.gold) / currency_rate.gold
            self.usd += currency_to_buy_in_usd.gold
        
@dataclass
class Sell(Action):
    def __init__(self, currency_rate: Currency, commission_rate: Currency, currency_to_sell: Currency) -> None:
        # Nan currency rate represent a closed market
        if not np.isnan(currency_rate.btc):
            self.btc = currency_to_sell.btc
            self.usd += (currency_rate.btc * currency_to_sell.btc) * (1 - commission_rate.btc)
        if not np.isnan(currency_rate.gold):
            self.gold = currency_to_sell.gold
            self.usd += (currency_rate.gold * currency_to_sell.gold) * (1 - commission_rate.gold)
        
@dataclass
class Hold(Action): pass

# Get current currency rates from dataframe. Rate will be numpy.nan if market is closed for that currency and include_nan is True. Use include_nan = False to get the latest valid currency rate even if the market is closed.
def get_currency_rate(dataframe, include_nan: bool = True):
    if include_nan: return Currency(1, dataframe.BTC.values[-1], dataframe.GOLD.values[-1])
    btc = dataframe.BTC.values
    gold = dataframe.GOLD.values
    return Currency(1, btc[~np.isnan(btc)][-1], gold[~np.isnan(gold)][-1])

# Run the trade simulation
def simulate(data, commission_rate, wallet, action_model):
    for day in data.index:
        # Only look at past stream of daily prices 
        stream_history = data[:day+1]
        
        # Debug output
        print("-" * 50)
        print(f"Day {day}")
        print(wallet)
        
        # Get action based on action model
        action = action_model(day, stream_history, commission_rate, wallet)
        
        # Adjust wallet to lock in the results of the chosen action
        if isinstance(action, Buy):
            wallet.usd -= action.usd
            wallet.btc += action.btc
            wallet.gold += action.gold
        elif isinstance(action, Sell):
            wallet.usd += action.usd
            wallet.btc -= action.btc
            wallet.gold -= action.gold

    # Convert all currency to USD to get estimate of portfolio's worth
    currency_rate = get_currency_rate(data, include_nan=False)
    action = Sell(currency_rate, commission_rate, Currency(0, wallet.btc, wallet.gold))
    return wallet.usd + action.usd

# Buy bitcoin and hold
def null_action_btc(day, stream_history, commission_rate, wallet) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if wallet.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, wallet.usd, 0))
    
# Buy gold and hold
def null_action_gold(day, stream_history, commission_rate, wallet) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if wallet.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, 0, wallet.usd))    

def main():
    data = pd.read_csv("data/full.csv")
    commission_rate = Currency(1, 0.02, 0.01)
    wallet = Currency(1000, 0, 0)
    
    portfolio_worth = simulate(data, commission_rate, wallet, null_action_btc)
    print(f"Total USD: ${round(portfolio_worth, 2)}")
    
main()