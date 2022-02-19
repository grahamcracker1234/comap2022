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
def get_currency_rate(dataframe, include_nan=True):
    if include_nan: return Currency(1, dataframe.BTC.values[-1], dataframe.GOLD.values[-1])
    btc = dataframe.BTC.values
    btc = btc[~np.isnan(btc)]
    gold = dataframe.GOLD.values
    gold = gold[~np.isnan(gold)]
    if len(btc) > 0 and len(gold) > 0: return Currency(1, btc[-1], gold[-1])
    return Currency(1, np.nan, np.nan)

# Convert all currency to USD to get estimate of portfolio's worth
def get_portfolio_worth(dataframe, commission_rate, portfolio):
    currency_rate = get_currency_rate(dataframe, include_nan=False)
    action = Sell(currency_rate, commission_rate, Currency(0, portfolio.btc, portfolio.gold))
    return portfolio.usd + action.usd

# Run the trade simulation
def simulate(data, commission_rate, portfolio, action_model):
    for day in data.index:
        # Only look at past stream of daily prices 
        stream_history = data[:day+1]
        
        # Debug output
        print(f"Day {day}")
        print(f"Portfolio Worth: ${round(get_portfolio_worth(stream_history, commission_rate, portfolio), 2)}")
        print(f"\tUSD: {portfolio.usd}")
        print(f"\tBTC: {portfolio.btc}")
        print(f"\tGOLD: {portfolio.gold}")
        print()
        
        # Get action based on action model
        action = action_model(day, stream_history, commission_rate, portfolio)
        
        # Adjust portfolio to lock in the results of the chosen action
        if isinstance(action, Buy):
            portfolio.usd -= action.usd
            portfolio.btc += action.btc
            portfolio.gold += action.gold
        elif isinstance(action, Sell):
            portfolio.usd += action.usd
            portfolio.btc -= action.btc
            portfolio.gold -= action.gold

    return get_portfolio_worth(data, commission_rate, portfolio)

# Buy both and hold
def null_action(day, stream_history, commission_rate, portfolio) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if np.isnan(currency_rate.btc) or np.isnan(currency_rate.gold) or portfolio.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, portfolio.usd / 2, portfolio.usd / 2))

# Buy bitcoin and hold
def null_action_btc(day, stream_history, commission_rate, portfolio) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if portfolio.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, portfolio.usd, 0))
    
# Buy gold and hold
def null_action_gold(day, stream_history, commission_rate, portfolio) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if portfolio.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, 0, portfolio.usd))    

def main():
    data = pd.read_csv("data/full.csv")
    commission_rate = Currency(1, 0.02, 0.01)
    portfolio = Currency(1000, 0, 0)
    
    portfolio_worth = simulate(data, commission_rate, portfolio, null_action)
    print(f"Total USD: ${round(portfolio_worth, 2)}")
    
main()