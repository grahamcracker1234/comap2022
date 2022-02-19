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
        if not np.isnan(currency_rate.btc):
            self.btc = currency_to_buy_in_usd.btc * (1 - commission_rate.btc) / currency_rate.btc
            self.usd += currency_to_buy_in_usd.btc
        if not np.isnan(currency_rate.gold):
            self.gold = currency_to_buy_in_usd.gold * (1 - commission_rate.gold) / currency_rate.gold
            self.usd += currency_to_buy_in_usd.gold
        
@dataclass
class Sell(Action):
    def __init__(self, currency_rate: Currency, commission_rate: Currency, currency_to_sell: Currency) -> None:
        if not np.isnan(currency_rate.btc):
            self.btc = currency_to_sell.btc
            self.usd += (currency_rate.btc * currency_to_sell.btc) * (1 - commission_rate.btc)
        if not np.isnan(currency_rate.gold):
            self.gold = currency_to_sell.gold
            self.usd += (currency_rate.gold * currency_to_sell.gold) * (1 - commission_rate.gold)
        
@dataclass
class Hold(Action): pass

def get_currency_rate(dataframe):
    return Currency(1, dataframe.BTC.values[-1], dataframe.GOLD.values[-1])

def null_action_btc(day, stream_history, commission_rate, wallet) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if wallet.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, wallet.usd, 0))
    
def null_action_gold(day, stream_history, commission_rate, wallet) -> Action:
    currency_rate = get_currency_rate(stream_history)
    if wallet.usd == 0: return Hold()
    return Buy(currency_rate, commission_rate, Currency(0, 0, wallet.usd))    
    
    
def simulate(data, commission_rate, wallet):
    data = pd.read_csv("data/full.csv")
    commission_rate = Currency(1, 0.02, 0.01)
    wallet = Currency(1000, 0, 0)

    for day in data.index:
        stream_history = data[:day+1]
        print("-" * 50)
        print(f"Day {day}")
        print(wallet)
        action = null_action_gold(day, stream_history, commission_rate, wallet)
        if isinstance(action, Buy):
            wallet.usd -= action.usd
            wallet.btc += action.btc
            wallet.gold += action.gold
        elif isinstance(action, Sell):
            wallet.usd += action.usd
            wallet.btc -= action.btc
            wallet.gold -= action.gold

    # Convert all currency to USD to get estimate of portfolio's worth
    currency_rate = get_currency_rate(data)
    action = Sell(currency_rate, commission_rate, Currency(0, wallet.btc, wallet.gold))
    return wallet.usd + action.usd

def main():
    data = pd.read_csv("data/full.csv")
    commission_rate = Currency(1, 0.02, 0.01)
    wallet = Currency(1000, 0, 0)
    portfolio_worth = simulate(data, commission_rate, wallet)
    print(f"Total USD: ${round(portfolio_worth, 2)}")
    
main()