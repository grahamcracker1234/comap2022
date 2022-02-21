import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

from scipy.signal import argrelextrema
import geometric_brownian_motion as gbm
from smoother import exponential_weighted_mean as ewm, gaussian_kernel_smoother as gks

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

# Geometric Brownian Motion with Exponential Weighted Mean Action Heuristic
def gbm_with_ewm_action(day, stream_history, commission_rate, portfolio) -> Action:
    currency_rate = get_currency_rate(stream_history)
    buy_action = Buy(currency_rate, commission_rate, Currency(0, portfolio.usd, 0))
    sell_action = Sell(currency_rate, commission_rate, Currency(0, portfolio.btc, 0))

    if day == 0: return buy_action
    
    if day % 25 != 0: return
    
    # currency_rate = get_currency_rate(stream_history)
    btc_prediction = gbm.geometric_brownian_motion(stream_history.BTC, 100)
    
    prediction_stream = np.concatenate((stream_history.BTC.values, btc_prediction), axis=None)
    smooth_stream = ewm(prediction_stream)
    smooth_stream = gks(np.array(list(range(smooth_stream.size))), smooth_stream)

    min_indices = argrelextrema(smooth_stream, np.less)[0]
    local_mins = smooth_stream[min_indices]
    
    max_indices = argrelextrema(smooth_stream, np.greater)[0]
    local_maxes = smooth_stream[max_indices]
    
    indices = np.concatenate((min_indices, max_indices), axis=None)
    most_recent_past_index = max(filter(lambda i: i <= day, indices), default=None)
    most_recent_future_index = min(filter(lambda i: i > day, indices), default=None)
    
    # plt.plot(data.BTC)
    # plt.plot(prediction_stream)
    # plt.plot(smooth_stream)
    # plt.plot(max_indices, local_maxes, "ro")
    # plt.plot(min_indices, local_mins, "bo")
    # plt.axvline(x=day)
    # plt.show()
        
    # print(f"{max_indices=}")
    # print(f"{min_indices=}")
    # print(f"{most_recent_past_index=}")
    
    if most_recent_past_index in min_indices and most_recent_past_index in max_indices: 
        raise Exception("most_recent_past_index is both local_min and local_max")
    
    buy_action = Buy(currency_rate, commission_rate, Currency(0, portfolio.usd, 0))
    if most_recent_past_index in min_indices:
        min_index = most_recent_past_index
        max_index = most_recent_future_index
        
        min_value = local_mins[min_index]
        max_value = local_maxes[max_index]
        
        if abs(max_value - min_value) / most_recent_future_index < commission_rate.btc:
            return Hold()
        return buy_action
    
    if most_recent_past_index in max_indices:
        min_index = most_recent_future_index
        max_index = most_recent_past_index
        
        min_value = local_mins[min_index]
        max_value = local_maxes[max_index]
        
        if abs(max_value - min_value) / most_recent_future_index < commission_rate.btc:
            return Hold()
        return sell_action
    
    # plt.plot(data.BTC)
    # plt.plot(prediction_stream)
    # plt.plot(smooth_stream)
    # plt.plot(max_indices, local_max, "ro")
    # plt.plot(min_indices, local_min, "bo")
    # plt.axvline(x=day)
    # plt.show()
    
    # print(stream_history)
    # print(btc_prediction)

def main():
    global data
    data = pd.read_csv("data/full.csv", converters={"DATE": pd.to_datetime})
    commission_rate = Currency(1, 0.02, 0.01)
    portfolio = Currency(1000, 0, 0)
    
    portfolio_worth = simulate(data, commission_rate, portfolio, gbm_with_ewm_action)
    print(f"Total USD: ${round(portfolio_worth, 2)}")
    
if __name__ == "__main__":
    main()
    
    
# def policyFunction(Q, epsilon, actions):
#     def policy_function(state):
#         action_probs = np.ones(len(actions)) * epsilon / len(actions)
#         best_action = np.argmax(Q[state])
#         action_probs[best_action] += epsilon / len(actions)
#         return action_probs
#     return policy_function


# def step(state, action_function):
#     action_probs = action_function(state)
#     action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
#     next_state = action_function(action_function(state))
#     return next_state