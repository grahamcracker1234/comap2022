from turtle import color
from typing import final
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def geometric_brownian_motion(stock_data, day_num, horizon_length, num_scenarios):
    '''
    Parameters:
    stock_value: The stock value at this day
    dt: time unit increment (number of times the stock price change per day), in our case 1
    T: length of the horizon we are predicting, for testing purposes we want to test what gives the best outcomes
    N: number of time points in the prediction horizon (T/dt) 
    t: array with the time points to show the time progression in the prediction
    mu: mean return of the stock prices within the historical range
    sigma: std of returns, it will be useful for the random shocks in our prediction, it helps understand the magnitude of the movement
    b: array where we add randomness (stochastiness) to our model, it depends on the number of scenarios simulated
    W: the brownian path initiating from the stock_value 
    '''
    stock_value = stock_data[day_num]
    dt = 1
    T = horizon_length
    N = T/dt
    t = np.arange(1, int(N) + 1)

    # before calculating mu, need to calculate the return for each day 
    returns = [(stock_data[i] - stock_data[i-1])/stock_data[i-1] for i in range(1,len(stock_data))]
    # print(f"returns: \n{returns}")

    # calculating mu --> it will be used in the drift component calculation
    mu = np.mean(returns)
    # print(f"mu: {mu}")

    # calculating sigma
    sigma = np.std(returns)
    # print(f"sigma: {sigma}")

    # calculating b, it depends on the number of scenarios
    b = {str(scenario): np.random.normal(0,1, int(N)) for scenario in range(1, num_scenarios + 1)}
    # print(f"b: Random values with scenarios\n{b}")

    # calculating W, important to realize that it differs from b because is the random shock being applied to the
    # stock price at a time point when predicting the stock price of the next time point. Meaning that
    # at time point 3 --> 3rd stock price it predicts time point 4. 
    # W on the other hand is the path (total effect of randomness incorporated into stock_value)
    # it will depend on the number of scenarios again
    W = {str(scenario): b[str(scenario)].cumsum() for scenario in range(1, num_scenarios+1)}
    # print(f"W: Brownian path with scenarios\n{W}")

    # components of the Geometric Brownian Motion
    # -> Long-term trend in the stock_prices --> Drift
    # -> Short-term random fluctuations --> Diffusion

    # Drift:
    # Long-term trend in stock prices, it is calculated by:
    # mu - half sigma squared. The importance of drift is that is being use as an exponential equation
    # for the stock price prediction
    # this component is constant for each day prediction because mu and sigma does not change

    # Difussion:
    # Short-term random shockness
    # the Diffusion component makes it possible to create different stock price prediction scenarios.
    # Diffusion is what gives the stochastiness to our model.
    # the diffusion component helps us create as many scenarios as we want since it involves 
    # Wiener process(It creates independent, stationary and normally distributed random shocks)
    # Diffusion also helps to decrease monotonus smoothness in increasing or decreasing trends in order to
    # maintain the random shockness.


    # assumptions:
    # 1) length of the time period between (k-1) and (k), which is dt, is in line with the historical data frequency. 
    # For k being the time point within the length of the time period.
    # 2) the time in our simulation progresses through counting time periods.


    # the Geometric Brownian Motion model uses the initial stock_value given with the combination of 
    # k many drifts and the cumulative diffusion up to k. Look the equations in: https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18

    # calculating drift
    drift = (mu - (0.5*sigma**2))*t
    # print(f"Drift in t time steps: {drift}")

    # calculating diffusion, it takes into account the different scenarios
    diffusion = {str(scenario): sigma*W[str(scenario)] for scenario in range(1, num_scenarios+1)}
    # print(f"Diffusion in different W scenarios: {drift}")

    # predictions per scenario
    Stock_Prediction = np.array([stock_value * np.exp(drift + diffusion[str(scen)]) for scen in range(1, num_scenarios + 1)])
    Stock_Prediction = np.hstack((np.array([[stock_value] for i in range(num_scenarios)]), Stock_Prediction))

    # final prediction calculation (mean of all scenarios prediction)
    final_Stock_Prediction = []
    for i in range(len(Stock_Prediction[0])):
        predictions = []
        for j in range(num_scenarios): 
            predictions.append(Stock_Prediction[j][i])
        final_Stock_Prediction.append(np.mean(predictions))

    final_Stock_Prediction.pop(0)
    return final_Stock_Prediction

def testing_prediction(stock_data_btc, stock_data_gold):
    prediction_full_btc, prediction_full_gold = [], []
    for day_num in range(0, len(stock_data_btc), 14):
        # bitcoin prediction
        final_stock_prediction = geometric_brownian_motion(data=stock_data_btc, day_num=day_num, horizon_length=14, num_scenarios=10)
        prediction_full_btc.extend(final_stock_prediction)

        # gold prediction
        final_stock_prediction = geometric_brownian_motion(data=stock_data_gold, day_num=day_num, horizon_length=14, num_scenarios=10)
        prediction_full_gold.extend(final_stock_prediction)
    # remove extra predictions
    for i in range(len(prediction_full_btc) - len(stock_data_btc)):
        prediction_full_btc.pop(len(prediction_full_btc)-1)
        prediction_full_gold.pop(len(prediction_full_btc)-1)

    prediction_full_btc, prediction_full_gold = np.array(prediction_full_btc), np.array(prediction_full_gold)
    rsme_btc = np.sqrt(np.mean((np.array(stock_data_btc) - prediction_full_btc)**2))
    rsme_gold = np.sqrt(np.mean((np.array(stock_data_gold) - prediction_full_gold)**2))
    print("\n"*3)
    print("#"*25)
    print(f"Root Square Mean Error for Bitcoin: {rsme_btc}\n")
    print("#"*25)
    print(f"Root Square Mean Error for Gold: {rsme_gold}\n")
    print("#"*25)
    print("\n"*3)

def get_prediction(stock_data, horizon_length, num_scenarios):
    prediction_full = []
    for day_num in range(0, len(stock_data)-horizon_length, horizon_length):
        # bitcoin prediction
        final_stock_prediction = geometric_brownian_motion(stock_data=stock_data, day_num=day_num, horizon_length=horizon_length, num_scenarios=num_scenarios)
        prediction_full.extend(final_stock_prediction)
    return prediction_full

def show_prediction(data, horizon_length, num_scenarios, graph_title):
    prediction_full = []
    for day_num in range(0, len(data)-horizon_length, horizon_length):
        # bitcoin prediction
        final_stock_prediction = geometric_brownian_motion(data=data, day_num=day_num, horizon_length=horizon_length, num_scenarios=num_scenarios)
        prediction_full.extend(final_stock_prediction)
    plt.plot(data, label=f"Original {graph_title} Data")
    plt.plot(prediction_full, label="Prediction Data")
    plt.title(f"{graph_title} Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("data/full.csv", converters={"DATE": pd.to_datetime})
    stock_data_btc = [float(data.BTC.values[i]) for i in range(len(data))]
    stock_data_gold = [float(data.GOLD_FULL.values[i]) for i in range(len(data))]
    testing_prediction(stock_data_btc, stock_data_gold)
    show_prediction(stock_data_btc, 14, 10, "Bitcoin")
