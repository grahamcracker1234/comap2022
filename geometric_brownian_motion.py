import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/full.csv", converters={"DATE": pd.to_datetime})

def geometric_brownian_motion(data, day_num, horizon_length, num_scenarios):
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
    stock_data = [float(data.BTC.values[i]) for i in range(len(data))]
    stock_data = stock_data[day_num: day_num+horizon_length]
    stock_value = stock_data[0]
    dt = 1
    T = horizon_length
    N = T/dt
    t = np.arange(1, int(N) + 1)

    # before calculating mu, need to calculate the return for each day 
    returns = [(stock_data[i] - stock_data[i-1])/stock_data[i-1] for i in range(len(stock_data))]
    print(f"returns: \n{returns}")

    # calculating mu --> it will be used in the drift component calculation
    mu = np.mean(returns)
    print(f"mu: {mu}")

    # calculating sigma
    sigma = np.std(returns)
    print(f"sigma: {sigma}")

    # calculating b, it depends on the number of scenarios
    b = {str(scenario): np.random.normal(0,1, int(N)) for scenario in range(1, num_scenarios + 1)}
    print(f"b: Random values with scenarios\n{b}")

    # calculating W, important to realize that it differs from b because is the random shock being applied to the
    # stock price at a time point when predicting the stock price of the next time point. Meaning that
    # at time point 3 --> 3rd stock price it predicts time point 4. 
    # W on the other hand is the path (total effect of randomness incorporated into stock_value)
    # it will depend on the number of scenarios again
    W = {str(scenario): np.cumsum(b[str(scenario)]) for scenario in range(1, num_scenarios+1)}
    print(f"W: Brownian path with scenarios\n{W}")

    # components of the Geometric Brownian Motion
    # -> Long-term trend in the stock_prices --> Drift
    # -> Short-term random fluctuations --> Diffusion
    # Difussion is what gives the stochastiness to our model

    # assumptions:
    # 1) length of the time period between (k-1) and (k), which is dt, is in line with the historical data frequency. 
    # For k being the time point within the length of the time period.
    # 2) the time in our simulation progresses through counting time periods.


    # the Geometric Brownian Motion model uses the initial stock_value given with the combination of 
    # k many drifts and the cumulative diffusion up to k. Look the equations in: https://towardsdatascience.com/simulating-stock-prices-in-python-using-geometric-brownian-motion-8dfd6e8c6b18

    # calculating drift
    drift = (mu - 0.5*sigma**2)*t
    print(f"Drift in t time steps: {drift}")

    # calculating diffusion, it takes into account the different scenarios
    diffusion = {str(scenario): sigma*W[str(scenario)] for scenario in range(1, num_scenarios+1)}
    print(f"Diffusion in different W scenarios: {drift}")

    # predictions per scenario
    Stock_Prediction = np.array([stock_value * np.exp(drift + diffusion[str(scen)]) for scen in range(1, num_scenarios + 1)])
    Stock_Prediction = np.hstack((np.array([[stock_value] for scenario in range(num_scenarios)]), Stock_Prediction))

    # final prediction calculation (mean of all scenarios prediction)
    final_Stock_Prediction = []
    for i in range(len(Stock_Prediction[0])):
        predictions = []
        for j in range(num_scenarios): 
            predictions.append(Stock_Prediction[j][i])
        final_Stock_Prediction.append(np.mean(predictions))

    print(f"Final Stock Prediction: \n{final_Stock_Prediction}")
    plt.plot(stock_data)
    plt.plot(final_Stock_Prediction)
    plt.show()



geometric_brownian_motion(data=data, day_num=200, horizon_length=50, num_scenarios=10)


