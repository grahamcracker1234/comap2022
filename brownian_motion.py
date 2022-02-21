import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from readData import openFile, getValuesOnly
from forcasting import forcast_attempt

class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def stock_price(
                    self,
                    s0=100,
                    mu=0.2,
                    sigma=0.68,
                    deltaT=52,
                    dt=0.1
                    ):
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
        
        Arguments:
            s0: Iniital stock price, default 100
            mu: 'Drift' of the stock (upwards or downwards), default 1
            sigma: 'Volatility' of the stock, default 1
            deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
            dt (optional): The granularity of the time-period, default 0.1
        
        Returns:
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        n_step = int(deltaT/dt)
        time_vector = np.linspace(0,deltaT,num=n_step)
        # Stock variation
        stock_var = (mu-(sigma**2/2))*time_vector
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0=0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma*self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0*(np.exp(stock_var+weiner_process))
        
        return s


def plot_stock_price(mu,sigma, x):
    """
    Plots stock price for multiple scenarios
    """
    b = Brownian(x)
    plt.figure(figsize=(9,4))
    for i in range(5):
        stock = b.stock_price(mu=mu,
                               sigma=sigma,
                               deltaT=42,
                               s0=x,
                               dt=0.1)
        # print(stock)
        plt.plot(stock)
    plt.legend(['Scenario-'+str(i) for i in range(1,6)],
               loc='upper left')
    plt.hlines(y=100,xmin=0,xmax=520,
               linestyle='--',color='k')
    plt.show()

def predicts_stock_price(stock_value, volatility, num_weeks):
    # this function make the actual prediction of the data for the next 98 days
    # the volatility rate will need to change as we look from previous historic data
    model = Brownian(stock_value)
    return model.stock_price(s0=stock_value, sigma=volatility, deltaT= num_weeks)

def predict_final_stock_price(stock_value, volatility, num_weeks, num_scenarios):
    model = Brownian(stock_value)
    scenarios_predictions = []
    for i in range(num_scenarios):
        stock_prediction = model.stock_price(s0=stock_value, sigma=volatility, deltaT=num_weeks)
        scenarios_predictions.append(stock_prediction)
    
    final_prediction = []
    for i in range(len(scenarios_predictions[0])):
        values = []
        for j in range(len(scenarios_predictions)):
            values.append(scenarios_predictions[j][i])
        final_prediction.append(np.mean(values))

    return final_prediction

def volatility(data, day_num):
    # this is a function that will define the volatility
    # it will look at previous data and analyze it
    data = data[:day_num]
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    sigma = np.std(data)
    variance = sigma**2
    # print((sigma*2.6)+variance)
    return sigma*3 + variance

def volatility_complex(data, day_num):
    # use the data from the previous days until today
    data = data[:day_num+1]
    # get the sum of stock returns squared
    price_today = data[day_num]
    returns = [((data[i] - data[i-1])/data[i-1]) for i in range(1,len(data))]
    miu = np.mean(returns)
    sigma = np.std(returns)
    return sigma

def simulate_prediction(data, day_num):
    # simulates the prediction
    sigma = volatility(data, day_num)
    # the parameters can change in terms of number of weeks to check 
    # the num of scenarios will also change and it will eventually lead to the
    # natural distribution of the brownian motion
    stock_prices_f = predict_final_stock_price(stock_value=data[day_num], volatility=sigma, num_weeks=5, num_scenarios=5)
    return stock_prices_f

def simulation_calculation(data, day_num):
    # get the prediction average
    stock_price_predictions_days_average = []
    for i in range(10):
        stock_price_predictions_days = simulate_prediction(data, day_num)
        stock_price_predictions_days_average.append(stock_price_predictions_days)
        
    # getting the average stock price predictions for the future (next 100 days)
    final_stock_prediction_days = []
    for i in range(len(stock_price_predictions_days_average[0])):
        values = []
        for j in range(len(stock_price_predictions_days_average)):
            values.append(stock_price_predictions_days_average[j][i])
        final_stock_prediction_days.append(np.mean(values))
    return final_stock_prediction_days

def simulateAllDays(data):
    all_predictions = []
    for day_num in range(1, len(data)-50, 50):
        stock_prediction_days = simulation_calculation(data, day_num)
        all_predictions.extend(stock_prediction_days)
    plt.plot(data)
    plt.plot(all_predictions)
    plt.show() 

data = pd.read_csv("data/full.csv")
btc = [float(data.BTC.values[i]) for i in range(len(data))]
gold = [float(data.GOLD_FULL.values[i]) for i in range(len(data))]
# print(btc[0])
simulateAllDays(btc)