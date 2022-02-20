## create a function that reads a file contining current bitcoin and gold prices
## and returns a forcast of future prices using ARIMA
from posixpath import split
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def read_file(file_name):
    df = pd.read_csv(file_name, index_col=0, parse_dates=True)
    df["BTC"] = df["BTC"].astype(float)
    df["GOLD"] = df["GOLD"].astype(float)
    return df


def split_data(data):
    train = data[: int(len(data) * 0.4)]
    test = data[int(len(data) * 0.6) :]
    return train, test


def forcast():
    data = read_file("data/full.csv")
    df = data.BTC
    train, test = split_data(df)
    model = ARIMA(df, order=(1, 1, 1))
    model_fit = model.fit()
    print(model_fit.summary())
    forcast_value = model_fit.predict(len(train), len(train), typ="levels")
    print(forcast_value.values[0])

def forcast_attempt(data):
    train, test = split_data(data)
    model = ARIMA(data, order=(1,1,1))
    model_fit = model.fit()
    forcast_value = model_fit.predict(len(train), len(train), typ="levels")
    return forcast_value



if __name__ == "__main__":
    forcast()
