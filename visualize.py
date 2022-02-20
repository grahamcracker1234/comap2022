import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime


def nearest_neighbor_smoother(x, y, n):
    smooth = []
    for xi in x:
        xs = ((x - xi).apply(lambda x: x.days) ** 2).sort_values()
        xs = xs[~np.isnan(y[xs.index])][:n]
        ys = y[xs.index]
        smooth.append(ys.sum() / n)
    return np.array(smooth)


def gaussian_kernel_smoother(x, y, b):
    smooth = []
    for xi in x:
        new_xi = np.exp(-((x - xi).apply(lambda x: x.days)
                        ** 2) / (2 * (b ** 2)))
        new_xi /= new_xi.sum()
        smooth.append((y * new_xi).sum())
    return np.array(smooth)


def main():
    data = pd.read_csv("data/full.csv", converters={"Date": pd.to_datetime})

    plt.plot(data.BTC)

    g_smooth = nearest_neighbor_smoother(data.Date, data.BTC, 20)
    plt.plot(g_smooth)

    g_smooth = gaussian_kernel_smoother(
        data.Date[~np.isnan(data.BTC)], data.BTC, 10)
    plt.plot(g_smooth)

    max_indices = argrelextrema(g_smooth, np.greater)[0]
    local_max = g_smooth[max_indices]
    plt.plot(max_indices, local_max, "ro")

    min_indices = argrelextrema(g_smooth, np.less)[0]
    local_min = g_smooth[min_indices]
    plt.plot(min_indices, local_min, "bo")

    plt.show()


if __name__ == "__main__":
    main()
