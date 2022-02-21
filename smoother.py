import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from datetime import datetime
# from sklearn.kernel_approximation import RBFSampler
# from sklearn import preprocessing
from timeit import timeit
from itertools import zip_longest

# from skfda.preprocessing.smoothing

# Generate dataframe column without nan values
def generate_column(column):
    new_column = []
    last_value = np.nan
    for y in column:
        if np.isnan(y) and last_value: 
            new_column.append(last_value)
            continue
        
        if ~np.isnan(y): 
            last_value = y
            new_column.append(last_value)
            continue
            
        new_column.append(np.nan)
    
    first_value = np.nan
    first_index = 0
    for i, y in enumerate(new_column):
        if np.isnan(y): continue
        first_value = y
        first_index = i
        break
    
    for i in range(first_index):
        new_column[i] = first_value
        
    return np.array(new_column)
        
# Slow
def nearest_neighbor_smoother(x, y, n):
    smooth = []
    for xi in x:
        xs = ((x - xi).apply(lambda x: x.days) ** 2).sort_values()
        ys = y[xs.index][:n]
        smooth.append(ys.sum() / n)
    return np.array(smooth)

# Fast, but wrong width
# def nearest_neighbor_smoother(x, y, n):
#     ys = tuple(y[i:] for i in range(n))
#     smooth = tuple(sum(y) / n for y in zip(*ys))
#     return np.array(smooth)

# def nearest_neighbor_smoother(x, y, n):
#     ys = tuple(y[i:] for i in range(n))
#     smooth = tuple(sum(fy := tuple(filter(lambda yi: ~np.isnan(yi), y))) / len(fy) for y in zip_longest(*ys, fillvalue=np.nan))
#     return np.array(smooth)


def exponential_weighted_mean(y, halflife=14):
    return np.array(pd.Series(y).ewm(halflife=halflife).mean().values)

# def gaussian_kernel_smoother(x, y, b=10):
#     smooth = []
#     for xi in x:
#         new_xi = np.exp(-((x - xi).apply(lambda x: x.days) ** 2) / (2 * (b ** 2)))
#         new_xi /= new_xi.sum()
#         smooth.append((y * new_xi).sum())
#     return np.array(smooth)

def gaussian_kernel_smoother(x, y, b=10):
    smooth = []
    for xi in x:
        new_xi = np.exp(-((x - xi) ** 2) / (2 * (b ** 2)))
        new_xi /= new_xi.sum()
        smooth.append((y * new_xi).sum())
    return np.array(smooth)

def main():
    data = pd.read_csv("data/full.csv", converters={"DATE": pd.to_datetime})
    np.set_printoptions(threshold=np.inf)
    x = data.DATE
    y = data.BTC
    plt.plot(y)

    # n_smooth = nearest_neighbor_smoother(x, y, 300)
    n_smooth = y.ewm(halflife=14).mean().values #nearest_neighbor_smoother(x, y, 300)
    n_smooth = gaussian_kernel_smoother(np.array(list(range(n_smooth.size))), n_smooth, 10) #nearest_neighbor_smoother(x, y, 300)
    plt.plot(n_smooth)
    # print(x.size)
    # print(n_smooth.size)
    
    # print(timeit(lambda: nearest_neighbor_smoother(x, y, 250), number=1))
    
    # print(y.describe())
    # norm_y = (y - y.mean()) / y.std()
    # print(norm_y.describe())
    # norm_y = (y - y.min()) / (y.max() - y.min())
    # print(norm_y.describe())
    # print(norm_y)

    # print(timeit(lambda: nearest_neighbor_smoother(x, y, 50), number=1))
    # print(timeit(lambda: gaussian_kernel_smoother(x, y, 10), number=1))
    # g_smooth = gaussian_kernel_smoother(x, y, 10)
    # plt.plot(g_smooth)

    max_indices = argrelextrema(n_smooth, np.greater)[0]
    local_max = n_smooth[max_indices]
    plt.plot(max_indices, local_max, "ro")

    min_indices = argrelextrema(n_smooth, np.less)[0]
    local_min = n_smooth[min_indices]
    plt.plot(min_indices, local_min, "bo")

    plt.show()


if __name__ == "__main__":
    main()
