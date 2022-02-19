import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/full.csv")

plt.plot(data.BTC)
plt.plot(data.GOLD)
plt.show()