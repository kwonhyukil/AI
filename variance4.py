from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as pit
import numpy as np

x = np.arange(10)  # [0, 1, 2, ..., 9]

print(x.mean(), x.std())

mean = x.mean()

values = [ item - mean for item in x ]
# values = [float(item - mean) for item in x]

print(values, sum(values))
