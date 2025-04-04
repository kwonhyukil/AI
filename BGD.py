import numpy as np
import random
import matplotlib.pyplot as plt


# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [val + np.random.rand() * 5 for val in x_train]

# BGD
# 1. H(X) = w * x

# 2. optimizer [GD] : W = w -slope of cost function for given W

# 2-1 optimizer [BGD] : W = w - ?
# 2-1 optimizer [SGD] : W = w - ?