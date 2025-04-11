import numpy as np

num_of_samples = 5
num_of_features = 2

# data set
# H(x) = 5x + 3X + 3

np.random.seed(1)
np.set_printoptions(False, suppress=True)
x = np.random.rand(num_of_samples, num_of_features) * 10
# print(x)
x_true = [5, 3]
b_true = 4
noise = np.random.rand(num_of_samples) * 0.5

y = x[:, 0] * 5 + x[:, 1] * 3 + b_true + noise


print(y)
print(x)
print(x[:, 0]*5)
print(x[:, 1]*3)
print(x[:, 0] * 5 + x[:, 1] * 3 + b_true)