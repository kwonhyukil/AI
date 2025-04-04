import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# x : 0 ~ 10 사이 무작위 수 100개
# y : 2.5 + 약간의 노이즈

np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(10, 1) * 10
# H(x) = w * x + b
pos = 2.5 * X
bar = np.random.rand(10, 1) * 2
y = 2.5 * X + bar
print(X)
print("---" * 10)
print(pos)
print("---" * 10)
print(bar)
print("---" * 10)
print(y)
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2.5 * np.random.rand(100, 1) * 2
y = y.ravel()

model = SGDRegressor(max_iter=1000,
                     learning_rate='constant',
                     eta0=0.01,
                     penalty=None,
                     random_state=0)

model.fit(X, y)

np.set_printoptions(suppress=True, precision=2)


bar = np.random.rand(2, 3)
print(bar)
print("---" * 10)

bar = bar * 10
print(bar)

bar = np.zeros((2))
foo = np.zeros((3, 2))
pos = np.zeros((2, 3, 2))

print(bar)
print()
print(foo)
print()
print(pos)
print(f"bar.shape: {bar.shape}")

print(f"foo.shape: {foo.shape}")

print(f"pos.shape: {pos.shape}")