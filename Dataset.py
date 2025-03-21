import numpy as np
import matplotlib.pyplot as plt


# Data set
# input
# input data, features
# H(x) -> input data : x1 -> xn
x_train = [ np.random.rand() * 10 for _ in range(50)]
y_train = [ val * np.random.rand() * 5 for val in x_train]

print(x_train)
print("-" * 100)
print(y_train)

# BGC (Batch Gradient Descent) 배치경사하강법을
# 이용하여 Linear Rergeresion 적용


plt.scatter(x_train,y_train)
plt.show()
# Output
# label
# f(x1) 0> f(x2)