import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(50, 1) * 10 # 0.0 ~ 10.0
y = [2*val + np.random.rand() * 9 for val in x]

print(x)
print()
print(y)

plt.scatter(x, y)
plt.grid()
plt.show()