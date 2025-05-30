import numpy as np

x = np.arange(10) 

x_sum = sum(x)
x_avg = x_sum / len(x)

# 분산 계산
squared_deviation = [(xi - x_avg) ** 2 for xi in x]
variance = sum(squared_deviation) / len(x)

print(squared_deviation)
print(variance)

# 분산 (바닐라 버전)
variance = 0.0
for item in x:
    variance += (item - x_avg) **2
variance /= len(x)
std = np.sqrt(variance)

print(x_avg, variance, std)

# 분산 (numpy 사용)

np_avg = x.mean()
np_variance = x.var()
np_std = x.std()
print(np_avg, np_variance, np_std)