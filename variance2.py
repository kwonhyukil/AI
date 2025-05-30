from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

np.random.randint(1, 100)

np.random.randn() # 평균이 0이고, 표준편차가 1인 정규분포 난수 생성하기

values = [ np.random.randn() * 10 + 50 for _ in range(10000)]

plt.hist(values, bins=40)

plt.show()

print(values)

