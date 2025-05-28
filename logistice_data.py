import numpy as np
# from sklearn.datasets import load_breast_cancer
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = np.arange(0, 10)

print(X)

print(sum(X))
avg = sum(X)/len(X)


deviation = 0

for item in X:
    deviation += item - avg

print(deviation)
































# # 1. 데이터 로드 및 분할
# dataset = load_breast_cancer()
# X = dataset.data
# y = dataset.target

# np.set_printoptions(suppress=True, precision=5)
# print(X[0, :])

# # 3. 특성 표준화 (평균 0, 분산 1)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)