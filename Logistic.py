import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드 및 분할
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
learning_rate = 0.01
# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1] # 특성 개수 (30개)

y_train = y_train.reshape(-1, 1) # y_train을 열 벡터로 변환

w = np.random.randn(num_features, 1) # 가중치 초기화
b = np.random.randn() # 편향 초기화

# z = wx + b
z = X_train @ w + b # @는 행렬 곱셈

# prediction = 1 / (1 + e^(-z))
prediction = 1 / (1 + np.exp(-z)) # 시그모이드 함수

print(prediction)
print(prediction.shape)
# error = prediction - y_train
error = prediction - y_train # 예측값과 실제값의 차이

# gradient descent_w, gradient descent_b
gradient_w = X_train.T @ error / len(X_train) # 가중치에 대한 기울기
gradient_b = error.mean() # 편향에 대한 기울기

print(gradient_w.shape)
print(gradient_b.shape)
# update parameters : w, b
# calculate loss
w = w -learning_rate * gradient_w # 가중치 업데이트
b = b - learning_rate * gradient_b # 편향 업데이트

loss = np.mean(-y_train * np.log(prediction) +    (1 - y) * np.log(1 - prediction))
# 4. 로지스틱 회귀 모델 정의
