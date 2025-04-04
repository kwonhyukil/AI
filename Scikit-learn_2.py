import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error

# x : 0 ~ 10 사이 무작위 수 100개
# y : 2.5 + 약간의 노이즈

np.set_printoptions(suppress=True, precision=1)
X = np.random.rand(10, 1) * 10
y = 2.5 * X + np.random.rand(10, 1) * 2
y = y.ravel()

# 모델 생성 후 하이퍼파라미터 생성

model = SGDRegressor( # epoch 수 * 샘플수 = 업데이트 횟수수
    max_iter=1000, # 학습 반복 횟수 (epoch 수)
    learning_rate='constant',   ## 세트
    eta0=0.01, # 고정 학습률     ## 세트
    # 학습률
    # constant   : 1. 고정 학습률   : 학습률을 고정
    # invscaling : 2. 학습률 감소   : 에폭이 지날수록 학습률을 점점 줄여서 더 안정적인 수렴 유도
    # adapive    : 3. 적응형 학습률  : 각 파라미터에 따라 학습률을 자동 조정

    penalty=None, # 정규화 없음
    random_state=0
)

# 학습 실시 !!
model.fit(X, y) # 모델 학습

# 평가 !!
# Loss 
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print(f"평균 제곱 오차(MSE): {mse:.4f}")

bar = np.random.rand(3, 1)
print(bar)
print("--" * 10)
print(bar.ravel())



