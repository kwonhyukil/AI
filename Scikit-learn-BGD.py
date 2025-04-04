import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 데이터 생성
x = np.random.rand(100, 1) * 10
y = 2.5 * np.random.rand(100, 1) * 4

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 모델 학습
model = LinearRegression()
model.fit(x_train,y_train)

# 예측 및 평가
y_pred = model.predict(x_test)

# 플롯 준비
fig, axes = plt.subplots(1, 2, figsize=(25, 10))

# 원본 데이터 산점도
axes[0].scatter(x, y, color="blue", label="Original Data")
axes[0].set_title('Scatter plot of data')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend()

# 회귀선 및 데이터 산점도
x_val = np.linspace(0, 10, 100)
y_val = model.coef_[0, 0] * x_val + model.intercept_[0]

axes[1].scatter(x, y, color='blue', label='Original Data')
axes[1].plot(x_val, y_val, color='red', label='Regression Line')
axes[1].set_title('Linear Regression')
axes[1].legend()

plt.tight_layout()
plt.show()

# 회귀식 출력
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mean_squared_error(y_test, y_pred)}, 회귀 계수: {model.coef_[0, 0]}, 절편: {model.intercept_[0]}")