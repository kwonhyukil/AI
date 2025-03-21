import matplotlib.pyplot as plt
import numpy as np

# 랜덤 데이터 생성
np.random.seed(42)  # 재현성을 위해 시드 설정
x = np.random.rand(50, 1) * 10  # 0.0 ~ 10.0 사이 값
y = np.array([2 * val + np.random.rand() * 9 for val in x])  # H(x) = 2x + noise

# 초기 가중치 설정
w = 0.1
learning_rate = 0.01
avg = []
epoch = 20  # 학습 반복 횟수

# 학습 과정
for _ in range(epoch):
    loss = 0.0
    slope_sum = 0

    for x_train, y_train in zip(x, y):
        # 현재 W에 대한 예측값
        pred_val = w * x_train

        # 손실 값 (MSE)
        loss += np.sum((pred_val - y_train) ** 2)

        # 기울기(Gradient) 계산
        slope_sum += np.sum(x_train * (w * x_train - y_train))

    # 가중치 업데이트 (경사 하강법 적용)
    w -= learning_rate * (2 * slope_sum / len(x))

    # 평균 손실 저장
    avg.append(loss / len(x))
    print(f"Epoch {_ + 1}, w: {w}")

# 결과 출력
print("최종 가중치 (w):", w)

# 데이터 산점도 & 학습된 선형 회귀 그래프 출력
plt.scatter(x, y, label="Data")
plt.grid()

# 회귀선 그리기
x_range = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = w * x_range
plt.plot(x_range, y_pred, color='red', label="Fitted Line")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
