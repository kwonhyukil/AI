import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [2, 4, 6, 8]

# 초기 가중치 설정
w = 0.1
learning_rate = 0.001  # 학습률
avg = []

# 20번 반복 (Epoch)
for _ in range(20):
    loss = 0.0  # 손실 초기화
    slope_sum = 0  # 기울기 합 초기화

    for x_train, y_train in zip(x, y):
        # 현재 W에 대한 예측값
        pred_val = w * x_train

        # 손실 값 (MSE)
        loss += (pred_val - y_train) ** 2

        # 기울기(Gradient) 계산
        slope_sum += x_train * (w * x_train - y_train)

    # 평균 기울기 적용 (경사 하강법)
    w -= learning_rate * (2 * slope_sum / len(x))

    # 평균 손실 저장
    avg.append(loss / len(x))

# 결과 출력
print("최종 가중치 (w):", w)

# 손실 그래프 출력
plt.plot(range(1, 21), avg, marker='o', label="Loss")  # x축을 1~20으로 수정
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Iterations")
plt.legend()
plt.show()
