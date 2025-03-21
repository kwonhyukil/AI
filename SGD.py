import numpy as np
import matplotlib.pyplot as plt

# 가상의 손실 함수: L(w) = (w - 3)^2
def loss(w):
    return (w - 3) ** 2

# 그래디언트 함수
def grad(w):
    return 2 * (w - 3)

# SGD 시뮬레이션
def simulate_sgd(epochs=50, lr=0.1):
    w = np.random.randn()  # 랜덤 초기화
    losses = []
    for epoch in range(epochs):
        noise = np.random.randn() * 0.5  # SGD의 불규칙성 표현
        g = grad(w + noise)
        w -= lr * g
        losses.append(loss(w))
    return losses

# 실행
epochs = 50
sgd_losses = simulate_sgd(epochs)

# 그래프 출력
plt.figure(figsize=(10, 5))
plt.plot(range(epochs), sgd_losses, label='SGD Loss', linestyle='--', marker='o', color='orange')
plt.title('SGD 손실 값 변화')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
