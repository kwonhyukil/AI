import numpy as np

num_features = 4
num_samples = 1000

# 일정한 값 유지
np.random.seed(5)

X = np.random.rand(num_samples, num_features) * 2
w_true = np.random.randint(1, 11, (num_features, 1))
b_true = np.random.rand() * 0.5

Y = X @ w_true + b_true

print(Y)

##############################################################

# 초기의 w 값은 달라야한다
w = np.random.rand(num_features, 1)
b = np.random.rand()
learning_rate = 0.01

print(f"W: {w} \n B: {b}")
for _ in range(11000):
    # 예측 값
    predict_y = X @ w +b

    # 오차
    error = predict_y - Y

    # 기울기
    gradinet_w = X.T @ error / num_samples
    gradinet_b = error.mean()

    # w, b 업데이트
    w = w - learning_rate * gradinet_w
    b = b - learning_rate * gradinet_b

print(f"W: {w},\n B: {b}")


print(gradinet_w.shape)
print(gradinet_b.shape)


# # 1 ~ 3 사이의 수를 생성
# x = np.random.randint(1, 4, (2, 2)) #vector # 1차원
# y = np.random.randint(1, 4, (2, 2))


# print(f"정상\n{x}\n\n{y}\n{x+y}")

# x = np.random.randint(1, 4, (2, 2)) #vector # 1차원
# y = np.random.randint(1, 4, (2, 1))


# print(f"열 1개\n{x}\n\n{y}\n{x+y}")

# x = np.random.randint(1, 4, (2, 2)) #vector # 1차원
# y = np.random.randint(1, 4, (2, 2))


# print(f"행 1개\n{x}\n\n{y}\n{x+y}")

# # w = np.random.randint(1, 4, (2,))

# # print(f"{w}\n{w + 2}\n{w - 2}\n{w * 2}\n {w / 2}")

# Q = np.array([[1, 2], [3, 4]])

# E = np.array([[2], [3]])

# print(x.shape)
# print(y.shape)

# print(f"{Q}\n{E}\n{Q@E + 2}")

