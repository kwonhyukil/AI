from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터셋 로딩 및 분할
digits = load_digits()
features = digits.data                    # (1797, 64): 8x8 이미지 벡터
labels = digits.target                    # (1797,): 0~9 클래스 정수

print(features)
print(labels.shape)
print(labels[:30])

# 2. 학습/테스트 셋 분할
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3. 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

num_features = X_train_std.shape[1] # 64 특성 개수
num_samples = X_train_std.shape[0] # 1438 훈련 샘플 개수
num_classes = 10  # 클래스 개수 (0-9)

# 정규 분포에 따르는 가중치 randn 사용 (rand 과 randn의 차이)
W = np.random.randn(num_features, num_classes)  # (64, 10): 가중치 초기화

print(f"가중치 행렬 크기: {W.shape}")

# 0.8은 훈련 데이터로 사용, 0.2는 테스트 데이터로 사용
print(f"훈련 데이터 크기: {X_train_std.shape}, 테스트 데이터 크기: {X_test_std.shape}")

















# # 4. 소프트맥스 회귀 모델 정의
# num_features = X_train_std.shape[1]  # 특성 개수 (64)
# num_classes = 10  # 클래스 개수 (0-9)
# # 가중치 및 편향 초기화
# weights = np.random.randn(num_features, num_classes)  # (64, 10)
# bias = np.random.randn(num_classes)  # (10,)
# # 학습률 및 에폭 설정
# learning_rate = 0.01
# epochs = 300
# # 5. 소프트맥스 회귀 학습
# for epoch in range(epochs):
#     # z = wx + b
#     logits = X_train_std @ weights + bias  # (n_samples, n_classes)
#     logits -= logits.max(axis=1, keepdims=True)  # 오버플로우 방지

#     # 소프트맥스 계산
#     exp_logits = np.exp(logits)  # e^z
#     sum_exp = np.sum(exp_logits, axis=1, keepdims=True)  # 각 샘플의 지수 합
#     softmax = exp_logits / sum_exp  # 소프트맥스 확률

#     # 오차 계산
#     y_train_one_hot = np.eye(num_classes)[y_train]  # 원-핫 인코딩
#     error = softmax - y_train_one_hot  # 예측값과 실제값의 차이

#     # 가중치 및 편향 기울기 계산
#     gradient_weights = X_train_std.T @ error / len(X_train_std)  # (n_features, n_classes)
#     gradient_bias = error.mean(axis=0)  # (n_classes,)

#     # 파라미터 업데이트
#     weights -= learning_rate * gradient_weights
#     bias -= learning_rate * gradient_bias

#     # 손실 계산
#     loss = -np.sum(y_train_one_hot * np.log(softmax + 1e-15)) / len(X_train_std)

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Loss: {loss:.4f}")
# # 6. 테스트 데이터에 대한 예측
# test_logits = X_test_std @ weights + bias
# test_logits -= test_logits.max(axis=1, keepdims=True)  # 오버플로우 방지
# test_exp_logits = np.exp(test_logits)
# test_sum_exp = np.sum(test_exp_logits, axis=1, keepdims=True)  # 각 샘플의 지수 합
# test_softmax = test_exp_logits / test_sum_exp  # 소프트맥스 확률
# # 예측 클래스
# y_pred = np.argmax(test_softmax, axis=1)  # (n_samples,)
# # 정확도 계산
# accuracy = np.mean(y_pred == y_test)
# print(f"Test Accuracy: {accuracy:.4f}")
# # 7. 결과 시각화
# def plot_digits(X, y, num_images=10):
#     plt.figure(figsize=(10, 4))
#     for i in range(num_images):
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(X[i].reshape(8, 8), cmap='gray')
#         plt.title(f'Label: {y[i]}')
#         plt.axis('off')
#     plt.tight_layout()
#     plt.show()
# # 시각화
# plot_digits(X_test, y_test, num_images=10)
