from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

np.set_printoptions(suppress=True)
# 1. 데이터 로드 및 분할
dataset = load_digits()
X = dataset.data
y = dataset.target

# 2. 훈련/테스트 셋 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 특성 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 기본 설정
num_features = X_train.shape[1]  # 특성 개수
num_samples = X_train.shape[0]  # 샘플 개수
num_classes = 10  # 클래스 개수 (0-9)

# 5. 원-핫 인코딩
one_hot = np.eye(num_classes)
y_train_one_hot = one_hot[y_train]
y_test_one_hot = one_hot[y_test]

# 6. 가중치 및 편향 초기화
learning_rate = 0.01
epochs = 300
w = np.random.randn(num_features, num_classes)  # 가중치 초기화
b = np.random.randn(num_classes)  # 편향 초기화

# 7. 소프트맥스 회귀 학습
for epoch in range(epochs):
    logits = X_train @ w + b  # z = wx + b
    logits -= logits.max(axis=1, keepdims=True)  # 오버플로우 방지

    exp_logits = np.exp(logits)  # e^z
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)  # 각 샘플의 지수 합
    softmax = exp_logits / sum_exp  # 소프트맥스 계산

    # 오차 계산
    error = softmax - y_train_one_hot  # 예측값과 실제값의 차이

    # 가중치 및 바이어스 기울기 계산
    gradient_w = X_train.T @ error / num_samples  # 가중치에 대한 기울기
    gradient_b = error.mean(axis=0)  # 편향에 대한 기울기

    # 경사하강법을 이용한 파라미터 업데이트
    w -= learning_rate * gradient_w  # 가중치 업데이트
    b -= learning_rate * gradient_b  # 편향 업데이트

    # 크로스 엔트로피 손실 계산
    loss = -np.sum(y_train_one_hot * np.log(softmax + 1e-15)) / num_samples

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 8. 테스트 데이터에 대한 예측
test_logits = X_test @ w + b
test_logits -= test_logits.max(axis=1, keepdims=True)  # 오버플로우 방지
exp_test_logits = np.exp(test_logits)
sum_exp_test = np.sum(exp_test_logits, axis=1, keepdims=True)  # 각 샘플의 지수 합
softmax_test = exp_test_logits / sum_exp_test  # 소프트맥스 계산

# 테스트 손실 계산
test_loss = -np.sum(y_test_one_hot * np.log(softmax_test + 1e-15)) / y_test_one_hot.shape[0]
print(f'Test Loss: {test_loss:.4f}')

# 예측 값
y_pred = np.argmax(softmax_test, axis=1)

# 정확도 계산
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.4f}')
print(f"Accuracy: {accuracy:.4f}%")