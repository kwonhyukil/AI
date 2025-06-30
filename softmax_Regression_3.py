# 💡 [필요한 라이브러리 불러오기]
from sklearn.datasets import load_digits              # 손글씨 숫자 이미지 데이터셋 (0~9)
from sklearn.model_selection import train_test_split  # 훈련/테스트 분할 함수
from sklearn.preprocessing import StandardScaler      # 평균 0, 표준편차 1로 표준화
import numpy as np                                     # 수치 계산 라이브러리

np.set_printoptions(suppress=True)  # 지수 표기(e.g. 1e-5) 생략하고 보기 쉽게 출력

# 📌 1. 데이터 불러오기
dataset = load_digits()     # 손글씨 이미지 (64차원 벡터), 10개 클래스 (0~9)
X = dataset.data            # 입력 특성 (8x8 이미지 → 64차원 벡터)
y = dataset.target          # 정답 레이블 (0~9)

# 📌 2. 훈련/테스트 데이터 분리 (클래스 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y: 훈련/테스트 세트에서 각 클래스 비율을 동일하게 유지

# 📌 3. 특성 값 표준화 (Standardization)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 평균 0, 분산 1로 변환 (훈련 기준)
X_test = scaler.transform(X_test)        # 동일한 기준으로 테스트 데이터도 변환

# 📌 4. 학습 설정
num_features = X_train.shape[1]  # 특성 개수 = 64 (8x8 이미지)
num_samples = X_train.shape[0]   # 훈련 샘플 수
num_classes = 10                 # 숫자 클래스 (0~9)

# 📌 5. 정답 레이블을 One-hot 인코딩
one_hot = np.eye(num_classes)        # 단위행렬 (10x10) → 예: 3 → [0 0 0 1 0 0 0 0 0 0]
y_train_one_hot = one_hot[y_train]   # 훈련 데이터용 one-hot 레이블
y_test_one_hot = one_hot[y_test]     # 테스트 데이터용 one-hot 레이블

# 📌 6. 가중치(w), 편향(b) 초기화 (정규분포 기반)
learning_rate = 0.01
epochs = 300
w = np.random.randn(num_features, num_classes)  # (64, 10): 각 특성별 클래스 가중치
b = np.random.randn(num_classes)                # (10,): 각 클래스별 편향

# 📌 7. 학습 시작 (Softmax 회귀 반복 학습)
for epoch in range(epochs):

    # 7-1. 선형결합 (logits 계산): z = Xw + b
    logits = X_train @ w + b                   # shape: (샘플 수, 클래스 수)
    logits -= logits.max(axis=1, keepdims=True)  # 오버플로우 방지 (softmax 안정성 처리)

    # 7-2. Softmax 함수 적용 (확률로 변환)
    exp_logits = np.exp(logits)                # 지수 계산: e^z
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)  # 클래스별 합
    softmax = exp_logits / sum_exp             # 확률 분포로 정규화

    # 7-3. 오차(error) 계산: 예측값 - 실제값
    error = softmax - y_train_one_hot          # shape: (샘플 수, 클래스 수)

    # 7-4. 기울기 계산 (Gradient 계산)
    gradient_w = X_train.T @ error / num_samples  # 가중치 w에 대한 평균 기울기
    gradient_b = error.mean(axis=0)               # 편향 b에 대한 평균 기울기

    # 7-5. 파라미터 업데이트 (Gradient Descent)
    w -= learning_rate * gradient_w             # 가중치 업데이트
    b -= learning_rate * gradient_b             # 편향 업데이트

    # 7-6. 손실 함수 계산 (Cross Entropy Loss)
    loss = -np.sum(y_train_one_hot * np.log(softmax + 1e-15)) / num_samples
    # log(softmax) 값이 0이 되지 않도록 1e-15 더함 (log 0 방지)

    # 7-7. 중간 출력 (100 epoch마다 손실 출력)
    if epoch % 100 == 0:
        print(f'[Epoch {epoch}] Loss: {loss:.4f}')

# 📌 8. 테스트 데이터 예측
test_logits = X_test @ w + b
test_logits -= test_logits.max(axis=1, keepdims=True)  # 오버플로우 방지
exp_test_logits = np.exp(test_logits)
sum_exp_test = np.sum(exp_test_logits, axis=1, keepdims=True)
softmax_test = exp_test_logits / sum_exp_test          # softmax 결과 (확률 분포)

# 📌 9. 테스트 손실 계산
test_loss = -np.sum(y_test_one_hot * np.log(softmax_test + 1e-15)) / y_test_one_hot.shape[0]
print(f'\n[Test Loss] {test_loss:.4f}')

# 📌 10. 테스트 정확도 계산
y_pred = np.argmax(softmax_test, axis=1)       # 가장 높은 확률을 가진 클래스 선택
accuracy = np.mean(y_pred == y_test)           # 정답과 일치한 비율
print(f'[Test Accuracy] {accuracy:.4f} ({accuracy * 100:.2f}%)')
