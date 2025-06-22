# 필요한 라이브러리 불러오기
from sklearn.datasets import load_digits  # 손글씨 숫자 데이터셋
from sklearn.model_selection import train_test_split  # 학습/테스트 분할
from sklearn.preprocessing import StandardScaler  # 특성 표준화 도구
import numpy as np  # 수치 연산 라이브러리

np.set_printoptions(suppress=True)  # 넘파이 출력 시 지수표기 생략

# 1. 데이터 로드
dataset = load_digits()  # 손글씨 숫자 (0~9) 이미지 데이터셋 로드
X = dataset.data         # X: 각 이미지(64차원 벡터)
y = dataset.target       # y: 정답(0~9)

# 2. 훈련/테스트 데이터셋 분리 (클래스 비율 유지 stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 특성 값 표준화 (평균 0, 분산 1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 훈련 데이터 기준으로 표준화
X_test = scaler.transform(X_test)        # 동일한 방식으로 테스트 데이터도 표준화

# 4. 학습 설정
num_features = X_train.shape[1]  # 특성 수 (64개 픽셀)
num_samples = X_train.shape[0]   # 샘플 수 (훈련 데이터 개수)
num_classes = 10                 # 클래스 수 (0~9)

# 5. 정답 데이터를 One-hot 인코딩 (정답 위치만 1, 나머지는 0)
one_hot = np.eye(num_classes)       # 단위 행렬로 [0,0,1,...] 형태 생성용
y_train_one_hot = one_hot[y_train]  # 훈련용 정답 one-hot
y_test_one_hot = one_hot[y_test]    # 테스트용 정답 one-hot

# 6. 가중치(w), 편향(b) 초기화 (정규분포 기반 무작위)
learning_rate = 0.01
epochs = 300
w = np.random.randn(num_features, num_classes)  # (64, 10) 크기의 가중치 행렬
b = np.random.randn(num_classes)                # (10,) 크기의 편향 벡터

# 7. 소프트맥스 회귀 학습 시작
for epoch in range(epochs):

    # 7-1. 선형결합: z = Xw + b
    logits = X_train @ w + b  # 각 샘플에 대해 클래스별 점수 계산
    logits -= logits.max(axis=1, keepdims=True)  # 오버플로우 방지 (softmax 안정화)

    # 7-2. Softmax 함수 적용: 확률값으로 변환
    exp_logits = np.exp(logits)  # e^z
    sum_exp = np.sum(exp_logits, axis=1, keepdims=True)  # 샘플별 전체 지수합
    softmax = exp_logits / sum_exp  # softmax 결과 (확률 분포)

    # 7-3. 예측값과 정답의 차이 (오차) 계산
    error = softmax - y_train_one_hot  # 예측 확률 - 실제 one-hot

    # 7-4. 기울기 계산 (경사하강법용)
    gradient_w = X_train.T @ error / num_samples  # 가중치에 대한 기울기
    gradient_b = error.mean(axis=0)               # 편향에 대한 기울기

    # 7-5. 파라미터 업데이트
    w -= learning_rate * gradient_w  # 가중치 업데이트
    b -= learning_rate * gradient_b  # 편향 업데이트

    # 7-6. 손실 함수(Cross Entropy Loss) 계산
    loss = -np.sum(y_train_one_hot * np.log(softmax + 1e-15)) / num_samples  # 로그 손실

    # 7-7. 중간 결과 출력 (100 epoch마다)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 8. 테스트 데이터로 예측 수행
test_logits = X_test @ w + b
test_logits -= test_logits.max(axis=1, keepdims=True)  # 오버플로우 방지
exp_test_logits = np.exp(test_logits)
sum_exp_test = np.sum(exp_test_logits, axis=1, keepdims=True)
softmax_test = exp_test_logits / sum_exp_test  # 테스트용 softmax 결과

# 9. 테스트 손실 계산
test_loss = -np.sum(y_test_one_hot * np.log(softmax_test + 1e-15)) / y_test_one_hot.shape[0]
print(f'Test Loss: {test_loss:.4f}')

# 10. 예측값과 정답 비교하여 정확도 계산
y_pred = np.argmax(softmax_test, axis=1)  # softmax 결과 중 확률 가장 높은 클래스 선택
accuracy = np.mean(y_pred == y_test)      # 정답과 예측이 일치한 비율
print(f'Accuracy: {accuracy:.4f}')        # 예: 0.9350
print(f"Accuracy: {accuracy:.4f}%")       # 예: 93.50%
