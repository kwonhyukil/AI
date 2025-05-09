from sklearn.datasets import fetch_california_housing

# 1. 데이터 로드
data = fetch_california_housing()

# 2. 주요 속성 확인
X = data.data   # 입력 데이터 (numpy.ndarray)
Y = data.target # 타겟값 (중간 집값, 단위: 100,000$)
feature_name = data.feature_names # 특성 이름 리스트

print(type(X),type(Y),type(feature_name))

print("입력 X shape: ", X.shape) # 샘플 수: 26,000  특성(featrue): 8개
print("타켓 Y shape: ", Y.shape) # 타켓 수: 26,000
print("특성 이름: ", feature_name) # ['MedInc','HouseAge','AveRooms', .....]
print("설명: ", data.DESCR[:1000]) # 데이터셋 설명 일부 출력