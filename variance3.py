from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(suppress=True, precision=30)

# 데이터 준비
values1 = np.array((160, 170, 190, 180)).reshape(-1, 1)  # 키 (cm)
values2 = np.array((400000000, 700000000, 200000000, 30000000)).reshape(-1, 1)  # 소득 (원)

# 각각에 대해 스케일러 생성
scaler1 = StandardScaler()
scaler2 = StandardScaler()

# fit: 평균, 분산, 표준편차 계산
fit_values1 = scaler1.fit_transform(values1)
fit_values2 = scaler2.fit_transform(values2)

print(fit_values1, fit_values2)

# # 출력
# print("🔹 values1 (키)")
# print("평균:", fit_values1.mean_)
# print("분산:", fit_values1.var_)
# print("표준편차:", fit_values1.scale_)

# print("\n🔹 values2 (소득)")
# print("평균:", fit_values2.mean_)
# print("분산:", fit_values2.var_)
# print("표준편차:", fit_values2.scale_)





# values = np.arange(10).reshape(-1, 1)  # [[0], [1], ..., [9]]

# fit : 현재 데이터 셋의 평균, 분산, 표준편차를 계산하는 역할
# fit_value = scaler.fit(values)



# print("평균:", fit_value.mean_)       # array([4.5])
# print("분산:", fit_value.var_)        # array([8.25])
# print("표준편차:", fit_value.scale_)  # array([2.87228132])
