
samples = []
y = []

w = [0.2, 0.3]
b = 0.1

gradient_w = [0.0, 0.0]
gradient_b = 0.0

# 모든 샘플 순회 : 1 epoch
for dp, y_ in zip(samples, y):
    # 예측값
    predict_y = w[0] * dp[0] + w[1] * dp[1] + b

    # 오차 : 예측 값 - 정답 값
    error = predict_y - y_

    # 기울기 값 누적
    gradient_w[0] += dp[0] * error
    gradient_w[1] += dp[1] * error
    gradient_b += error

# update gradient of each w
w[0] = w[0] - gradient_w[0] / len[samples]
w[1] = w[1] = gradient_w[1] / len[samples]

# update gradient of b
b = b - gradient_b / len[samples]

for f, y_ in samples:

    # 예측 값
    predict_y = w * f + b
    
    # Error = 예측 값 - 정답 값
    error = predict_y - y_ 
    
    # W의 기울기 : sum(error * each f)/ 샘플의 갯수
    gradient_w += error * f

    # b의 기울기
    gradient_b += error 


w = w - gradient_w / len(samples)
b = b - gradient_b / len(samples)