import numpy as np

# kin = np.ones((2,3,4))
# bar = np.zeros((5,2))
# # foo = np.array([[1],[2],[3]])

# print(f"{kin.shape},{bar.shape}")
# print(f"{kin},{bar}")
# # print(f"{type(kin)},{type(bar),{type(foo)}}")

num_feature = 3
num_samples = 2

np.random.seed(1)
np.set_printoptions(suppress=True, precision=3)

x = np.random.rand(num_samples, num_feature) * 10 

# h(x) = wx1 + wx2 + wx3 + b
w_true = np.random.randint(1, 10, num_feature)
b_true = np.random.randn() * 0.5

y = x[:, 0] * w_true[0] + x[:, 1] * w_true[1] + x[:, 2] * w_true[2] + b_true
y_ = x @ w_true + b_true



# Learning
w = np.random.rand(num_feature)
b = np.random.randn()
learning_rate = 0.01
epochs = 10000

print(w, b)


# prediction
prediction = x @ w + b
print(f"prediction: {prediction}")
print(f"x,w:{x @ w}")

# error
error = prediction - y
print(f"error: {error}")
print(f"x.t :\n{x.T}")
print(f"x.t * error\n{x.T @ error}")

# gradient
gradient = (x.T @ error) / num_samples
print(f"gradinet {gradient}")

w = w - learning_rate * gradient
b = b - learning_rate * error.mean()
print(f"w: {w}")
print(f"b: {b}")

print(f"x:\n{x}")
print(f"w_true: {w_true}")
print(f"b_true: {round(b_true, 3)}")
print(f"y: {y}")
print(f"y_: {y_}")