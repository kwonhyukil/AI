import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, stratify=y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

num_features = X_train.shape[1]
epochs = 100000
learning_rate = 0.01

weights = np.random.randn(num_features, 1)
bias = np.random.randn()

y_train = y_train.reshape(-1, 1)

for epoch in range(epochs):
  z = X_train @ weights + bias
  predictions = 1 / (1 + np.exp(-z))
  errors = predictions - Y_train
  grad_weights = X_train.T @ errors / len(X_train)
  grad_bias = np.mean(errors)
  weights -= learning_rate * grad_weights
  bias -= learning_rate * grad_bias

  loss = -np.mean(
      y_train * np.log(predictions + 1e-15) +
      (1 - Y_train) * np.log(1 - predictions + 1e-15)
  )
  if epoch % 1000 == 0:
      print(f"Epoch {epoch}, Loss: {loss:.4f}")
z_test = X_test @ weights + bias
y_prob_test = 1 / (1 + np.exp(-z_test))
y_pred_test = (y_prob_test >= 0.5).astype(int)
test_accuracy = np.mean(y_pred_test.reshape(-1) == y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")


bar = np.array([val for val in range(1, 21)])

print(bar.reshape(-1, 10))