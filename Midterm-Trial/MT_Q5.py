import numpy as np
from numpy.linalg import inv

X = np.array([[1, 1, 2], [1, 0, 6], [1, 1, 0], [1, 0, 5], [1, 1, 7]])
y = np.array([1, 2, 3, 4, 5])

w = inv(X.T @ X) @ X.T @ y

# (1)
y_train = X @ w
MSE_train = np.mean(np.power((y - y_train), 2))
print(f"MSE_train: {MSE_train}")

# (2)
X_test = np.array([1, 1, 3])
y_test = X_test @ w
print(f"y_test: {y_test}")
