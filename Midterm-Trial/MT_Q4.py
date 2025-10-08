import numpy as np
from numpy.linalg import inv

X = np.array([[1, 3], [1, 4], [1, 5], [1, 6], [1, 7]])

y = np.array([5, 4, 3, 2, 1])

w = inv(X.T @ X) @ X.T @ y

y_pre = np.array([1, 8]) @ w

print(f"y_pre: {y_pre}")


