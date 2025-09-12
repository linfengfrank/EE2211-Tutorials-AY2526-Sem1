import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

# Without bias, this is an even-determined system and X is invertible.
X   = np.array([[1, 0, 1], [2, -1, 1], [1, 1, 5]])
y = np.array([[1], [2], [3]])

b   = np.ones( (len(X),1) )
X_b = np.hstack((b, X)) # X matrix with bias

#(a) Perform a linear regression with addition of a bias/offset term
w = inv(X)@y
print(f"w={w}\n")

X_t = np.array([[-1, 2, 8], [1, 5,-1]])
y_t = X_t@w
print(f"y_t={y_t}\n")

#(b) After adding bias, it becomes an under-determined system.
w_b = X_b.T@inv(X_b@X_b.T)@y
print(f"w_b={w_b}\n")



# Generate a grid for 3D plane visualization
x1_vals = np.linspace(-2, 3, 10)
x2_vals = np.linspace(-2, 3, 10)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute the corresponding X3 for each model
# For w (without bias): Solve X3 in terms of X1, X2
X3_w = (-w[0, 0] * X1 - w[1, 0] * X2) / w[2, 0]

# For w_b (with bias): Solve X3 in terms of X1, X2
X3_wb = (-w_b[1, 0] * X1 - w_b[2, 0] * X2 - w_b[0, 0]) / w_b[3, 0]

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot original data points
ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='blue', label="Training Points (X, y)", s=50)

# Plot test points
ax.scatter(X_t[:, 0], X_t[:, 1], X_t[:, 2], color='red', marker='^', label="Test Points (X_t, y_t)", s=50)

# Plot the regression plane for w (without bias)
ax.plot_surface(X1, X2, X3_w, color='green', alpha=0.5, edgecolor='k', label="Regression Plane w (No Bias)")

# Plot the regression plane for w_b (with bias)
ax.plot_surface(X1, X2, X3_wb, color='purple', alpha=0.5, edgecolor='k', label="Regression Plane w_b (With Bias)")

# Labels
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("X3")
ax.set_title("3D Visualization of Regression Planes")
ax.legend()
ax.grid()

plt.show()