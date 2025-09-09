import numpy as np
from numpy.linalg import inv

# Create a 3x2 matrix X:
X = np.array([[1, 2], [2, 4], [1, -1]])

# Create a vector y representing the target values
y = np.array([0, 0.1, 1])

# Compute the least squares solution for w
w = inv(X.T @ X) @ X.T @ y
print(w)


import numpy as np
import matplotlib.pyplot as plt

# Define the range for w1
w1 = np.linspace(-2, 2, 400)

# Define the equations of the lines
w2_1 = -w1 / 2  # From w1 + 2 * w2 = 0, w2 = -w1 /2
w2_2 = (0.1 - 2*w1) / 4  # From 2*w1 + 4 * w2 = 0.1, w2 = (0.1 - 2*w1) / 4
w2_3 = -(1-w1) # From w1 - w2 = 1, w2 = - (1 - w1)

# Plot the lines
plt.plot(w1, w2_1, 'r-', label='Equation (1)')
plt.plot(w1, w2_2, 'b--', label='Equation (2)')
plt.plot(w1, w2_3, 'k:', label='Equation (3)')

# Add labels and title
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Plot of the lines')

# Add a legend
plt.legend()

# Show the plot
plt.grid(True)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.show()