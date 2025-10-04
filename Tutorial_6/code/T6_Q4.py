import numpy as np 
from numpy.linalg import inv 
from sklearn.preprocessing import PolynomialFeatures
X = np.array([[1,-1], [1,0], [1,0.5], [1,0.3], [1,0.8]]) 
y = np.array([1, 1, -1, 1, -1]) 

## Linear regression for classification 
w = inv(X.T @ X) @ X.T @ y 
print(f"w = {w}") 

Xt = np.array([[1,-0.1], [1,0.4]]) 
y_predict = Xt @ w 
print(f"y_predict = {y_predict}") 

y_class_predict = np.sign(y_predict) 
print(f"y_class_predict = {y_class_predict}")