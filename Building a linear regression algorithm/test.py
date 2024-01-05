import numpy as np

x = np.array([[1, 2, 3],
               [4, 5, 6]])

y = np.array([[3, 5, 7],
               [9, 11, 13]])

print(np.sum(np.square(y - x)) / y.shape[0])