import library.ch7lib as ch7
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
A = np.array([[4, 1, 0, 0],
              [1, 5, 1, 0],
              [0, 1, 6, 1],
              [1, 0, 1, 4]])
b = np.array([1, 7, 16, 14])
x = np.array([0, 0, 0, 0])
print(ch7.gauss_seidel(A, b))
