import library.ch7lib as ch7
import numpy as np

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
A = np.random.uniform(-5, 5, (50, 50))
b = np.random.uniform(-5, 5, (50,1))
A_, i = ch7.lu_pp(A)
x = ch7.lu_solve_pp(A_, b, i)
x_ = np.linalg.solve(A, b)
print(np.abs(np.sum(x) - np.sum(x_)))
