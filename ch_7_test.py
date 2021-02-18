import library.ch7lib as ch7
import numpy as np

n = 15
A = ch7.matrix(n, 'K')
b = np.random.uniform(0, 10, (n, 1))

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# print(ch7.gaussian_elim_pp(A, b))
# print(np.linalg.solve(A, b))

A = ch7.matrix(3, 'T')
b = np.array([6, 12, 14])

L = ch7.factorization(A)
print(L)
x = ch7.lu_solver(L, b)
print(x)

pivot_A, idx = ch7.lu_pp(A)
pivot_x = ch7.lu_solve_pp(pivot_A, b, idx)
print(pivot_x)
