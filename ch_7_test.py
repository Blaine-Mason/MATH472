import library.ch7lib as ch7
import numpy as np

n = 15
A = ch7.matrix(n, 'K')
b = np.random.uniform(0, 10, (n, 1))

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

# print(ch7.gaussian_elim_pp(A, b))
# print(np.linalg.solve(A, b))

A = ch7.matrix(5, 'T')
b = np.array([1, 6, 12, 18, 19])
print(A)
print(b)
print(np.linalg.solve(A, b))

print(ch7.factorization(A))

pivot_A, idx = ch7.lu_pp(A)
print(pivot_A)
print(b)
pivot_x = ch7.lu_solve_pp(pivot_A, b, idx)
print(pivot_x)
