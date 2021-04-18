import numpy 
import numpy as np


def __Sn(i, j):
    i = i + 1
    j = j + 1
    if i == j or abs(i - j) == 1:
        return 1/(i + j - 1)
    else:
        return 0

def custom_matrix(n, f):
    retmat = numpy.zeros((n, n), dtype=numpy.float64)
    for i in range(n):
        for j in range(n):
            retmat[i, j] = f(i, j)
    # Initialize matrix size nxn:
    return retmat


def matrix(n, version):
    if version == 'H':
        return custom_matrix(n, __Hn)
    elif version == 'K':
        return custom_matrix(n, __Kn)
    elif version == 'T':
        return custom_matrix(n, __Tn)
    elif version == 'A':
        return custom_matrix(n, __An)
    elif version == 'S':
        return custom_matrix(n, __Sn)
    else:
        return custom_matrix(n, __ConeTwiddle)


def gaussian_elim_pp(A_, b_):
    A = numpy.copy(A_)
    b = numpy.copy(b_)
    n = A.shape[0]
    for i in range(n-1):
        am = abs(A[i, i])
        p = i
        for j in range(i+1, n):
            if abs(A[j, i]) > am:
                am = abs(A[j, i])
                p = j
        if p > i:
            for k in range(i, n):
                hold = A[i, k]
                A[i, k] = A[p, k]
                A[p, k] = hold
            hold = b[i].copy()
            b[i] = b[p].copy()
            b[p] = hold.copy()
        for j in range(i+1, n):
            m = A[j, i]/A[i, i]
            for k in range(i+1, n):
                A[j, k] = A[j, k] - m*A[i, k]
            b[j] = b[j] - m*b[i]
    n = A.shape[0]
    x = numpy.zeros(n)
    x[n - 1] = b[n - 1]/A[n - 1, n - 1]
    for i in range(n-1, -1, -1):
        sum_ = 0
        for j in range(i+1, n):
            sum_ = sum_ + A[i, j]*x[j]
        x[i] = (b[i] - sum_)/A[i, i]
    return A, b, x

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
A = matrix(10, 'S')
b = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
A_, b_, x = gaussian_elim_pp(A, b)
print(f"Solution vector:{x}")