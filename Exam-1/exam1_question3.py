import numpy 
import numpy as np

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

def __Kn(i, j):
    if i == j:
        return 2
    elif abs(i-j) == 1:
        return -1
    else:
        return 0


def __Tn(i, j):
    if i == j:
        return 4
    elif abs(i-j) == 1:
        return 1
    else:
        return 0


def lu_pp(A_):
    A = A_.copy()
    n = A_.shape[0]
    indx = list(range(n))
    for i in range(n-1):
        #print(f"Iteration {i}:")
        #print(A)
        am = abs(A[i, i])
        p = i
        for j in range(i+1, n):
            if abs(A[j, i]) > am:
                am = abs(A[j, i])
                p = j
        if p > i:
            for k in range(n):
                hold = A[i, k].copy()
                A[i, k] = A[p, k].copy()
                A[p, k] = hold.copy()
            ihold = indx[i]
            indx[i] = indx[p]
            indx[p] = ihold
        for j in range(i+1, n):
            A[j, i] = A[j, i]/A[i, i]
            for k in range(i+1, n):
                A[j, k] = A[j, k] - A[j, i]*A[i, k]
    return [A, indx]

def lu_solve_pp(A_, b_, i):
    b = b_.copy()
    A = A_.copy()
    n = A_.shape[0]
    x = numpy.zeros((n, 1))
    for k in range(n):
        x[k] = b[i[k]]
    for k in range(n):
        b[k] = x[k]
    y = [b[0]]
    for i in range(1, n):
        s = 0.0
        for j in range(i):
            s = s + A[i, j] * y[j]
        y.append(b[i] - s)
    x[n-1] = y[n-1]/A[n-1, n-1]
    for i in range(n-2, -1, -1):
        s = 0.0
        for j in range(i+1, n):
            s = s + A[i, j] * x[j]
        x[i] = (y[i] - s)/A[i, i]
    return x


ex_3_A = matrix(10, 'K')@matrix(10, 'T')
ex_3_b = np.array([-4, -1, 0, 0, 0, 0, 0, 0, 10, 40])
numpy_sol = np.linalg.solve(ex_3_A, ex_3_b)
ex_3_A, indx = lu_pp(ex_3_A)
ex_3_x = lu_solve_pp(ex_3_A, ex_3_b, indx)

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(f"Solution X:\n {ex_3_x}")
print(f"Solution X(numpy):\n {numpy_sol.T}")
print(f"Matrix LU(Not Split):\n {ex_3_A}")