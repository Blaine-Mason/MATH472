import numpy 
import numpy as np
import matplotlib.pyplot as plt

def __Hn(i, j):
    return (1.0/(i + j + 1.0))


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

def __An(i, j):
    if i == j:
        return 1
    elif i - j == 1:
        return 4
    elif i - j == -1:
        return -4
    else:
        return 0

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


def cond_estimation(A):
    alpha = numpy.linalg.norm(A, ord=numpy.inf)

    A, idx = lu_pp(A)

    _y = numpy.random.uniform(0, 1, (A.shape[0], 1))
    for i in range(5):
        _y = _y/numpy.linalg.norm(_y, numpy.inf)
        _y = lu_solve_pp(A, _y, idx)
    _v = numpy.linalg.norm(_y, numpy.inf)
    return numpy.dot(_v, alpha)


x = np.arange(4, 21, 1)
t_ = np.array([])
k_ = np.array([])
h_ = np.array([])
a_ = np.array([])
s_ = np.array([])
for i in range(4, 21):
    t_ = np.append(t_, cond_estimation(matrix(i, 'T')))
    k_ = np.append(k_, cond_estimation(matrix(i, 'K')))
    h_ = np.append(h_, cond_estimation(matrix(i, 'H')))
    a_ = np.append(a_, cond_estimation(matrix(i, 'A')))
    s_ = np.append(s_, cond_estimation(matrix(i, 'S')))
    
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
print(f"My Computation for Tn: {t_}")
print(f"My Computation for Kn: {k_}")
print(f"My Computation for Hn: {h_}")
print(f"My Computation for An: {a_}")
print(f"My Computation for Sn: {s_}")

 
plt.subplot(3,2,1).set_title("Tn")
plt.plot(x, t_)
plt.subplot(3,2,2).set_title("Kn")
plt.plot(x, k_, label="Kn")
plt.subplot(3,2,3).set_title("Hn")
plt.plot(x, h_, label="Hn")
plt.subplot(3,2,4).set_title("An")
plt.plot(x, a_, label="An")
plt.subplot(3,2,5).set_title("Sn")
plt.plot(x, s_, label="Sn")
plt.legend()
plt.rcParams['figure.figsize'] = [5, 5]

plt.show()
