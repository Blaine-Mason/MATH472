import numpy


# Custom Matrix creator function:
# Parameters:
#    n- size of the (square) matrix
#    f- Entry function for matrix
# return: Hn-style (numpy) matrix
def custom_matrix(n, f):
    retmat = numpy.zeros((n, n), dtype=numpy.float64)
    for i in range(n):
        for j in range(n):
            retmat[i, j] = f(i, j)
    # Initialize matrix size nxn:
    return retmat


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


def __ConeTwiddle(i, j):
    return (abs((-1)**i + (-1)**j)/2.0)


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


def factorization(A_):
    """
    Perform Decomposition
    """
    n = A_.shape[0]
    A = A_.copy()
    for i in range(n):
        for j in range(i+1, n):
            A[j, i] = A[j, i]/A[i, i]
            for k in range(i+1, n):
                A[j, k] = A[j, k] - A[j, i]*A[i, k]
    return A


def lu_solver(A_, b_):
    """
    Solve Ly = b
    """
    n = A_.shape[0]
    b = b_.copy()
    x = numpy.zeros((n, 1))
    y = [b[0]]
    for i in range(1, n):
        sum_ = 0.0
        for j in range(i):
            sum_ = sum_ + A_[i, j]*y[j]
        y.append(b[i] - sum_)
    """
    Solve Ux = y
    """
    x[n-1] = y[n-1]/A_[n-1, n-1]
    for i in range(n-2, -1, -1):
        sum_ = 0.0
        for j in range(i+1, n):
            sum_ = sum_ + A_[i, j]*x[j]
        x[i] = (y[i] - sum_)/A_[i, i]
    return x


def lu_pp(A_):
    A = A_.copy()
    n = A_.shape[0]
    indx = list(range(n))
    for i in range(n-1):
        print(f"Iteration {i}:")
        print(A)
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


def cond_estimation(A, given):
    alpha = numpy.linalg.norm(A, ord=numpy.inf)

    A, idx = lu_pp(A)

    if given:
        _y = numpy.array([[.219, .0470, .6789]])

    _y = numpy.random.uniform(0, 1, (A.shape[0], 1))
    for i in range(5):
        _y = _y/numpy.linalg.norm(_y, numpy.inf)
        _y = lu_solve_pp(A, _y, idx)
    _v = numpy.linalg.norm(_y, numpy.inf)
    return numpy.dot(_v, alpha)


def growth_factor(A):
    A_ = A.copy()
    largest = 0
    n = A_.shape[0]
    for i in range(n-1):
        for j in range(i+1, n):
            m = A_[j, i]/A_[i, i]
            for k in range(n):
                A_[j, k] = A_[j, k] - m*A_[i, k]
        largest = max(numpy.max(numpy.abs(A_)), largest)
    return largest/numpy.linalg.norm(A_, numpy.inf)


# Hn Matrix creator function:
# Parameters:
#    n- size of the (square) matrix
# return: Hn-style (numpy) matrix
def matrix(n, version):
    if version == 'H':
        return custom_matrix(n, __Hn)
    elif version == 'K':
        return custom_matrix(n, __Kn)
    elif version == 'T':
        return custom_matrix(n, __Tn)
    elif version == 'A':
        return custom_matrix(n, __An)
    else:
        return custom_matrix(n, __ConeTwiddle)
