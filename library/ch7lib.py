import math
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

def guassian_elim_pp(A, b):
    n = A.shape[0]
    for i in range(n):
        am = abs(A[i,i])
        p = i
        for j in range(i+1, n):
            if abs(A[j,i]) > am:
                am = abs(A[j,i])
                p = j
        if p > i:
            for k in range(i, n):
                hold = A[i,k]
                A[i,k] = A[p,k]
                A[p,k] = hold
            hold = b[i]
            b[i] = b[p]
            b[p] = hold
        for j in range(i+1, n):
            m = A[j,i]/A[i,i]
            for k in range(i+1, n):
                A[j,k] = A[j,k] - m*A[i,k]
            b[j] = b[j] - m*b[i]

    return A, b


def __Hn(i, j):
    return (1.0/(i + j + 1.0))


def __ConeTwiddle(i, j):
    return (abs((-1)**i + (-1)**j)/2.0)


# Hn Matrix creator function:
# Parameters:
#    n- size of the (square) matrix
# return: Hn-style (numpy) matrix
def hn_matrix(n):
    retmat = custom_matrix(n, __Hn)
    # Initialize matrix size nxn:
    return retmat


def main():
    print(custom_matrix(5, __Hn))
    print(hn_matrix(5))
    print(custom_matrix(5, __ConeTwiddle))

if (__name__ == "__main__"):
    main()

