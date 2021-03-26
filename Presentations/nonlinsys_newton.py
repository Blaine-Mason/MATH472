# Justin Ventura MATh472

"""
This script is where I will be working on creating newton's method
for non-linear systems.

I will begin with example one from chapter 7.8
"""

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

def f1(x, y):
    return (2 * x) - y + ((1/9) * np.exp(-x)) + 1


def f2(x, y):
    return -x + (2 * y) + ((1/9) * np.exp(-y)) - 1


def f(x, y):
    return np.array([f1(x, y), f2(x, y)])


def df1x(x, y):
    return 2 - ((1/9) * np.exp(-x))


def df1y(x, y):
    return -1


def df2x(x, y):
    return -1


def df2y(x, y):
    return 2 - ((1/9) * np.exp(-y))


def A_f(x, y):
    return np.array([[df1x(x, y), df1y(x, y)],
                     [df2x(x, y), df2y(x, y)]], dtype=np.float64)


if __name__ == '__main__':
    num = 10
    print(f'Newtons Method {num} iterations:')
    convergence_newtons = list()
    convergence_secant = list()
    convergence_secant_updates = list()

    x_0 = np.array([1, 1], dtype=np.float64)
    x_n = x_0 - ((la.inv(A_f(*x_0))) @ f(*x_0))
    for i in range(num):
        temp = x_n
        x_n = x_n - ((la.inv(A_f(*x_n))) @ f(*x_n))
        convergence_newtons.append(max(abs(x_n - temp)))
        print(x_n)

    input('Press enter to move on to Secant/Chord method.')

    x_0 = np.array([1, 1], dtype=np.float64)
    x_1 = np.array([-1, -1], dtype=np.float64)
    fixed = (la.inv(A_f(*x_0)))
    x_n = x_1 - (fixed @ f(*x_1))
    for i in range(num):
        temp = x_n
        x_n = x_n - (fixed @ f(*x_n))
        convergence_secant.append(max(abs(x_n - temp)))
        print(x_n)

    # TODO: update 'fixed' every 3 iterations
    input('Press enter to move on to Secant/Chord method (with updates).')

    x_0 = np.array([1, 1], dtype=np.float64)
    x_1 = np.array([-1, -1], dtype=np.float64)
    fixed = (la.inv(A_f(*x_0)))
    x_n = x_1 - (fixed @ f(*x_1))
    for i in range(num):
        temp = x_n
        if(i % 3 == 0 and i != 0):
            fixed = (la.inv(A_f(*x_n)))
        x_n = x_n - (fixed @ f(*x_n))
        convergence_secant_updates.append(max(abs(x_n - temp)))
        print(x_n)
    print(convergence_newtons)
    print(convergence_secant)
    print(convergence_secant_updates)
    iters = [i for i in range(num)]
    plt.plot(iters, convergence_newtons, label="Newtons")
    plt.plot(iters, convergence_secant, label="Secant")
    plt.plot(iters, convergence_secant_updates, label="Secant Updates")
    plt.legend()
    plt.show()
