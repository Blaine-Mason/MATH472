import math


def f(x):
    return math.exp(x) + x


def df(x):
    return math.exp(x) + 1


# Fix this
def newtons_method(iterations, x_0):
    x = [x_0] * iterations
    for i in range(1, iterations):
        if i == 1:
            print(f'For initial guess: {x[i]}')
        else:
            x[i] = x[i - 1] - f(x[i - 1]) / df(x[i - 1])
            print(f'For guess #{i}: {x[i]}')


if __name__ == '__main__':
    newtons_method(11, 0)
