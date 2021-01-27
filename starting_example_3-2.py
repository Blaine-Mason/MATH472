import math


def f(x):
    return math.exp(x) + x


def df(x):
    return math.exp(x) + 1

#Fix this
def newtons_method(iterations, x_0):
    x = [0]*iterations
    for i in range(iterations):
        if i == 0:
            print(f'For initial guess: {x_0}')
        else:
            x[i] = x[i-1] - f(x[i-1]) / df(x[i-1])
            print(f'For guess #{i}: {x[i + 1]}')
            x[i + 1] = x[i] - f(x[i]) / df(x[i])


if __name__ == '__main__':
    newtons_method(11, 0)
