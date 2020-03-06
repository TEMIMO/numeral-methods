import math
import numpy
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

alpha = 0.2
betta = 0

abig = 0
bbig = 2.2


def f11(x):
    return 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x


def f22(x):
    return ((x - abig) ** (-alpha)) * ((bbig - x) ** (-betta))


def f(x):
    y = 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x
    return y


def f1(x):
    y = f11(x) * f22(x)
    return y


def gauss(n, a, b):
    sum = 0
    moments = [0 for i in range(0, 2 * n)]
    finishmoments = [0 for i in range(0, n)]
    for j in range(0, 2 * n):
        moments[j] = (((b ** (0.8 + j)) / (j + 0.8)) - ((a ** (0.8 + j)) / (j + 0.8)))
    for j in range(0, n):
        finishmoments[j] = (((b ** (0.8 + j)) / (j + 0.8)) - ((a ** (0.8 + j)) / (j + 0.8)))

    matrix = [[0 for j in range(n)] for l in range(n)]
    matrix1 = [[0 for j in range(n)] for l in range(n)]
    goodmoments = []
    for j in range(0, n):
        goodmoments = numpy.append(goodmoments, (-1) * moments[n + j])
    for j in range(0, n):
        for l in range(0, n):
            matrix[j][l] = moments[j + l]
    solution = numpy.linalg.solve(matrix, goodmoments)
    sol = []
    sol = numpy.append(sol, solution)
    sol = numpy.append(sol, 1.0)
    p = P(sol)
    roots = p.roots()
    matrix1 = [[0 for j in range(n)] for l in range(n)]
    for j in range(0, n):
        for l in range(0, n):
            matrix1[j][l] = roots[l] ** j
    akoef = numpy.linalg.solve(matrix1, finishmoments)
    for j in range(0, n):
        sum += akoef[j] * f(roots[j])
    return sum


exv = integrate.quad(f1, abig, bbig)[0]
summ = 0.0
summ += gauss(int(3), 0, 1.5)
summ += gauss(int(6), 1.5, 2.0)
summ += gauss(int(6), 2.0, 2.2)
print(abs(exv - summ))

