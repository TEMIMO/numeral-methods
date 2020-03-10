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
    if x-abig == 0:
        y = 1
    else:
        y = ((x - abig) ** (-alpha)) * ((bbig - x) ** (-betta))
    return y


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


def newton(n, a, b):
    x = [0 for i in range(n)]
    delta = (b - a) / (n - 1)
    for i in range(0, n):
        x[i] = a + i * delta
    moments = [0 for i in range(0, n)]
    for i in range(0, n):
        moments[i] = (((b ** (0.8 + i)) / (i + 0.8)) - ((a ** (0.8 + i)) / (i + 0.8)))
    matrix = [[0 for i in range(n)] for j in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            matrix[i][j] = (x[j] ** i)
    solution = numpy.linalg.solve(matrix, moments)
    s = 0
    for i in range(0, n):
        s += solution[i] * f(float(x[i]))
    return s

k = int(input())
delta = bbig/k
exv = integrate.quad(f1, abig, bbig)[0]
sumg = 0.0
sumn = 0.0
for i in range(k):
    sumg += gauss(2, abig+i*delta, abig+(i+1)*delta)
    sumn += newton(3, abig + i * delta, abig + (i + 1) * delta)
print('Разница метода Гаусса: ', abs(exv - sumg))
print('Разница метода Ньютона: ', abs(exv - sumn))

