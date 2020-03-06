import math
import numpy
from numpy.polynomial import Polynomial as P
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt

a = 0
b = 2.2
alpha = 0.2
betta = 0


def f11(x):
    return 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x


def f22(x):
    return ((x - a) ** (-alpha)) * ((b - x) ** (-betta))


def f(x):
    y = f11(x)
    return y


def f1(x):
    y = f11(x) * f22(x)
    return y


# Точное значение:
exv = (integrate.quad(f1, a, b)[0])
cin = []
cout = []
scout = []
for i in range(3, 15):
    sum = 0
    n = i
    cin.append(i)
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
    ss = 0
    for j in range(0, n):
        sum += akoef[j] * f(roots[j])
        ss += abs(akoef[j])

    cout.append(-math.log10(abs(sum - exv)))
    scout.append(ss)
plt.subplot(221)
plt.plot(cin, cout)
plt.grid(True)

plt.subplot(222)
plt.plot(cin, scout)
plt.grid(True)

plt.show()
# Точное значение интеграла
print("Точное значение: ", exv)

