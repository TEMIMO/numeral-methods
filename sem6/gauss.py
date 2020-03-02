import math
import numpy
from numpy.polynomial import Polynomial as P
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt

# Вариант Ньютона-Котса

a = float(input("Введите левую границу:"))
b = float(input("Введите правую границу:"))


def f(x):
    return 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x
    #return 2 * math.cos(3.5 * x) * math.exp(5 * x / 3) + 3 * math.sin(1.5 * x) * math.exp(-4 * x) + 3

def f1(x):
    return (2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x) / ((x - 0.1) ** 0.2)
    #return (2 * math.cos(3.5 * x) * math.exp(5 * x / 3) + 3 * math.sin(1.5 * x) * math.exp(-4 * x) + 3) / ((x - 1.5) ** 0.2)

# Точное значение:
exv = integrate.quad(f1, a, b)[0]
cin = []
cout = []
scout = []
for i in range(3, 15):
    sum = 0
    n = i
    moments = []
    cin.append(i)
    finishmoments = []
    for j in range(0, 2*n):
        moments = numpy.append(moments, (2.2 ** (0.8 + j)) / (j + 0.8))
    for j in range(0, n):
        finishmoments = numpy.append(finishmoments, (2.2 ** (0.8 + j)) / (j + 0.8))

    matrix = [[0 for j in range(n)] for l in range(n)]
    matrix1 = [[0 for j in range(n)] for l in range(n)]
    goodmoments = []
    for j in range(0, n):
        goodmoments = numpy.append(goodmoments, (-1)*moments[n+j])
    for j in range(0, n):
        for l in range(0, n):
            matrix[j][l] = moments[j+l]
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
        sum += akoef[j]*f(roots[j])
        ss += abs(akoef[j])
    if i >= 15:
        cout.append(0)
    else:
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
