import math
import numpy
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt

# Вариант Ньютона-Котса

a = float(input("Введите левую границу:"))
b = float(input("Введите правую границу:"))


def f(x):
    #return 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x
    return 2 * math.cos(3.5 * x) * math.exp(5 * x / 3) + 3 * math.sin(1.5 * x) * math.exp(-4 * x) + 3

def f1(x):
    #return (2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x) / ((x - 0.1) ** 0.2)
    return (2 * math.cos(3.5 * x) * math.exp(5 * x / 3) + 3 * math.sin(1.5 * x) * math.exp(-4 * x) + 3) / ((x - 1.5) ** 0.2)

# Точное значение:
exv = integrate.quad(f1, a, b)[0]
cin = []
out = []
outcheb = []
scout = []
scoutcheb = []
for l in range(3, 20):
    ##############
    cin.append(l)
    n = l
    x = []
    cheb = []
    delta = (b - a) / (n - 1)
    for i in range(0, n):
        x = numpy.append(x, a + i * delta)
        cheb = numpy.append(cheb, 1/2*((b-a)*math.cos((2*i+1)/(2*(n+1)*math.pi))+(b+a)))
    moments = []

    for i in range(0, n):
        moments = numpy.append(moments, (2.2 ** (0.8 + i)) / (i + 0.8))
    matrix = [[0 for i in range(n)] for j in range(n)]
    matrixcheb = [[0 for i in range(n)] for j in range(n)]
    for i in range(0, n):
        for j in range(0, n):
            matrix[i][j] = (x[j] ** i)
            matrixcheb[i][j] = (cheb[j] ** i)

    solution = numpy.linalg.solve(matrix, moments)
    solutioncheb = numpy.linalg.solve(matrixcheb, moments)
    s = 0
    scheb = 0
    ss = 0
    sscheb = 0
    for i in range(0, n):
        s += solution[i] * f(float(x[i]))
        scheb += solutioncheb[i] * f(float(cheb[i]))
        ss += abs(solution[i])
        sscheb += abs(solutioncheb[i])
    print(s)
    print(ss)
    out.append(-math.log10(abs(s - exv)))
    outcheb.append(-math.log10(abs(scheb - exv)))
    scout.append(ss)
    scoutcheb.append(sscheb)
    #############


plt.subplot(221)
plt.plot(cin, out)
plt.grid(True)

plt.subplot(222)
plt.plot(cin, scout)
plt.grid(True)

plt.subplot(223)
plt.plot(cin, outcheb)
plt.grid(True)

plt.subplot(224)
plt.plot(cin, scoutcheb)
plt.grid(True)

plt.show()

# Точное значение интеграла
print("Точное значение: ", exv)

