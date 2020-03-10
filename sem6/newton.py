import math
import numpy
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt

a = 0
b = 2.2
alpha = 0.2
betta = 0


def f11(x):
    y = 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x
    return y


def f22(x):
    if x-a == 0:
        y = 1
    else:
        y = ((x - a) ** (-alpha)) * ((b - x) ** (-betta))
    return y


def f(x):
    y = f1(x)
    return y


def f1(x):
    y = f11(x) * f22(x)
    return y

# Точное значение:
exv = integrate.quad(f1, a, b)[0]
cin = []
out = []
outcheb = []
scout = []
scoutcheb = []
for l in range(3, 25):
    cin.append(l)
    n = l
    x = [0 for i in range(n)]
    cheb = [0 for i in range(n)]
    delta = (b - a) / (n - 1)
    for i in range(0, n):
        x[i] = a + i * delta
        cheb[i] = 1/2*((b-a)*math.cos((2*i+1)/(2*(n+1)*math.pi))+(b+a))

    moments = [0 for i in range(0, n)]
    for i in range(0, n):
        moments[i] = (delta ** (0.8 + i)) / (i + 0.8)
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

