import math
import numpy
from pylab import *
from scipy import integrate
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P

# Степени
alpha = 0.2
betta = 0

# Граница области определения
abig = 0
bbig = 2.2


def f11(x):
    return 2.5 * math.cos(2 * x) * math.exp(2 * x / 3) + 4 * math.sin(3.5 * x) * math.exp(-3 * x) + 3 * x


def f22(x):
    if x - abig == 0:
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


exv = integrate.quad(f1, abig, bbig)[0]
print('Точное значение интеграла: ', exv)
#
#       Метод Рунге для оценки погрешности вычисления интеграла
#       В цикле перебирем i как k, k - число равных частичных отрезков. h = (b-a)/k
#       АСТ(Ньютона) = 2, АСТ(Гаусса) = 3, АСТ = m - 1, L принимаем за 2
#       sumgh1 - квадратурная сумма методом Гаусса с шагов h1, sumnh1 - соответственно методом Ньютона
#       rgh1 - погрешность методом Гаусса с шагов h1, rnh1 - соответственно методом Ньютона
#       Цикл идет, пока значение не достигнет заранее заданной точности
#
# # # # # # # # # # # # # # # ПРАВИЛО РУНГЕ С ВЫБОРОМ ОПТИМАЛЬНОГО ШАГА ВАРИАНТ НЬЮТОНА-КОТЕСА # # # # # # # # # # # #
eps = 0.0001
i = 1
rnh1 = 1
rnh2 = 0
hopt = 0
sumnh1 = 0
sumnh2 = 0
while abs(rnh1) > eps:
    i += 1
    if hopt != 0:
        i = math.ceil((bbig - abig) / hopt)  # Округление вверх
        h = (bbig - abig) / i
    else:
        h = (bbig - abig) / i
    l = 2
    h1 = h
    h2 = h / l
    sumnh1 = 0
    sumnh2 = 0
    for j in range(i):
        sumnh1 += newton(3, abig + j * h1, abig + (j + 1) * h1)
    for j in range(i * 2):
        sumnh2 += newton(3, abig + j * h2, abig + (j + 1) * h2)
    print('****************************')
    print('Разбиваем на ', i, 'участков')
    rnh1 = (sumnh2 - sumnh1) / (1 - l ** (-3))
    print('Погрешность методом Ньютона для h1=', h1, 'равна', rnh1)
    rnh2 = (sumnh2 - sumnh1) / (l ** (3) - 1)
    print('Погрешность методом Ньютона для h2=', h2, 'равна', rnh2)
    # Проверяем равенство
    print(h1*((eps/abs(rnh1))**(1/3)))
    print(h2*((eps/abs(rnh2))**(1/3)))
    # Находим оптимальный шаг разбиения
    hopt = h1 * ((eps / abs(rnh1)) ** (1 / 3))

print('!!!!!!!!!!!!!!!!!!!!!')
# # # # # # # # # # # # # # МЕТОД РИЧАРДСОНА С ВЫБОРОМ ОПТИМАЛЬНОГО ШАГА ВАРИАНТ НЬЮТОНА-КОТЕСА # # # # # # # # # # # #


inn1 = []
outn1 = []
outn2 = []
outn3 = []
outn4 = []
for j in range(1, 10):
    i = 1
    rnh1 = 1
    rnh2 = 0
    rnh3 = 0
    sumnh1 = 0
    sumnh2 = 0
    sumnh3 = 0
    hopt = 0
    cm = 0
    eps = 10 ** (-j)
    inn1.append(-j)
    while abs(rnh1) > eps:
        i += 1
        if hopt != 0:
            i = math.ceil((bbig - abig) / hopt)  # Округление вверх
            h = (bbig - abig) / i
        else:
            h = (bbig - abig) / i
        l = 2
        h1 = h
        h2 = h / l
        h3 = h / (l * l)
        sumnh1 = 0
        sumnh2 = 0
        sumnh3 = 0
        for j in range(i):
            sumnh1 += newton(3, abig + j * h1, abig + (j + 1) * h1)
        for j in range(i * 2):
            sumnh2 += newton(3, abig + j * h2, abig + (j + 1) * h2)
        for j in range(i * 4):
            sumnh3 += newton(3, abig + j * h3, abig + (j + 1) * h3)
        # Ищем скорость сходимости
        m = - (log((sumnh3 - sumnh2) / (sumnh2 - sumnh1)) / log(l))
        cm = (sumnh2 - sumnh1) / ((h ** m) * (1 - 2 ** (-m)))
        print('****************************')
        print('Разбиваем на ', i, 'участков')
        print('Скорость сходимости = ', m)
        rnh1 = (sumnh2 - sumnh1) / (1 - l ** (-m))
        print('Погрешность методом Ньютона для h1=', h1, 'равна', rnh1)
        rnh2 = (sumnh2 - sumnh1) / (l ** m - 1)
        print('Погрешность методом Ньютона для h2=', h2, 'равна', rnh2)
        hopt = h1 * ((eps / abs(rnh1)) ** (1 / m))
    outn1.append(m)
    outn2.append(cm)
    outn3.append(i)
    outn4.append(hopt)

figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(inn1, outn1, '--r')
xlabel('eps')
ylabel('m')

plt.subplot(2, 2, 2)
plt.plot(inn1, outn2, '--g')
xlabel('eps')
ylabel('Cm')

plt.subplot(2, 2, 3)
plt.plot(inn1, outn3, '--b')
xlabel('eps')
ylabel('i')

plt.subplot(2, 2, 4)
plt.plot(inn1, outn4, '--y')
xlabel('eps')
ylabel('h_opt')
show()

print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
# # # # # # # # # # # # ДАЛЕЕ ВЫПОЛНЯЕМ ВСЕ ТО ЖЕ САМОЕ, НО УЖЕ ИСПОЛЬЗУЕМ МЕТОД ГАУССА # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # ПРАВИЛО РУНГЕ С ВЫБОРОМ ОПТИМАЛЬНОГО ШАГА ВАРИАНТ ГАУССА # # # # # # # # # # # #
i = 1
rgh1 = 1
rgh2 = 0
hopt = 0
sumgh1 = 0
sumgh2 = 0
while abs(rgh1) > eps:
    i += 1
    if hopt != 0:
        i = math.ceil((bbig - abig) / hopt)  # Округление вверх
        h = (bbig - abig) / i
    else:
        h = (bbig - abig) / i
    l = 2
    h1 = h
    h2 = h / l
    sumgh1 = 0
    sumgh2 = 0
    for j in range(i):
        sumgh1 += gauss(2, abig + j * h1, abig + (j + 1) * h1)
    for j in range(i * 2):
        sumgh2 += gauss(2, abig + j * h2, abig + (j + 1) * h2)
    print('****************************')
    print('Разбиваем на ', i, 'участков')
    rgh1 = (sumgh2 - sumgh1) / (1 - l ** (-4))
    print('Погрешность методом Гаусса для h1=', h1, 'равна', rgh1)
    rgh2 = (sumgh2 - sumgh1) / (l ** (4) - 1)
    print('Погрешность методом Гаусса для h2=', h2, 'равна', rgh2)
    # Проверяем равенство
    print(h1*((eps/abs(rgh1))**(1/4)))
    print(h2*((eps/abs(rgh2))**(1/4)))
    # Находим оптимальный шаг разбиения
    hopt = h1 * ((eps / abs(rgh1)) ** (1 / 4))

print('!!!!!!!!!!!!!!!!!!!!!')
# # # # # # # # # # # # # # МЕТОД РИЧАРДСОНА С ВЫБОРОМ ОПТИМАЛЬНОГО ШАГА ВАРИАНТ ГАУССА # # # # # # # # # # # #
inn2 = []
outn21 = []
outn22 = []
outn23 = []
outn24 = []
print('coool')
for j in range(10, 11):
    i = 1
    rgh1 = 1
    rgh2 = 0
    rgh3 = 0
    sumgh1 = 0
    sumgh2 = 0
    sumgh3 = 0
    hopt = 0
    cm = 0
    eps = 10 ** (-j)
    inn2.append(-j)
    while abs(rgh1) > eps:
        i += 1
        if hopt != 0:
            i = math.ceil((bbig - abig) / hopt)  # Округление вверх
            h = (bbig - abig) / i
        else:
            h = (bbig - abig) / i
        l = 2
        h1 = h
        h2 = h / l
        h3 = h / (l * l)
        sumgh1 = 0
        sumgh2 = 0
        sumgh3 = 0
        for j in range(i):
            sumgh1 += gauss(2, abig + j * h1, abig + (j + 1) * h1)
        for j in range(i * 2):
            sumgh2 += gauss(2, abig + j * h2, abig + (j + 1) * h2)
        for j in range(i * 4):
            sumgh3 += gauss(2, abig + j * h3, abig + (j + 1) * h3)
        # Ищем скорость сходимости
        m = - (log((sumgh3 - sumgh2) / (sumgh2 - sumgh1)) / log(l))
        cm = (sumnh2 - sumnh1) / ((h ** m) * (1 - 2 ** (-m)))
        print('****************************')
        print('Скорость сходимости = ', m)
        rgh1 = (sumgh2 - sumgh1) / (1 - l ** (-m))
        rgh2 = (sumgh2 - sumgh1) / (l ** m - 1)
        hopt = h1 * ((eps / abs(rgh1)) ** (1 / m))
        print('Cm = ', cm)
        print('Количество разбиений = ', i)
        print('h_opt ', hopt)
    outn21.append(m)
    outn22.append(cm)
    outn23.append(i)
    outn24.append(hopt)

figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(inn2, outn21, 'r')
xlabel('eps')
ylabel('m')

plt.subplot(2, 2, 2)
plt.plot(inn2, outn22, 'r')
xlabel('eps')
ylabel('Cm')

plt.subplot(2, 2, 3)
plt.plot(inn2, outn23, 'r')
xlabel('eps')
ylabel('i')

plt.subplot(2, 2, 4)
plt.plot(inn2, outn24, 'r')
xlabel('eps')
ylabel('h_opt')
show()
print(math.pi)
