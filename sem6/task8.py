import math
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

# Наша расчетная схема есть функция
from matplotlib.pyplot import figure

c2 = 0.2
a = 2
b = -1
c = -1
f = []
in1 = []
out1 = []
out1op = []
in2 = []
out2 = []
out22 = []
in3 = []
out3 = []
in4 = []
in4op = []
out4 = []
out4op = []
in5 = []
in5op = []
out5 = []
out5op = []
in6 = []
in6op = []
out6 = []
out6op = []
in7 = []
in7op = []
out7 = []
out7op = []
inr1 = []
inr2 = []
outr1 = []
outr2 = []
outr3 = []
outr4 = []
outr12 = []
outr22 = []
outr32 = []
outr42 = []

def der(x, y):
    f = [0, 0, 0, 0]
    if y[1] == 0 or x == 0:
        f[0] = 0
    else:
        f[0] = 2 * x * (y[1] ** (1 / b)) * y[3]
    if x == 0:
        f[1] = 0
    else:
        f[1] = 2 * b * x * math.exp(b / c * (y[2] - a)) * y[3]
    if x == 0:
        f[2] = 0
    else:
        f[2] = 2 * c * x * y[3]
    if x == 0:
        f[3] = 0
    else:
        f[3] = (-2) * x * math.log(y[0])
    return f


for i in range(4, 16):
    in1.append(-i)
    in2.append(-i)
    out2.append(-i * 2)
    out22.append(-i * 3)
    h = (1 / (2 ** i))
    x0 = 0
    x0op = 0
    y0 = [1, 1, a, 1]
    y0op = [1, 1, a, 1]
    y0next = [1, 1, 1, 1]
    fret = [0, 0, 0, 0]
    cons = [0, 0, 0, 0]
    cons1 = [0, 0, 0, 0]
    while x0 < 5:
        k1 = der(x0, y0)
        for j in range(4):
            cons[j] = y0[j] + h * c2 * k1[j]
        fret = der(x0 + c2 * h, cons)
        for j in range(4):
            y0next[j] = y0[j] + h * (10 / 4) * fret[j] + h * (-3 / 2) * k1[j]
        for j in range(4):
            y0[j] = y0next[j]
        x0 += h

    while x0op < 5:
        k1 = der(x0op, y0op)
        for j in range(4):
            cons[j] = y0op[j] + h * (1 / 3) * k1[j]
        k2 = der(x0op + (1 / 3) * h, cons)
        for j in range(4):
            cons1[j] = y0op[j] + (2 / 3) * h * k2[j]
        k3 = der(x0op + (2 / 3) * h, cons1)
        for j in range(4):
            y0next[j] = y0op[j] + h * (1 / 4) * k1[j] + h * (3 / 4) * k3[j]
        for j in range(4):
            y0op[j] = y0next[j]
        x0op += h

    out1.append(
        (math.log2(math.sqrt((math.exp(math.sin(5 * 5)) - y0[0]) ** 2 + (math.exp(b * math.sin(5 * 5)) - y0[1]) ** 2 + (
                c * math.sin(5 * 5) + a - y0[2]) ** 2 + (math.cos(5 * 5) - y0[3]) ** 2))))

    out1op.append(
        (math.log2(
            math.sqrt((math.exp(math.sin(5 * 5)) - y0op[0]) ** 2 + (math.exp(b * math.sin(5 * 5)) - y0op[1]) ** 2 + (
                    c * math.sin(5 * 5) + a - y0op[2]) ** 2 + (math.cos(5 * 5) - y0op[3]) ** 2))))

figure(figsize=(12, 8))

plt.plot()
plt.plot(in1, out1, 'r')
plt.plot(in1, out1op, 'b')
plt.plot(in2, out2, 'g')
plt.plot(in2, out22, 'g')
xlabel('Степень двойки')
ylabel('Двоичный логарифм нормы погрешности')

show()

# Обычный метод, движение с выбором оптимального шага

for i in range(5, 6):
    in3.append(i)
    h1 = (1 / (2 ** i))
    h2 = h1 / 2
    x1 = 0
    x2 = 0
    y1 = [1, 1, a, 1]
    y2 = [1, 1, a, 1]
    y1next = [1, 1, 1, 1]
    y2next = [1, 1, 1, 1]
    fret1 = [0, 0, 0, 0]
    cons1 = [0, 0, 0, 0]
    fret2 = [0, 0, 0, 0]
    cons2 = [0, 0, 0, 0]
    k1 = 0
    k2 = 0
    while x2 < 5:
        if x1 < 5:
            k1 = der(x1, y1)
        k2 = der(x2, y2)
        for j in range(4):
            if x1 < 5:
                cons1[j] = y1[j] + h1 * c2 * k1[j]
            cons2[j] = y2[j] + h2 * c2 * k2[j]
        if x1 < 5:
            fret1 = der(x1 + c2 * h1, cons1)
        fret2 = der(x2 + c2 * h2, cons2)
        for j in range(4):
            if x1 < 5:
                y1next[j] = y1[j] + h1 * (10 / 4) * fret1[j] + h1 * (-3 / 2) * k1[j]
            y2next[j] = y2[j] + h2 * (10 / 4) * fret2[j] + h2 * (-3 / 2) * k2[j]
        for j in range(4):
            if x1 < 5:
                y1[j] = y1next[j]
            y2[j] = y2next[j]
        if x1 < 5:
            x1 += h1
        x2 += h2

    norm = (sqrt((y1[0] - y2[0]) ** 2 + (y1[1] - y2[1]) ** 2 + (y1[2] - y2[2]) ** 2 + (y1[3] - y2[3]) ** 2) )

    htol = (h1 / 2) * math.sqrt(0.00001 / norm)  # Нашли оптимальный шаг для i-ой степени двойки

    h = htol

    x0 = 0
    y0 = [1, 1, a, 1]
    y0next = [1, 1, 1, 1]
    fret = [0, 0, 0, 0]
    cons = [0, 0, 0, 0]
    while x0 < 5:
        in4.append(x0)
        k1 = der(x0, y0)
        for j in range(4):
            cons[j] = y0[j] + h * c2 * k1[j]
        fret = der(x0 + c2 * h, cons)
        for j in range(4):
            y0next[j] = y0[j] + h * (10 / 4) * fret[j] + h * (-3 / 2) * k1[j]
        for j in range(4):
            y0[j] = y0next[j]
        x0 += h
        out4.append(
            math.sqrt((math.exp(math.sin(x0 * x0)) - y0[0]) ** 2 + (math.exp(b * math.sin(x0 * x0)) - y0[1]) ** 2 + (
                    c * math.sin(x0 * x0) + a - y0[2]) ** 2 + (math.cos(x0 * x0) - y0[3]) ** 2))

    # out3.append(math.sqrt((math.exp(math.sin(5 * 5)) - y0[0]) ** 2 + (math.exp(b * math.sin(5 * 5)) - y0[1]) ** 2 + (
    # c * math.sin(5 * 5) + a - y0[2]) ** 2 + (math.cos(5 * 5) - y0[3]) ** 2))
# Метод оппонент, движение с выбором оптимального шага
for i in range(5, 6):
    in3.append(i)
    h1 = (1 / (2 ** i))
    h2 = h1 / 2
    x1 = 0
    x2 = 0
    y1 = [1, 1, a, 1]
    y2 = [1, 1, a, 1]
    y1next = [1, 1, 1, 1]
    y2next = [1, 1, 1, 1]
    fret1 = [0, 0, 0, 0]
    cons1 = [0, 0, 0, 0]
    fret2 = [0, 0, 0, 0]
    cons2 = [0, 0, 0, 0]
    k11 = [0, 0, 0, 0]
    k12 = [0, 0, 0, 0]
    k21 = [0, 0, 0, 0]
    k22 = [0, 0, 0, 0]
    k31 = [0, 0, 0, 0]
    k32 = [0, 0, 0, 0]
    k1 = 0
    k2 = 0
    while x2 < 5:
        if x1 < 5:
            k11 = der(x1, y1)
        k12 = der(x2, y2)
        for j in range(4):
            if x1 < 5:
                cons1[j] = y1[j] + h1 * (1 / 3) * k11[j]
            cons2[j] = y2[j] + h2 * (1 / 3) * k12[j]
        if x1 < 5:
            k21 = der(x1 + (1 / 3) * h1, cons1)
        k22 = der(x2 + (1 / 3) * h2, cons2)
        for j in range(4):
            if x1 < 5:
                fret1[j] = y1[j] + (2 / 3) * h1 * k21[j]
            fret2[j] = y2[j] + (2 / 3) * h2 * k22[j]
        if x1 < 5:
            k31 = der(x1 + (2 / 3) * h1, fret1)
        k32 = der(x2 + (2 / 3) * h2, fret2)
        for j in range(4):
            if x1 < 5:
                y1next[j] = y1[j] + h1 * (1 / 4) * k11[j] + h1 * (3 / 4) * k31[j]
            y2next[j] = y2[j] + h2 * (1 / 4) * k12[j] + h2 * (3 / 4) * k32[j]
        for j in range(4):
            if x1 < 5:
                y1[j] = y1next[j]
            y2[j] = y2next[j]
        if x1 < 5:
            x1 += h1
        x2 += h2

    norm = (sqrt((y1[0] - y2[0]) ** 2 + (y1[1] - y2[1]) ** 2 + (y1[2] - y2[2]) ** 2 + (y1[3] - y2[3]) ** 2) )

    htol = (h1 / 2) * ((0.00001 / norm) ** (1 / 3))  # Нашли оптимальный шаг для i-ой степени двойки

    h = htol

    x0op = 0
    y0op = [1, 1, a, 1]
    y0next = [1, 1, 1, 1]
    fret = [0, 0, 0, 0]
    cons = [0, 0, 0, 0]
    cons1 = [0, 0, 0, 0]
    while x0op < 5:
        in4op.append(x0op)
        k1 = der(x0op, y0op)
        for j in range(4):
            cons[j] = y0op[j] + h * (1 / 3) * k1[j]
        k2 = der(x0op + (1 / 3) * h, cons)
        for j in range(4):
            cons1[j] = y0op[j] + (2 / 3) * h * k2[j]
        k3 = der(x0op + (2 / 3) * h, cons1)
        for j in range(4):
            y0next[j] = y0op[j] + h * (1 / 4) * k1[j] + h * (3 / 4) * k3[j]
        for j in range(4):
            y0op[j] = y0next[j]
        x0op += h
        out4op.append(math.sqrt(
            (math.exp(math.sin(x0op * x0op)) - y0op[0]) ** 2 + (math.exp(b * math.sin(x0op * x0op)) - y0op[1]) ** 2 + (
                    c * math.sin(x0op * x0op) + a - y0op[2]) ** 2 + (math.cos(x0op * x0op) - y0op[3]) ** 2))

figure(figsize=(12, 8))

plt.plot()
plt.plot(in4, out4, 'r')
plt.plot(in4op, out4op, 'b')
xlabel('x на промежутке интегрирования')
ylabel('Норма точной полной погрешности')

show()

# 4 пункт

atol = 10e-12
rtol = 10e-6
tol = 0.000001
x01 = 0
x02 = 0
x01n = 0
x02n = 0
y1 = [1, 1, a, 1]
y2 = [1, 1, a, 1]
y1n = [1, 1, a, 1]
y2n = [1, 1, a, 1]
h1 = 0.0512
h2 = h1 / 2
cons = [0, 0, 0, 0]
y1next = [0, 0, 0, 0]
y2next = [0, 0, 0, 0]
y1nextn = [0, 0, 0, 0]
y2nextn = [0, 0, 0, 0]
r1n = [0, 0, 0, 0]
x = 0
y = [1, 1, a, 1]
ynext = [0, 0, 0, 0]
k = [0, 0, 0, 0]
fret = [0, 0, 0, 0]
in5.append(x)
out5.append(h1)
in6.append(x)
out6.append(0)

while x < 5:
    tol = (rtol * sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]) + atol)
    x01 = x
    x02 = x
    for i in range(4):
        y1[i] = y[i]
    for i in range(4):
        y2[i] = y[i]

    k1 = der(x01, y1)
    for j in range(4):
        cons[j] = y1[j] + h1 * c2 * k1[j]
    fret = der(x01 + c2 * h1, cons)
    for j in range(4):
        y1next[j] = y1[j] + h1 * (10 / 4) * fret[j] + h1 * (-3 / 2) * k1[j]
    for j in range(4):
        y1[j] = y1next[j]
    x01 += h1

    for i in range(0, 2):
        k1 = der(x02, y2)
        for j in range(4):
            cons[j] = y2[j] + h2 * c2 * k1[j]
        fret = der(x02 + c2 * h2, cons)
        for j in range(4):
            y2next[j] = y2[j] + h2 * (10 / 4) * fret[j] + h2 * (-3 / 2) * k1[j]
        for j in range(4):
            y2[j] = y2next[j]
        x02 += h2

    for i in range(4):
        r1n[i] = ((y2[i] - y1[i]) /  3)

    norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))

    while norm > (tol * 4):
        x01 = x
        x02 = x
        for i in range(4):
            y1[i] = y[i]
        for i in range(4):
            y2[i] = y[i]
        h1 = h1 / 2
        h2 = h1 / 2

        k1 = der(x01, y1)
        for j in range(4):
            cons[j] = y1[j] + h1 * c2 * k1[j]
        fret = der(x01 + c2 * h1, cons)
        for j in range(4):
            y1next[j] = y1[j] + h1 * (10 / 4) * fret[j] + h1 * (-3 / 2) * k1[j]
        for j in range(4):
            y1[j] = y1next[j]
        x01 += h1

        for i in range(0, 2):
            k1 = der(x02, y2)
            for j in range(4):
                cons[j] = y2[j] + h2 * c2 * k1[j]
            fret = der(x02 + c2 * h2, cons)
            for j in range(4):
                y2next[j] = y2[j] + h2 * (10 / 4) * fret[j] + h2 * (-3 / 2) * k1[j]
            for j in range(4):
                y2[j] = y2next[j]
            x02 += h2

        for i in range(4):
            r1n[i] = ((y2[i] - y1[i]) / 3)

        norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))

    inr2.append(x)
    in5.append(x)
    out5.append(h1)

    out5.append(h1)
    out6.append(norm)
    if tol < norm <= tol * 4:
        x += h1
        for i in range(4):
            y[i] = y2[i]
        h1 = h1 / 2
        h2 = h1 / 2

    elif tol * 0.125 <= norm <= tol:
        x += h1
        for i in range(4):
            y[i] = y1[i]
        h1 = h1
        h2 = h1 / 2

    elif norm < tol * 0.125:
        x += h1
        for i in range(4):
            y[i] = y1[i]
        h1 = h1 * 2
        h2 = h1 / 2

    in5.append(x)
    in6.append(x)
    outr12.append(y[0])
    outr22.append(y[1])
    outr32.append(y[2])
    outr42.append(y[3])

x01 = 0
x02 = 0
x01n = 0
x02n = 0
y1 = [1, 1, a, 1]
y2 = [1, 1, a, 1]
y1n = [1, 1, a, 1]
y2n = [1, 1, a, 1]
h1 = 0.1024
h2 = h1 / 2
cons = [0, 0, 0, 0]
cons1 = [0, 0, 0, 0]
y1next = [0, 0, 0, 0]
y2next = [0, 0, 0, 0]
y1nextn = [0, 0, 0, 0]
y2nextn = [0, 0, 0, 0]
r1n = [0, 0, 0, 0]
x = 0
y = [1, 1, a, 1]
ynext = [0, 0, 0, 0]
k = [0, 0, 0, 0]
fret = [0, 0, 0, 0]
in5op.append(x)
out5op.append(h1)
in6op.append(x)
out6op.append(0)

while x < 5:
    tol = (rtol * sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]) + atol)
    x01 = x
    x02 = x
    for i in range(4):
        y1[i] = y[i]
    for i in range(4):
        y2[i] = y[i]

    k1 = der(x01, y1)
    for j in range(4):
        cons[j] = y1[j] + h1 * (1 / 3) * k1[j]
    k2 = der(x01 + (1 / 3) * h1, cons)
    for j in range(4):
        cons1[j] = y1[j] + (2 / 3) * h1 * k2[j]
    k3 = der(x01 + (2 / 3) * h1, cons1)
    for j in range(4):
        y1next[j] = y1[j] + h1 * (1 / 4) * k1[j] + h1 * (3 / 4) * k3[j]
    for j in range(4):
        y1[j] = y1next[j]
    x01 += h1

    for i in range(0, 2):
        k1 = der(x02, y2)
        for j in range(4):
            cons[j] = y2[j] + h2 * (1 / 3) * k1[j]
        k2 = der(x02 + (1 / 3) * h2, cons)
        for j in range(4):
            cons1[j] = y2[j] + (2 / 3) * h2 * k2[j]
        k3 = der(x02 + (2 / 3) * h2, cons1)
        for j in range(4):
            y2next[j] = y2[j] + h2 * (1 / 4) * k1[j] + h2 * (3 / 4) * k3[j]
        for j in range(4):
            y2[j] = y2next[j]
        x02 += h2

    for i in range(4):
        r1n[i] = ((y2[i] - y1[i]) / 7)

    norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))
    while norm > (tol * 8):
        x01 = x
        x02 = x
        for i in range(4):
            y1[i] = y[i]
        for i in range(4):
            y2[i] = y[i]
        h1 = h1 / 2
        h2 = h1 / 2

        k1 = der(x01, y1)
        for j in range(4):
            cons[j] = y1[j] + h1 * (1 / 3) * k1[j]
        k2 = der(x01 + (1 / 3) * h1, cons)
        for j in range(4):
            cons1[j] = y1[j] + (2 / 3) * h1 * k2[j]
        k3 = der(x01 + (2 / 3) * h1, cons1)
        for j in range(4):
            y1next[j] = y1[j] + h1 * (1 / 4) * k1[j] + h1 * (3 / 4) * k3[j]
        for j in range(4):
            y1[j] = y1next[j]
        x01 += h1

        for i in range(0, 2):
            k1 = der(x02, y2)
            for j in range(4):
                cons[j] = y2[j] + h2 * (1 / 3) * k1[j]
            k2 = der(x02 + (1 / 3) * h2, cons)
            for j in range(4):
                cons1[j] = y2[j] + (2 / 3) * h2 * k2[j]
            k3 = der(x02 + (2 / 3) * h2, cons1)
            for j in range(4):
                y2next[j] = y2[j] + h2 * (1 / 4) * k1[j] + h2 * (3 / 4) * k3[j]
            for j in range(4):
                y2[j] = y2next[j]
            x02 += h2

        for i in range(4):
            r1n[i] = ((y2[i] - y1[i]) / 7)

        norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))

    inr1.append(x)
    in5op.append(x)
    out5op.append(h1)

    out5op.append(h1)
    out6op.append(norm)

    if tol < norm <= tol * 8:
        x += h1
        for i in range(4):
            y[i] = y2[i]
        h1 = h1 / 2
        h2 = h1 / 2

    elif tol * 0.0625 <= norm <= tol:
        x += h1
        for i in range(4):
            y[i] = y1[i]
        h1 = h1
        h2 = h1 / 2

    elif norm < tol * 0.0625:
        x += h1
        for i in range(4):
            y[i] = y1[i]
        h1 = h1 * 2
        h2 = h1 / 2

    in5op.append(x)
    in6op.append(x)
    outr1.append(y[0])
    outr2.append(y[1])
    outr3.append(y[2])
    outr4.append(y[3])

for k in range(4, 9):
    rtol = 10 ** (-k)
    in7.append(-k)
    in7op.append(-k)
    x = 0
    y = [1, 1, a, 1]
    s = 0
    while x < 5:
        tol = rtol * sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]) + atol
        x01 = x
        x02 = x
        for i in range(4):
            y1[i] = y[i]
        for i in range(4):
            y2[i] = y[i]

        k1 = der(x01, y1)
        s += 1
        for j in range(4):
            cons[j] = y1[j] + h1 * c2 * k1[j]
        fret = der(x01 + c2 * h1, cons)
        s += 1
        for j in range(4):
            y1next[j] = y1[j] + h1 * (10 / 4) * fret[j] + h1 * (-3 / 2) * k1[j]
        for j in range(4):
            y1[j] = y1next[j]
        x01 += h1

        for i in range(0, 2):
            k1 = der(x02, y2)
            s += 1
            for j in range(4):
                cons[j] = y2[j] + h2 * c2 * k1[j]
            fret = der(x02 + c2 * h2, cons)
            s += 1
            for j in range(4):
                y2next[j] = y2[j] + h2 * (10 / 4) * fret[j] + h2 * (-3 / 2) * k1[j]
            for j in range(4):
                y2[j] = y2next[j]
            x02 += h2

        for i in range(4):
            r1n[i] = ((y2[i] - y1[i]) / 3)

        norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))
        while norm > (tol * 4):
            x01 = x
            x02 = x
            for i in range(4):
                y1[i] = y[i]
            for i in range(4):
                y2[i] = y[i]
            h1 = h1 / 2
            h2 = h1 / 2

            k1 = der(x01, y1)
            s += 1
            for j in range(4):
                cons[j] = y1[j] + h1 * c2 * k1[j]
            fret = der(x01 + c2 * h1, cons)
            s += 1
            for j in range(4):
                y1next[j] = y1[j] + h1 * (10 / 4) * fret[j] + h1 * (-3 / 2) * k1[j]
            for j in range(4):
                y1[j] = y1next[j]
            x01 += h1

            for i in range(0, 2):
                k1 = der(x02, y2)
                s += 1
                for j in range(4):
                    cons[j] = y2[j] + h2 * c2 * k1[j]
                fret = der(x02 + c2 * h2, cons)
                s += 1
                for j in range(4):
                    y2next[j] = y2[j] + h2 * (10 / 4) * fret[j] + h2 * (-3 / 2) * k1[j]
                for j in range(4):
                    y2[j] = y2next[j]
                x02 += h2

            for i in range(4):
                r1n[i] = (y2[i] - y1[i]) / 3

            norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))

        if tol < norm <= tol * 4:
            x += h1
            for i in range(4):
                y[i] = y2[i]
            h1 = h1 / 2
            h2 = h1 / 2

        elif tol * 0.125 <= norm <= tol:
            x += h1
            for i in range(4):
                y[i] = y1[i]
            h1 = h1
            h2 = h1 / 2

        elif norm < tol * 0.125:
            x += h1
            for i in range(4):
                y[i] = y1[i]
            h1 = h1 * 2
            h2 = h1 / 2
    out7.append(log2(s))

    x = 0
    y = [1, 1, a, 1]
    s = 0
    while x < 5:
        tol = (rtol * sqrt(y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[3] * y[3]) + atol)
        x01 = x
        x02 = x
        for i in range(4):
            y1[i] = y[i]
        for i in range(4):
            y2[i] = y[i]

        k1 = der(x01, y1)
        s += 1
        for j in range(4):
            cons[j] = y1[j] + h1 * (1 / 3) * k1[j]
        k2 = der(x01 + (1 / 3) * h1, cons)
        s += 1
        for j in range(4):
            cons1[j] = y1[j] + (2 / 3) * h1 * k2[j]
        k3 = der(x01 + (2 / 3) * h1, cons1)
        s += 1
        for j in range(4):
            y1next[j] = y1[j] + h1 * (1 / 4) * k1[j] + h1 * (3 / 4) * k3[j]
        for j in range(4):
            y1[j] = y1next[j]
        x01 += h1

        for i in range(0, 2):
            k1 = der(x02, y2)
            s += 1
            for j in range(4):
                cons[j] = y2[j] + h2 * (1 / 3) * k1[j]
            k2 = der(x02 + (1 / 3) * h2, cons)
            s += 1
            for j in range(4):
                cons1[j] = y2[j] + (2 / 3) * h2 * k2[j]
            k3 = der(x02 + (2 / 3) * h2, cons1)
            s += 1
            for j in range(4):
                y2next[j] = y2[j] + h2 * (1 / 4) * k1[j] + h2 * (3 / 4) * k3[j]
            for j in range(4):
                y2[j] = y2next[j]
            x02 += h2

        for i in range(4):
            r1n[i] = ((y2[i] - y1[i]) / 7)

        norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))
        while norm > (tol * 8):
            x01 = x
            x02 = x
            for i in range(4):
                y1[i] = y[i]
            for i in range(4):
                y2[i] = y[i]
            h1 = h1 / 2
            h2 = h1 / 2

            k1 = der(x01, y1)
            s += 1
            for j in range(4):
                cons[j] = y1[j] + h1 * (1 / 3) * k1[j]
            k2 = der(x01 + (1 / 3) * h1, cons)
            s += 1
            for j in range(4):
                cons1[j] = y1[j] + (2 / 3) * h1 * k2[j]
            k3 = der(x01 + (2 / 3) * h1, cons1)
            s += 1
            for j in range(4):
                y1next[j] = y1[j] + h1 * (1 / 4) * k1[j] + h1 * (3 / 4) * k3[j]
            for j in range(4):
                y1[j] = y1next[j]
            x01 += h1

            for i in range(0, 2):
                k1 = der(x02, y2)
                s += 1
                for j in range(4):
                    cons[j] = y2[j] + h2 * (1 / 3) * k1[j]
                k2 = der(x02 + (1 / 3) * h2, cons)
                s += 1
                for j in range(4):
                    cons1[j] = y2[j] + (2 / 3) * h2 * k2[j]
                k3 = der(x02 + (2 / 3) * h2, cons1)
                s += 1
                for j in range(4):
                    y2next[j] = y2[j] + h2 * (1 / 4) * k1[j] + h2 * (3 / 4) * k3[j]
                for j in range(4):
                    y2[j] = y2next[j]
                x02 += h2

            for i in range(4):
                r1n[i] = (y2[i] - y1[i]) / 7

            norm = (sqrt(r1n[0] ** 2 + r1n[1] ** 2 + r1n[2] ** 2 + r1n[3] ** 2))

        if tol < norm <= tol * 8:
            x += h1
            for i in range(4):
                y[i] = y2[i]
            h1 = h1 / 2
            h2 = h1 / 2

        elif tol * 0.0625 <= norm <= tol:
            x += h1
            for i in range(4):
                y[i] = y1[i]
            h1 = h1
            h2 = h1 / 2

        elif norm < tol * 0.0625:
            x += h1
            for i in range(4):
                y[i] = y1[i]
            h1 = h1 * 2
            h2 = h1 / 2

    out7op.append(log2(s))

figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(in5, out5, 'r')
plt.plot(in5op, out5op, 'b')
xlabel('x на промежутке интегрирования')
ylabel('Длина выбираемого шага')

plt.subplot(2, 2, 2)
plt.plot(in6, out6, 'r')
plt.plot(in6op, out6op, 'b')
xlabel('x на промежутке интегрирования')
ylabel('Норма точной полной погрешности')

plt.subplot(2, 2, 3)
plt.plot(in7, out7, 'r')
plt.plot(in7op, out7op, 'b')
xlabel('Степень rtol 10')
ylabel('Двоичный логарифм числа обращений к правой части')

plt.subplot(2, 2, 4)
plt.plot(inr2, outr12, 'r')
plt.plot(inr2, outr22, 'r')
plt.plot(inr2, outr32, 'r')
plt.plot(inr2, outr42, 'r')
plt.plot(inr1, outr1, 'b')
plt.plot(inr1, outr2, 'b')
plt.plot(inr1, outr3, 'b')
plt.plot(inr1, outr4, 'b')
xlabel('x на промежутке интегрирования')
ylabel('решение, вектор y')


show()
# Красный наш, оппонент синий, прочее зеленый

