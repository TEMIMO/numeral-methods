from math import cos
from pylab import *
import numpy
from scipy import integrate

x = [-1, -0.5, 0.0, 0.65, 1.0]
y = []


def flam(p): return p ** 2 * cos(p)


def flam0(p): return p ** 2 * cos(p)


def flam1(p): return p ** 2 * cos(p) * p


def flam2(p): return p ** 2 * cos(p) * (3/2 * p * p - 1/2)


def flam3(p): return p ** 2 * cos(p) * (5/2 * p ** 3 - 3/2 * p)


# def flam5(p): return p ** 2 * cos(p) * (35/8 * x * x * x * x - 30/8 * x *x + 3/8)

for i in range(5):
    y.append(flam(x[i]))


def q0(p): return 1


def q1(p): return p


def q2(p): return (3 / 2) * p * p - (1 / 2)


def q3(p): return (5 / 2) * p * p * p - (3 / 2) * p


# def q4(p): return 35/8 * p * p * p * p - 30/8 * p * p + 3/8


def q02(p): return q0(p) ** 2


def q12(p): return q1(p) ** 2


def q22(p): return q2(p) ** 2


def q32(p): return q3(p) ** 2

# def q42(p):
#     return q4(p) ** 2


def deg(a=0, b=0):
    res = 0
    for i in range(5):
        res += (x[i] ** a) * (y[i] ** b)
    return res


M1 = numpy.array([[2*deg(6), 2*deg(5), 2*deg(4), 2*deg(3)], [2*deg(5), 2*deg(4), 2*deg(3), 2*deg(2)],
                  [2*deg(4), 2*deg(3), 2*deg(2), 2*deg(1)], [2*deg(3), 2*deg(2), 2*deg(1), 2]])
V1 = numpy.array(([2*deg(3, 1), 2*deg(2, 1), 2*deg(1, 1), 2*deg(0, 1)]))
solution = numpy.linalg.solve(M1, V1)

t1 = linspace(-1, 1, num=100)

az0, az1, az2, az3 = integrate.quad(q02, -1, 1)[0], integrate.quad(q12, -1, 1)[0], integrate.quad(q22, -1, 1)[0], \
                     integrate.quad(q32, -1, 1)[0]

# az4 = integrate.quad(q42, -1, 1)[0]

a1, a2, a3, a4 = integrate.quad(flam0, -1, 1)[0], integrate.quad(flam1, -1, 1)[0], integrate.quad(flam2, -1, 1)[0], \
                 integrate.quad(flam3, -1, 1)[0]

# a5 = integrate.quad(flam5, -1, 1)[0]

if a1 == 0:
    k1 = 0
else:
    k1 = a1 / az0

if a2 == 0:
    k2 = 0
else:
    k2 = a2 / az1

if a3 == 0:
    k3 = 0
else:
    k3 = a3 / az2

if a4 == 0:
    k4 = 0
else:
    k4 = a4 / az3

# if a5 == 0:
#     k5 = 0
# else:
#     k5 = a5/az4


t4 = k1 * q0(t1) + k2 * q1(t1) + k3 * q2(t1) + k4 * q3(t1)#+k5*q4(t1)
t3 = t1 * t1 * cos(t1)
t2 = t1 ** 3 * solution[0] + t1 ** 2 * solution[1] + t1 * solution[2] + solution[3]

figure(figsize=(12, 8))
plt.plot(x, y, '*', t1, t2, 'r', t1, t3, 'b', t1, t4, 'g')
axes = plt.gca()
axes.set_ylim([-1, 1])
axes.set_xlim([-1, 1])
xlabel('x')
ylabel('y')
show()
