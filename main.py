import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import math

def f(x):
  return x*np.log(x**2)

m = 60
n = 5

##------генерация точек-----##
x_ = np.linspace(-1, 1, n)
X = np.array([[i] * (m // n) for i in x_])
X = np.ndarray.flatten(X)
y_ = np.zeros((n, m // n))
for i in range(n):
    y_[i] = np.array([f(x_[i]) for j in range(m // n)]) + np.random.randn(m // n)/15
X_ = X
y = np.ndarray.flatten(y_)
##----------------------------


def norm_eq(x):
  answ = 0
  E = np.vander(X_, N = n, increasing=True)
  a = np.linalg.solve(np.transpose(E)@E, np.transpose(E)@y)
  for i in range(n):
      answ += a[i] * x**i
  return answ

x = 0
q = [i for i in range(n)]
q[0] = 1
alpha = [None for i in range(m)]
beta = [None for i in range(m)]
xsum = 0

def q(x, j):
  if j == 0: return 1
  if j == 1: return x - sum(X_) / m
  if alpha[j] == None:
        alpha[j] = sum([X_[i] * q(X_[i], j - 1)**2 for i in range(m)]) / sum([q(X_[i], j - 1)**2 for i in range(m)])
  if beta[j - 1] == None:
        beta[j - 1] = sum([X_[i] * q(X_[i], j - 1) * q(X_[i], j - 2) for i in range(m)]) / sum([q(X_[i], j - 2)**2 for i in range(m)])

  return x * q(x, j - 1) - alpha[j] * q(x, j - 1) - beta[j - 1] * q(x, j - 2)

coef = [0 for _ in range(n)]
for k in range(n):
    coef[k] = sum([q(X_[s], k) * f(X_[s]) for s in range(m)]) / sum([q(X_[s], k)**2 for s in range(m)])
x = 0
def _q(x):
    return sum([coef[k] * q(x, k) for k in range(n)])
result = []
for x in x_:
    result.append(_q(x))


def show_functions():
    x1 = np.linspace(-1, 1, 1000)
    plt.figure(figsize=(6, 5))
    plt.plot(x1, f(x1), label='real', c='r')
    plt.plot(x1, [_q(x) for x in x1], label='with orthogonal polynomials', c='b')
    plt.scatter(X_, y, label='dots', c='orange', marker='.')
    y1 = [_ for _ in range(len(x1))]
    for i in range(len(x1)): y1[i] = norm_eq(x1[i])
    plt.plot(x1, y1, label='with normal equations', c='g')

    plt.xlim([-1.1, 1.1])
    plt.ylim([-1.1, 1.4])
    plt.title("approximation")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.show()

show_functions()
