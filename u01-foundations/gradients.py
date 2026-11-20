"""Unit 1: derivatives, partial derivatives and the gradient.

We compare exact derivatives with numerical ones (the change in f
for a very small change in the input), then take a few steps
downhill on a function of two variables.
"""
import numpy as np


def f(w):                       # one variable: minimum at w = 3
    return (w - 3) ** 2


def g(v):                       # two variables: minimum at (1, -2)
    x, y = v
    return (x - 1) ** 2 + 2 * (y + 2) ** 2


def grad_g(v):                  # exact partial derivatives
    x, y = v
    return np.array([2 * (x - 1), 4 * (y + 2)])


h = 1e-6
print("f'(5) exact = 4, numerical =", round((f(5 + h) - f(5)) / h, 4))

v = np.array([4.0, 1.0])
num = [(g(v + h * e) - g(v)) / h for e in np.eye(2)]
print("gradient of g at (4, 1): exact", grad_g(v),
      "numerical", np.round(num, 3))

for lr in [0.05, 0.4, 0.6]:  # gradient descent, learning rate 0.05, 0.4, 0.6
    v = np.array([4.0, 1.0])

    print(f"\nLearning rate: {lr}")

    for step in range(1, 6):
        v = v - lr * grad_g(v)
        print(f"step {step}: v = {np.round(v, 3)}  g(v) = {g(v):.4f}")

# Which Learning Rate Converges Faster: Learning Rate 0.4
# Which Learning Rate Slow convergence: Learning Rate 0.05
# Which Learning Rate Diverges: Learning Rate 0.6
# The function is g(x,y)=(x-1)**2+2(y+2)**2. For gradient descent to converge 
# on this quadratic function, the learning rate must satisfy:
# 0<α<2/4=0.5.
# Therefore:
# 0.05 is within the convergence range but relatively small.
# 0.4 is within the range and converges faster.
# 0.6 exceeds the stability limit and diverges.

#A learning rate that is too small makes learning slow, while one that is too large can prevent convergence.
