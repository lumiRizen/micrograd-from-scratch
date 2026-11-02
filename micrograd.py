# micrograd = tiny autograd engine (autograd = automatic gradient).
# Goal: for ANY expression, compute how much the output moves when each input is nudged.
# That's backpropagation, the core of training neural nets.
import math
import numpy as np
import matplotlib.pyplot as plt





# ---- Part 1: derivative of a one-input function ----
def f(x):
    return 3*x**2 - 4*x + 5

print(f(3.0))                 # 20.0, matches the lecture
xs = np.arange(-5, 5, 0.25)   # -5 up to (not including) 5, step 0.25
ys = f(xs)                    # numpy applies f to every point at once
plt.plot(xs, ys)
plt.show()                    # needed in a .py file to display the plot

# slope ≈ (f(x+h) - f(x)) / h : "nudge x by h, how much does f move per unit of h?"
h = 0.000001
for x in (3.0, -3.0, 2/3):
    print(x, (f(x + h) - f(x)) / h)
# ≈14 (up), ≈-22 (down), ≈0 (flat = bottom of the curve).
# Why it matters: training = walking downhill on the loss, and slope says which way is down.





# ---- Part 2: several inputs, nudge one at a time ----
h = 0.0001
a, b, c = 2.0, -3.0, 10.0
d1 = a*b + c
c += h                         # nudge only c
d2 = a*b + c
print('d1', d1, 'd2', d2)
print('slope wrt c', (d2 - d1) / h)   # ≈1: d moves 1-for-1 with c (it's just added)

#----- Part 2b: nudge a and b separately --------
#prediction: slope wrt a = -3, slope wrt b = 2
a, b, c = 2.0, -3.0, 10.0                  #reset since part 2 changed c
h = 0.0001
d1 = a*b + c                               
d2 = (a + h)*b + c                         #nudge only a
print('slope wrt a', (d2 - d1)/h)
d3 = a*(b + h) + c                         #nudge only b
print('slope wrt b', (d3 - d1)/h)






# ---- Part 3: Value remembers how it was made ----
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data                # the number itself
        self._prev = set(_children)     # which Values produced this one (the graph)
        self._op = _op                  # which operation produced it

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):           # runs on  x + y
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):           # runs on  x * y
        return Value(self.data * other.data, (self, other), '*')

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a*b + c                             # Python reads: (a.__mul__(b)).__add__(c)
print(d)                                # Value(data=4.0)
print(d._prev, d._op)                   # the two parents of d, and '+'


