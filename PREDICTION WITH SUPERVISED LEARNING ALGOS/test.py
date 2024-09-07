import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 2, 2])
c = a.T
print(c, a)

res = np.dot(a, b)
print(f"output is = {res}")
