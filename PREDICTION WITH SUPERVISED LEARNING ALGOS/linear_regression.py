import numpy as np

Xtrain = np.array([6, 5, 1, 5, 6, 9, 4])
Ytrain = np.array([836, 359, 745, 710, 554, 381, 608])
m = Xtrain.shape[0]
w, b = 120, 340
alpha = 0.01

my_ft = 4

print(f"this sqft price before prediction {my_ft * w + b}\n")

def compute_cost(x: np.array, y: np.array, w: int, b: int, m: int) -> float:
    cost = 0
    for i in range(m):
        fx = x[i] * w + b
        cost += (fx - y[i]) ** 2
    return (1 / (2 * m)) * cost

def compute_gradient(x: np.array, y: np.array, m: int, w: int, b: int) -> [float, float]:
    djdw, djdb = 0, 0
    for i in range(m):
        fx = x[i] * w + b
        djdwi = (fx - y[i]) * x[i]
        djdbi = (fx - y[i])
        djdw += djdwi
        djdb += djdbi
    djdw /= m
    djdb /= m
    return djdw, djdb

iterations = 1000

for i in range(iterations):
    cost = compute_cost(Xtrain, Ytrain, w, b, m)
    djdw, djdb = compute_gradient(Xtrain, Ytrain, m, w, b)
    w -= alpha * djdw
    b -= alpha * djdb
    if i % 100 == 0:
        print(f"Iteration {i}: Cost = {cost}, w = {w}, b = {b}")

final_cost = compute_cost(Xtrain, Ytrain, w, b, m)
print(f"\nFinal output: Cost = {final_cost}, w = {w}, b = {b}")


print(f"\nthis sqft price after prediction = {my_ft * w + b}")