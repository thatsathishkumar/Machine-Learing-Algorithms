import numpy as np

def sigmoid_func(z: np.array) -> np.array:
    return 1 / (np.exp(-z) + 1)
    
def compute_logistic_cost(X: np.array, y: np.array, w: np.array, b: float) -> float:

    m = X.shape[0]

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        fx = sigmoid_func(z_i)
        cost = -(y[i] * np.log(fx) + (1 - y[i]) * np.log(1 - fx))
    return cost / m

def compute_gradient_descent(X: np.array, y: np.array, w: np.array, b: float):
    m, n = X.shape

    djdw = np.zeros((n,)) 
    djdb = 0

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        fx = sigmoid_func(z_i)
        error = fx - y[i]

        for j in range(n):
                   djdw[j] += (error * X[i,j])    
            
        djdb += error                                 
        
    return djdb / m, djdw / m
        

Xtrain = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
Ytrain = np.array([0, 0, 0, 1, 1, 1])
w = np.array([2.,3.])
b = 1.0
alpha = 0.0000001
num_iterations = 1000

error_val = compute_logistic_cost(Xtrain, Ytrain, w, b)

