import numpy as np
import copy, math

Xtrain = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
Ytrain = np.array([0, 0, 0, 1, 1, 1])

w = np.array([2., 3.])
b = 1.0
alpha = 0.0000001
num_iterations = 1000 

# Sigmoid function
def Sigmoid(fx: np.array) -> np.array:
    return 1 / (1 + np.exp(-fx))

# Compute gradient for logistic regression
def compute_gradient_logistic(X: np.array, Y: np.array, w: np.array, b: float) -> [list, float]:
    m, n = X.shape
    djdw = np.zeros((n, ))
    djdb = 0

    for i in range(m):
        fx = np.dot(X[i], w) + b
        error_val = Sigmoid(fx) - Y[i]  # Use Y[i] for the specific example
        
        # Update gradients
        for j in range(n):
            djdw[j] += error_val * X[i, j]
        djdb += error_val
    
    return djdb / m, djdw / m 

def compute_cost_logistic(X: np.array, y: np.array, w: np.array, b: float) -> float:
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        fx = Sigmoid(z_i)
        cost += -y[i] * np.log(fx) - (1 - y[i]) * np.log(1 - fx)
             
    return cost / m

def gradient_descent(X: np.array, Y: np.array, w_temp: np.array, b: float, alpha: float, num_iterations: int) -> [float, float, list]: 
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_temp)  #avoid modifying global w within function
    
    for i in range(num_iterations):
        # Calculate the gradient and update the parameters
        djdb, djdw = compute_gradient_logistic(X, Y, w, b)   
                
        print(f"djdw: {djdw}\ndjdb: {djdb}")

        w -= alpha * djdw               
        b -= alpha * djdb               
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, Y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iterations / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
        
    return w, b, J_history         #return final w,b and J history for graphing

w, b, J_history = gradient_descent(Xtrain, Ytrain, w, b, alpha, num_iterations)
print(f"\nthe result is = \n")
for i in J_history:
    print(i)
