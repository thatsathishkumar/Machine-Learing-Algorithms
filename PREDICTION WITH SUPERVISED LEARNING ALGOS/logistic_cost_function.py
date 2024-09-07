import numpy as np
import matplotlib.pyplot as plt

Xtrain = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
Ytrain = np.array([0, 0, 0, 1, 1, 1]) 
w = np.array([1, 1])
b = -3

def sigmoid_func(z: np.array) -> np.array:
    return 1 / (np.exp(-z) + 1)
    
def compute_logistic_cost(X: np.array, y: np.array, w: np.array, b: float) -> float:

    m = X.shape[0]

    for i in range(m):
        z_i = np.dot(X[i], w) + b
        fx = sigmoid_func(z_i)
        cost = -(y[i] * np.log(fx) + (1 - y[i]) * np.log(1 - fx))
    return cost / m

def plot_decision_boundary(X, y, w, b):
    # Scatter plot of the data points
    
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color = 'red', label = 'Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color = 'blue', label = 'Class 1')
    
    # Plot the decision boundary
    x1_vals = np.linspace(0, 4, 100)  # Values for x1 (from 0 to 4)
    x2_vals = -(np.dot(w[0], x1_vals) + b) / w[1]  # Solving for x2 from the equation w1*x1 + w2*x2 + b = 0
    
    plt.plot(x1_vals, x2_vals, color = 'green', label = 'Decision Boundary')
    
    # Set the limits and labels
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.grid(True)
    plt.show()

# Call the function to plot decision boundary
plot_decision_boundary(Xtrain, Ytrain, w, b)
    
loss_val = compute_logistic_cost(Xtrain, Ytrain, w, b)
print(f"this predicted loss is = {loss_val}") 