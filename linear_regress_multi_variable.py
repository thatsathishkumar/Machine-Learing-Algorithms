import numpy as np

# Given data
size_ft = np.array([1306, 1423, 1046, 1257])
no_of_room = np.array([3, 5, 2, 3])
no_of_floors = np.array([3, 4, 2, 3])
house_age = np.array([34, 37, 24, 45])

Xtrain = np.zeros((4, 4))

# Assigning the data to Xtrain
Xtrain[:, 0] = size_ft
Xtrain[:, 1] = no_of_room
Xtrain[:, 2] = no_of_floors
Xtrain[:, 3] = house_age

# Target values
Ytrain = np.array([200, 450, 160, 180]) 

w = np.array([103, 204, 173, 145])
b = 437

alpha = 0.000001  # Small learning rate to avoid large steps
num_iterations = 1000


# Function to compute the cost
def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b   # prediction
        cost += (f_wb_i - y[i])**2     # squared error
    cost /= (2 * m)              # average squared error
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, w, b, alpha, num_iterations):
    m = X.shape[0]  # number of examples
    cost_history = []  # to store cost for each iteration
    
    for i in range(num_iterations):
        # Compute predictions
        Ypred = np.dot(X, w) + b
        
        # Compute gradients
        error_val = Ypred - y
        dw = (1/m) * np.dot(X.T, error_val)  # gradient of the weights
        db = (1/m) * np.sum(error_val)         # gradient of the bias
        
        # Update the weights and bias
        w -= (alpha * dw)
        b -= (alpha * db)
        
        # Compute cost and store it
        if i % 100 == 0:  # Compute cost every 100 iterations for efficiency
            cost = compute_cost(X, y, w, b)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost {cost}")
    
    return w, b, cost_history


# Call gradient descent and compute cost
w_opt, b_opt, cost_history = gradient_descent(Xtrain, Ytrain, w, b, alpha, num_iterations)

# Final cost after gradient descent
final_cost = compute_cost(Xtrain, Ytrain, w_opt, b_opt)
print(f"Final cost: {final_cost}")
