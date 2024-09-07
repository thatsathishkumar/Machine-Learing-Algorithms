import numpy as np

Xtrain_tmp = np.arange(-10, 11)

def sigmoid_func(z: np.array) -> np.array:
    return 1 / (np.exp(-z) + 1)

Xres = sigmoid_func(Xtrain_tmp)
print(f"this is sigmoid function = \n{np.c_[Xtrain_tmp, Xres]}")