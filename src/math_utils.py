import numpy as np

def safelog(x):
    return np.log(x + 1e-9)

def softmax(x):
    stabilized = x - np.max(x)
    exp_x = np.exp(stabilized)
    return exp_x / np.sum(exp_x)