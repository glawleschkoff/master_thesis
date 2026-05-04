import numpy as np
from scipy.optimize import root
from scipy.special import digamma
import scipy.special as sp

def safelog(value):
    eps = np.finfo(float).eps
    safe_value = np.clip(value, eps, None)
    safe_log = np.log(safe_value)
    return safe_log

def softmax(x):
    stabilized = x - np.max(x)
    exp_x = np.exp(stabilized)
    return exp_x / np.sum(exp_x)