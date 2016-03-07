import numpy as np
import pylab as pl
import tensorflow as tf

# ---------- Test data -----------------

def get_test_model(n_dim=100):
    '''
    Return a linear function with n_dim random 
    coefficients plus noise
    '''
    coeffs = np.random.normal(size=n_dim)
    def func(x):
        return np.dot(x, coeffs) + np.random.normal(scale=0.5)
    return func

# --------------------------------------