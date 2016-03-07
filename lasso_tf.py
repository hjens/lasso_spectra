import numpy as np
import pylab as pl
import tensorflow as tf

# ---------- Test data -----------------

def get_test_model(n_features=100):
    '''
    Return a linear function with n_features random 
    coefficients plus noise
    '''
    coeffs = np.random.normal(size=n_features)
    def func(x):
        return np.dot(x, coeffs)
    return func


def get_test_data(n_features=100, n_datapoints=1e5):
    '''
    Generate a test set with the given dimensions,
    using a test model.
    Returns:
        input_data - n_features x n_datapoints
        output_data - n_datapoints
    '''
    input_data = np.random.random((n_datapoints, n_features))
    output_data = get_test_model(n_features)(input_data)
    return input_data, output_data

# --------------------------------------