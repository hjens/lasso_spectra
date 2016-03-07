import numpy as np
import pylab as pl
import tensorflow as tf



# ---------- For testing ----------------

def fit_lasso_scikit_learn(data, labels):
    '''
    '''
    from sklearn import linear_model
    lasso_model = linear_model.Lasso(alpha=0.001)
    lasso_model.fit(X=data, y=labels)
    return lasso_model


def get_test_model(n_features=100):
    '''
    Return a linear function with n_features random 
    coefficients plus noise
    '''
    coeffs = np.random.normal(size=n_features)
    def func(x):
        return np.dot(x, coeffs)
    func.coeffs = coeffs
    return func


def get_dataset(func, n_features=100, n_datapoints=1e5):
    '''
    Generate a test set with the given dimensions,
    using a test model.
    Returns:
        input_data - n_features x n_datapoints
        output_data - n_datapoints
    '''
    input_data = np.random.random((n_datapoints, n_features))
    output_data = func(input_data)
    return input_data, output_data

# --------------------------------------