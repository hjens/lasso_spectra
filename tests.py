import lasso_tf
import pylab as pl
import numpy as np
import generalized_lasso as gl
from sklearn.linear_model import LassoCV

# Data generation
def get_test_model(n_features=100):
    '''
    Return a linear function with n_features random 
    coefficients plus noise
    '''
    b = np.random.random()*4.
    coeffs = np.random.normal(size=(n_features, 1))*4.
    for i in range(len(coeffs)): #Make some coefficients zero
        if np.random.random() < 0.2:
            coeffs[i] = 0.
    def func(x):
        return np.dot(x, coeffs) + b
    func.coeffs = coeffs
    return func


def get_random_dataset(func, n_features=100, n_datapoints=1e4, 
        noise_level=1.e-10):
    '''
    Generate a test set with the given dimensions,
    using a test model.
    Returns:
        input_data - n_features x n_datapoints
        output_data - n_datapoints
    '''
    input_data = np.random.random((n_datapoints, n_features))*10.
    output_data = func(input_data) + np.random.normal(size=input_data.shape,
        scale=noise_level)
    return input_data, output_data


def sigmoid(x):
    return 1./(1 + np.exp(-x))

# Tests ------------------

def test_cross_validation():
    n_features = 2
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=2e3, n_features=n_features, noise_level=1.)
    alphas = 10**np.linspace(-3, 1, 10)

    # Fit scikit lasso
    lasso_scikit = LassoCV(alphas=alphas, cv=5, normalize=False)
    lasso_scikit.fit(dataset_train, labels_train[:,0])
    scikit_cost = lasso_scikit.mse_path_.mean(axis=1)

    # Fit tf lasso
    gen_lasso = gl.GeneralizedLasso(alpha=0.001, max_iter=500,
        link_function=None)
    gen_lasso.fit_CV(dataset_train, labels_train[:,0], alphas=alphas,
        n_folds=5)

    #pl.figure()
    #for i in range(n_features):
    #    pl.plot(alphas, gen_lasso.alpha_coeffs[:,i])
    #pl.xlabel('alpha')
    #pl.ylabel('coefficient value')

    pl.figure()
    pl.semilogx(alphas, gen_lasso.alpha_mse, label='tf')
    pl.loglog(lasso_scikit.alphas_, scikit_cost, label='scikit')
    #pl.loglog(lasso_scikit.alphas_, lasso_scikit.mse_path_)
    pl.legend(loc='best')
    pl.xlabel('alpha')
    pl.ylabel('cost')
    pl.show()


def test_linear_regression():
    np.random.seed(1)
    n_features = 5
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=5e2, n_features=n_features, noise_level=1.e-10)

    # Fit tf lasso
    gen_lasso = gl.GeneralizedLasso(alpha=1.e-10, max_iter=2000,
        link_function=None)
    gen_lasso.fit(dataset_train, labels_train[:,0])

    # Plot results
    pl.plot(gen_lasso.coeffs, 'o-', label='tf fit')
    pl.plot(func.coeffs, 'x-', label='true')
    pl.legend(loc='best')
    pl.title('Test linear regression')
    pl.ylabel('Coeff value')
    pl.show()


def test_regularization():
    np.random.seed(1)
    n_features = 5
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=1e3, n_features=n_features, noise_level=1.e-10)

    alphas = 10**np.linspace(-1, 3, 10)
    alpha_coeffs = np.zeros((n_features, len(alphas)))
    for i, alpha in enumerate(alphas):
        gen_lasso = gl.GeneralizedLasso(alpha=alpha, max_iter=2000,
        link_function=None)
        gen_lasso.fit(dataset_train, labels_train[:,0])
        alpha_coeffs[:,i] = gen_lasso.coeffs[:,0]

    # Plot results
    for i in range(n_features):
        pl.semilogx(alphas, alpha_coeffs[i,:], label='coeff no %d' % i)
        pl.semilogx(alphas, np.ones_like(alphas)*func.coeffs[i], ':')
    pl.legend(loc='best')
    pl.title('Test regularization')
    pl.ylabel('Coeff value')
    pl.xlabel('alpha')
    pl.show()



if __name__ == '__main__':
    #test_cross_validation()
    #test_linear_regression()
    test_regularization()