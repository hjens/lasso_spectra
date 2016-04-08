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
    coeffs = np.random.normal(size=(n_features, 1))*4.
    for i in range(len(coeffs)): #Make some coefficients zero
        if np.random.random() < 0.2:
            coeffs[i] = 0.
    def func(x):
        return np.dot(x, coeffs)
    func.coeffs = coeffs
    return func


def get_dataset(func, n_features=100, n_datapoints=1e4, noise_level=1.e-10):
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

# Tests

def test_cross_validation():
    n_features = 2
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_dataset(func, 
        n_datapoints=2e3, n_features=n_features, noise_level=.10)
    alphas = 10**np.linspace(-3, 1, 100)

    # Fit scikit lasso
    lasso_scikit = LassoCV(alphas=alphas, cv=3, normalize=False)
    lasso_scikit.fit(dataset_train, labels_train[:,0])
    scikit_cost = lasso_scikit.mse_path_.mean(axis=1)

    # Fit tf lasso
    #gen_lasso = gl.GeneralizedLasso(alpha=0.001, max_iter=500,
    #    link_function=None)
    #gen_lasso.fit_CV(dataset_train, labels_train[:,0], alphas=alphas,
    #    n_folds=3)

    #pl.figure()
    #for i in range(n_features):
    #    pl.plot(alphas, gen_lasso.alpha_coeffs[:,i])
    #pl.xlabel('alpha')
    #pl.ylabel('coefficient value')

    pl.figure()
    #pl.semilogx(alphas, gen_lasso.alpha_cost, label='tf')
    pl.loglog(alphas, scikit_cost, label='scikit')
    pl.legend(loc='best')
    pl.xlabel('alpha')
    pl.ylabel('cost')
    pl.show()


if __name__ == '__main__':


    #gen_lasso = gl.GeneralizedLasso(alpha=0.001, max_iter=500,
    #    link_function='sigmoid')
#    gen_lasso.fit(dataset_train, labels_train[:,0])

    #gen_lasso.fit_CV(dataset_train, labels_train[:,0])
    #print gen_lasso.predict(dataset_train[0,:])
    #print gen_lasso.cost(dataset_train, labels_train[:,0])

    #pl.figure()
    #pl.plot(gen_lasso.predict(dataset_train), 
    #    labels_train, 'o')
    #pl.xlabel('predicted')
    #pl.ylabel('true')

    #lasso = lasso_tf.fit_lasso_scikit_learn(dataset_train, labels_train, 
    #    alpha=5.)

    #pl.figure()
    #pl.plot(func.coeffs, 'b-^', label='true')
    #pl.plot(lasso.coef_, 'r-x', label='scikit')
    #pl.plot(gen_lasso.coeffs, 'k-o', label='gl')
    #pl.legend(loc='best')

    #pl.show()

    test_cross_validation()