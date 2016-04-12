import lasso_tf
import pylab as pl
import numpy as np
import generalized_lasso as gl
import lasso


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
    np.random.seed(1)
    n_features = 3
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=5e3, n_features=n_features, noise_level=1.)
    alphas = 10**np.linspace(-3, 2, 10)

    # Fit scikit lasso
    lasso_scikit = lasso.SKLasso(alpha=0.001, max_iter=1000)
    lasso_scikit.fit_CV(dataset_train, labels_train[:,0], alphas=alphas,
                        n_folds=5)

    # Fit tf lasso
    gen_lasso = gl.GeneralizedLasso(alpha=0.001, max_iter=1000,
                                    link_function=None)
    gen_lasso.fit_CV(dataset_train, labels_train[:,0], alphas=alphas,
                     n_folds=5)

    pl.figure()
    pl.title('CV test. TF will have higher errors, since it is not exact')
    pl.semilogx(alphas, gen_lasso.alpha_mse, 'o-', label='tf')
    pl.semilogx(alphas, lasso_scikit.alpha_mse, '*-', label='scikit')
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
    gen_lasso = gl.GeneralizedLasso(alpha=1.e-10, max_iter=5000,
        link_function=None)
    gen_lasso.fit(dataset_train, labels_train[:,0])

    # Fit scikit lasso
    lasso_scikit = lasso.SKLasso(alpha=1.e-10, max_iter=5000)
    lasso_scikit.fit(dataset_train, labels_train[:,0])

    # Print mse. This will be worse for TF, since it is not exact
    print 'Scikit mse', lasso_scikit.mse(dataset_train, labels_train[:,0])
    print 'TF mse', gen_lasso.mse(dataset_train, labels_train[:,0])

    # Plot results
    pl.plot(gen_lasso.coeffs, 'o-', label='tf fit')
    pl.plot(func.coeffs, 'x-', label='true')
    pl.plot(lasso_scikit.coeffs, '^', label='scikit')
    pl.legend(loc='best')
    pl.title('Test linear regression')
    pl.ylabel('Coeff value')
    pl.show()


def test_link_function():
    np.random.seed(3) #This seed gives both positive and negative coefficients
    n_features = 4
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=5e3, n_features=n_features, noise_level=1.e-10)
    labels_train = sigmoid(labels_train)

    # Fit tf lasso
    gen_lasso = gl.GeneralizedLasso(alpha=1.e-10, max_iter=2000,
        link_function='sigmoid', learning_rate=0.1)
    gen_lasso.fit(dataset_train, labels_train[:,0])

    # Predict values
    predicted = gen_lasso.predict(dataset_train)

    # Plot results
    pl.subplot(131)
    pl.plot(gen_lasso.coeffs, 'o-', label='tf fit')
    pl.plot(func.coeffs, 'x-', label='true')
    pl.legend(loc='best')
    pl.title('Test sigmoid link function')
    pl.ylabel('Coeff value')
    pl.subplot(132)
    pl.semilogy(gen_lasso._cost_history)
    pl.ylabel('cost')
    pl.xlabel('iterations')
    pl.subplot(133)
    pl.plot(predicted, labels_train, 'o')
    pl.xlabel('Predicted')
    pl.ylabel('True')
    pl.show()


def test_regularization():
    np.random.seed(1)
    n_features = 5
    func = get_test_model(n_features=n_features)
    dataset_train, labels_train = get_random_dataset(func, 
        n_datapoints=1e3, n_features=n_features, noise_level=1.e-10)

    alphas = 10**np.linspace(-1, 3, 10)
    alpha_coeffs = np.zeros([n_features, len(alphas)])
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
    test_cross_validation()
    #test_linear_regression()
    #test_regularization()
    #test_link_function()