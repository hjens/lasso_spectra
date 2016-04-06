import lasso_tf
import pylab as pl
import numpy as np
import generalized_lasso as gl

def sigmoid(x):
    return 1./(1 + np.exp(-x))

if __name__ == '__main__':
    n_features = 200
    func = lasso_tf.get_test_model(n_features=n_features)
    dataset_train, labels_train = lasso_tf.get_dataset(func, n_datapoints=1e3,
        n_features=n_features)
    dataset_test = lasso_tf.normalize_dataset(dataset_train)
    labels_train = sigmoid(labels_train)
    dataset_test, labels_test = lasso_tf.get_dataset(func, n_datapoints=1e3,
        n_features=n_features)

    gen_lasso = gl.GeneralizedLasso(alpha=0.001, max_iter=1000,
        link_function='sigmoid')
    gen_lasso.fit(dataset_train, labels_train[:,0])
    print gen_lasso.predict(dataset_train[0,:])
    pl.figure()
    pl.plot(gen_lasso.predict(dataset_train), 
        labels_train, 'o')
    pl.xlabel('predicted')
    pl.ylabel('true')

    lasso = lasso_tf.fit_lasso_scikit_learn(dataset_train, labels_train, 
        alpha=5.)

    pl.figure()
    pl.plot(func.coeffs, 'b-^', label='true')
    pl.plot(lasso.coef_, 'r-x', label='scikit')
    pl.plot(gen_lasso.coeffs, 'k-o', label='gl')
    pl.legend(loc='best')

    pl.show()