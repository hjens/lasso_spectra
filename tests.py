import lasso_tf
import pylab as pl

if __name__ == '__main__':
    func = lasso_tf.get_test_model()
    dataset_train, labels_train = lasso_tf.get_dataset(func, n_datapoints=1e5)
    dataset_test, labels_test = lasso_tf.get_dataset(func, n_datapoints=1e4)
    lasso = lasso_tf.fit_lasso_scikit_learn(dataset_train, labels_train)

    pl.plot(func.coeffs, 'bo', label='true')
    pl.plot(lasso.coef_, 'ro', label='fit')
    pl.show()