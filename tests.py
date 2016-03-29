import lasso_tf
import pylab as pl

if __name__ == '__main__':
    func = lasso_tf.get_test_model()
    dataset_train, labels_train = lasso_tf.get_dataset(func, n_datapoints=1e2)
    dataset_test, labels_test = lasso_tf.get_dataset(func, n_datapoints=1e2)
    lasso = lasso_tf.fit_lasso_scikit_learn(dataset_train, labels_train)

    tf_coeffs = lasso_tf.fit_lasso(dataset_train, labels_train)
    print tf_coeffs

    pl.plot(func.coeffs, 'b-^', label='true')
    pl.plot(lasso.coef_, 'r-x', label='scikit')
    pl.plot(tf_coeffs, 'g-o', label='tf')
    pl.legend(loc='best')
    pl.show()