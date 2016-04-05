import lasso_tf
import pylab as pl

if __name__ == '__main__':
    n_features = 3
    func = lasso_tf.get_test_model(n_features=n_features)
    dataset_train, labels_train = lasso_tf.get_dataset(func, n_datapoints=1e2,
        n_features=n_features)
    dataset_test = lasso_tf.normalize_dataset(dataset_train)

    dataset_test, labels_test = lasso_tf.get_dataset(func, n_datapoints=1e2,
        n_features=n_features)

    lasso = lasso_tf.fit_lasso_scikit_learn(dataset_train, labels_train, 
        alpha=0.0001)
    tf_coeffs, tf_cost = lasso_tf.fit_lasso(dataset_train, labels_train,
        alpha=0.1)

    pl.plot(func.coeffs, 'b-^', label='true')
    pl.plot(lasso.coef_, 'r-x', label='scikit')
    pl.plot(tf_coeffs, 'g-o', label='tf')
    pl.legend(loc='best')

    pl.figure()
    pl.plot(tf_cost)
    pl.show()