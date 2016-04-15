import numpy as np
import tensorflow as tf

#TODO: move properties to beginning

class GeneralizedLasso:
    def __init__(self, alpha=1.0, normalize=False, max_iter=1000, 
                link_function=None, learning_rate=0.01):
        """ A generalized linear model with L1 regularization and the
        possibility to specify different link functions.

        :param alpha: The regularization constant
        :param normalize: Whether to normalize features before fitting
        :param max_iter: The maximum number of learning epochs to
                use when fitting
        :param link_function: The link function to use.
                Valid values are:
                - None: No link function (standard lasso)
                - sigmoid: a sigmoid function, 1/(1+exp(-x))
        :param learning_rate: The learning rate
        :type link_function: string
        """
        self.alpha = alpha
        self._normalize = normalize
        self._max_iter = max_iter
        self._link_function = link_function
        self._learning_rate = learning_rate

    def fit(self, X, y, verbose=True):
        """ Fit the model to the given data.

        :param X: The data matrix. Shape must
                be (n_samples, n_features)
        :param y: The target
        :param verbose: If true, output steps

        TODO: normalize features
        """
        # Check dimensions of data
        if np.ndim(X) != 2:
            raise ValueError('X must have dimensions (n_samples, n_features')
        if np.ndim(y) != 1:
            raise ValueError('y must have dimension (n_samples)')
        if len(y) != X.shape[0]:
                raise ValueError('y must be of length X.shape[0]')

        # Add an empty dimension to y
        y = np.expand_dims(y, axis=1)

        # Setup placeholders and variables
        n_samples = X.shape[0]
        n_features = X.shape[1]
        x = tf.placeholder(tf.float32, [None, n_features])
        y_ = tf.placeholder(tf.float32, [None, 1])
        coeffs = tf.Variable(tf.random_normal(shape=[n_features, 1]))
        bias = tf.Variable(tf.random_normal(shape=[1]))

        # Prediction
        predict = self._get_predictor(x, coeffs, bias)

        # Cost function
        cost =  self._get_cost_function(predict, y_, n_samples, coeffs)

        # Minimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        train_step = optimizer.minimize(cost)

        # Prepare the fitting
        init = tf.initialize_all_variables()
        self._cost_history = np.zeros(self._max_iter)
        sess = tf.Session()
        sess.run(init)

        # Fit the model
        for i in range(self._max_iter):
            if i % 100 == 0 and verbose:
                print 'Step:', i
            sess.run(train_step, feed_dict={x: X, y_: y})
            self._cost_history[i] = sess.run(cost, feed_dict={x: X, y_:y})
            # Have we reached a minimum?
            if i > 2:
                if self._cost_history[-1] > self._cost_history[-2]:
                    break

        # Save coeffs
        self.coeffs = sess.run(coeffs)
        self.bias = sess.run(bias)

    def fit_CV(self, X, y, alphas=np.linspace(0.1, 10, 5), n_folds=10):
        """ Find the value of alpha that gives the lowest mse, using
        N-fold cross-validation. After this method has ran, the alpha
        property will be set to the value that gives the lowest cross-
        validation mse. The property alpha_mse will contain the mse
        for each value of alpha. The matrix alpha_coeffs will contain the
        coefficients for each alpha.

        :param X: The data matrix. Shape must
                be (n_samples, n_features)
        :param y: The target
        :param alphas: The values of alpha to try
        :param n_folds: The number of cross-validation folds
        """
        print 'Starting %d-fold cross-validation...' % n_folds

        # Fit a model and calculate the CV mse for each alpha
        self.alpha_mse = np.zeros_like(alphas)
        self.alpha_coeffs = np.zeros((len(alphas), X.shape[1]))
        self.alpha_bias = np.zeros(len(alphas))

        for i, alpha in enumerate(alphas):
            self.alpha = alpha
            mse_sum = 0.
            for fold in range(n_folds):
                train_idx, cv_idx = self._get_cv_idx(fold, n_folds, len(y))
                self.fit(X[train_idx], y[train_idx], verbose=False)
                mse_sum += self.mse(X[cv_idx], y[cv_idx])
            self.alpha_mse[i] = mse_sum/float(n_folds)
            print 'MSE for alpha=%.5f: %.5f' % (alpha, self.alpha_mse[i])
            # Save coeffs and bias
            self.alpha_coeffs[i,:] = np.squeeze(self.coeffs)
            self.alpha_bias[i] = self.bias

        # Find the alpha that gave the minimum mse
        best_idx = np.argmin(self.alpha_mse)
        self.alpha = alphas[best_idx]
        self.coeffs = np.expand_dims(self.alpha_coeffs[best_idx, :], axis=1)
        self.bias = self.alpha_bias[best_idx]

        # Print results
        print 'Finished %d-fold cross-validation' % n_folds
        print 'Found best alpha: ', self.alpha
        print 'Minimum mse: ', self.alpha_mse.min()

    def predict(self, X):
        """ Predict y for X

        :param X: The data matrix. Shape must
                be (n_samples, n_features) or
                (n_features)
        :return: The estimate of y
        """
        # Check that we have a model to use for prediction
        if not hasattr(self, 'coeffs'):
            raise Exception('Must fit a model before predicting values.')

        # Check dimensions of data
        if not (np.ndim(X) == 1 or np.ndim(X) == 2):
            raise ValueError('X must have dimensions (n_samples, n_features) \
                or (n_features)')
        if np.ndim(X) == 1:
            if not len(X) == len(self.coeffs):
                raise ValueError('X must have dimensions \
                    (n_samples, n_features) or (n_features)')
        elif np.ndim(X) == 2:
            if not X.shape[1] == len(self.coeffs):
                raise ValueError('X must have dimensions \
                    (n_samples, n_features) or (n_features)')

        # Add an extra dimension to X if needed
        if np.ndim(X) == 1:
            X = np.expand_dims(X, axis=0)

        # Run prediction
        n_features = len(self.coeffs)
        x = tf.placeholder(tf.float32, [None, n_features])
        coeffs = tf.Variable(tf.random_normal(shape=[n_features, 1]))
        bias = tf.Variable(tf.random_normal(shape=[1]))
        predict = self._get_predictor(x, coeffs, bias)
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        yhat = sess.run(predict, feed_dict={x: X, coeffs: self.coeffs, 
            bias: self.bias})

        return np.squeeze(yhat)

    def mse(self, X, y):
        """ Calculate the mean squared error for data and target

        :param X: The data matrix. Shape must
                be (n_samples, n_features) or
                (n_features)
        :param y: The target
        :return: The mse for the given data
        """
        # Run a forward prediction. This will also check
        # dimensions of X
        yhat = self.predict(X)

        # Check dimensions of y
        if np.ndim(y) != 1:
            raise ValueError('y must have shape (n_samples)')
        if len(y) != X.shape[0]:
            raise ValueError('y must be of length X.shape[0]')

        # Add an empty dimension to y
        y = np.expand_dims(y, axis=1)

        # Add an empty dimension to yhat
        yhat = np.expand_dims(yhat, axis=1)

        # Setup graph
        y_ = tf.placeholder(tf.float32, [None, 1])
        predict = tf.placeholder(tf.float32, [None, 1])
        mse = self._get_mse(predict, y_)

        # Run mse function
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        c = sess.run(mse, feed_dict={predict: yhat, y_: y})
        return c

    #--------- Methods for internal use -----------------------

    #--------- Building the tensorflow computational graph ----
    def _get_predictor(self, x, coeffs, bias):
        """Create the prediction part of the tf graph """
        if self._link_function is None:
            predict = tf.matmul(x, coeffs) + bias
        elif self._link_function == 'sigmoid':
            predict = tf.sigmoid(tf.matmul(x, coeffs) + bias)
        else:
            raise ValueError('Invalid link function: %s' % \
                 self._link_function)
        return predict

    def _get_cost_function(self, predict, y_, n_samples, coeffs):
        """Create the cost function part of the tf graph"""
        cost = tf.reduce_sum(tf.square(predict-y_))/(2.*n_samples) + \
                    self.alpha*tf.reduce_sum(tf.abs(coeffs))
        return cost

    def _get_mse(self, predict, y_):
        """Mean square error """
        mse = tf.reduce_mean(tf.square(predict-y_))
        return mse

    #--------- For cross-validation ---------------------------
    def _get_cv_idx(self, fold, n_folds, n_samples):
        """Get indices for the given CV fold.
        fold starts at 0
        Assume unordered data
        Return training_idx, cv_idx"""
        fold_size = n_samples/n_folds
        low_idx = fold*fold_size
        high_idx = (fold+1)*fold_size
        assert low_idx >= 0
        assert high_idx <= n_samples
        all_idx = np.arange(n_samples)
        cv_idx = (all_idx >= low_idx)*(all_idx < high_idx)
        train_idx = ~cv_idx
        return train_idx, cv_idx