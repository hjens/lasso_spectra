import numpy as np
import tensorflow as tf

class GeneralizedLasso:
    def __init__(self, alpha=1.0, normalize=False, max_iter=1000, 
                link_function=None, learning_rate=0.01):
        '''
        A generalized linear model with L1 regularization and the
        possibility to specify different link functions.

        Parameters:
            * alpha: float
                The regularization constant
            * normalize: boolean
                Whether to normalize features before fitting
            * max_iter: int
                The maximum number of learning epochs to 
                use when fitting
            * link_function: string or None
                The link function to use. Valid values are:
                - None: No link function (standard lasso)
                - sigmoid: a sigmoid function, 1/(1+exp(-x))
            * learning_rate: float
                The learning rate
        '''
        self.alpha = alpha
        self._normalize = normalize
        self._max_iter = max_iter
        self._link_function = link_function
        self._learning_rate = learning_rate


    def fit(self, X, y):
        '''
        Fit the model to the given data.

        Parameters:
            * X: numpy matrix
                The data matrix. Shape must 
                be (n_samples, n_features)
            * y: numpy array
                The labels

        TODO: normalize features
        '''
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
            if i % 100 == 0:
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
        '''
        Find the value of alpha that gives the lowest cost, using
        N-fold cross-validation. After this method has ran, the alpha
        property will be set to the value that gives the lowest cross-
        validation cost. The property alpha_cost will contain the cost
        for each value of alpha. The matrix alpha_coeffs will contain the
        coefficients for each alpha.

        Parameters:
            * X: numpy matrix
                The data matrix. Shape must 
                be (n_samples, n_features)
            * y: numpy array
                The labels
            * alphas: numpy array
                The values of alpha to try
            * n_folds: integer
                The number of cross-validation folds
        '''
        # Fit a model and calculate the cost for each alpha
        self.alpha_cost = np.zeros_like(alphas)
        for i, alpha in enumerate(alphas):
            self.alpha = alpha
            cost_sum = 0.
            for fold in range(n_folds):
                train_idx, cv_idx = get_cv_idx(fold)
                self.fit(X[train_idx], y[train_idx])
                cost_sum += self.cost(X[cv_idx], y[cv_idx])
            self.alpha_cost[i] = cost_sum/float(n_folds)
            # Save coeffs and bias

        # Find the alpha that gave the minimum cost
        best_idx = np.argmin(self.alpha_cost)
        self.alpha = alphas[best_idx]
        self.coeffs = self.alpha_coeffs[best_idx, :]
        self.bias = self.alpha_bias[best_idx]

        # Print results
        print 'Finished %d-fold cross-validation' % n_folds
        print 'Found best alpha: ', self.alpha
        print 'Minimum cost: ', self.alpha_cost.min()
        

    def predict(self, X):
        '''
        Predict y for X

        Parameters:
            * X: numpy matrix 
                The data matrix. Shape must 
                be (n_samples, n_features) or
                (n_features)

        Returns:
            * yhat: numpy array
                The estimate of y
        '''
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


    def cost(self, X, y):
        '''
        Calculate the cost for data and labels

        Parameters:
            * X: numpy matrix 
                The data matrix. Shape must 
                be (n_samples, n_features) or
                (n_features)
            * y: numpy array
                The labels
        Returns:
            * cost: float
                The cost for the given data
        '''

        pass


    #--------- Methods for internal use -----------------------

    #--------- Building the tensorflow computational graph ----
    def _get_predictor(self, x, coeffs, bias):
        '''
        Create the prediction part of the tf graph
        '''
        if self._link_function is None:
            predict = tf.matmul(x, coeffs) + bias
        elif self._link_function == 'sigmoid':
            predict = tf.sigmoid(tf.matmul(x, coeffs) + bias)
        else:
            raise ValueError('Invalid link function: %s' % \
                 self._link_function)
        return predict


    def _get_cost_function(self, predict, y_, n_samples, coeffs):
        '''
        Create the cost function part of the tf graph
        '''
        cost = tf.reduce_sum(tf.square(predict-y_))/(2.*n_samples) + \
                    self.alpha*tf.reduce_sum(tf.abs(coeffs))
        return cost


