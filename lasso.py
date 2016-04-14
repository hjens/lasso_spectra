import numpy as np
from sklearn.linear_model import LassoCV, Lasso


class SKLasso:
    #TODO: move properties here
    #TODO: add kwargs to sklearn calls
    def __init__(self, alpha=1.0, normalize=False, max_iter=1000):
        """ This class is basically just a wrapper around the
        scikit-learn Lasso class, but modified to have the
        same syntax and properties as the GeneralizedLasso class.
        If you do not need to use link functions, use this class,
        since it is much faster and more reliable


        :param alpha: The regularization constant
        :param normalize: Whether to normalize features before fitting
        :param max_iter: The maximum number of learning epochs to
                use when fitting
        """
        self.alpha = alpha
        self._normalize = normalize
        self._max_iter = max_iter

    def fit(self, X, y, verbose=True):
        """ Fit the model to the given data.

        :param X: The data matrix. Shape must
                be (n_samples, n_features)
        :param y: The labels
        :param verbose: If true, output steps
        """
        # Fit model
        model = Lasso(alpha=self.alpha, normalize=self._normalize)
        model.fit(X, y)

        # Save coeffs
        self.coeffs = model.coef_
        self.bias = model.intercept_

    def fit_CV(self, X, y, alphas=np.linspace(0.1, 10, 5), n_folds=10):
        """ Find the value of alpha that gives the lowest mse, using
        N-fold cross-validation. After this method has ran, the alpha
        property will be set to the value that gives the lowest cross-
        validation mse. The property alpha_mse will contain the mse
        for each value of alpha. The matrix alpha_coeffs will contain the
        coefficients for each alpha.

        :param X: The data matrix. Shape must
                be (n_samples, n_features)
        :param y: The labels
        :param alphas: The values of alpha to try
        :param n_folds: The number of cross-validation folds
        """
        # Fit model
        model = LassoCV(alphas=alphas, normalize=self._normalize,
                        max_iter=self._max_iter, cv=n_folds)
        model.fit(X, y)

        # Save coeffs and stuff
        self.coeffs = model.coef_
        self.bias = model.intercept_
        self.alpha = model.alpha_
        # This is reversed compared to alphas given
        self.alpha_mse = model.mse_path_.mean(axis=1)[::-1]

    def predict(self, X):
        """ Predict y for the given X

        :param X: The data matrix. Shape must
                be (n_samples, n_features) or
                (n_features)

        :return: The estimate of y
        """
        return self.bias + np.dot(X, self.coeffs)

    def mse(self, X, y):
        """ Calculate the mean squared error for data and labels

        :param X: The data matrix. Shape must
                be (n_samples, n_features) or
                (n_features)
        :param y: The labels
        :return: mse: float
        """
        yhat = self.predict(X)
        return np.mean((yhat-y)**2)