import numpy as np
import pylab as pl
import tensorflow as tf


def fit_lasso(data, labels):
    '''
    Fit a Lasso model using tensorflow
    '''
    # Setup placeholders and variables
    num_datapoints = data.shape[0]
    num_features = data.shape[1]
    x = tf.placeholder(tf.float32, [None, num_features])
    y_ = tf.placeholder(tf.float32, [None])
    coeffs = tf.Variable(tf.random_normal(shape=[num_features, 1]))
    bias = tf.Variable(tf.zeros([1]))

    # Prediction
    y = tf.matmul(x, coeffs) + bias

    # Cost function
    lasso_cost = tf.square(y-y_)

    # Minimizer
    LEARNING_RATE = 1e-11
    NUM_STEPS = 100
    #optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    optimizer = tf.train.AdamOptimizer()
    train_step = optimizer.minimize(lasso_cost)

    # Fit the model
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(NUM_STEPS):
        if i % 10 == 0:
            print 'Step:', i
        sess.run(train_step, feed_dict={x: data, y_: labels})

    return coeffs.eval(session=sess)


# ---------- For testing ----------------

def fit_lasso_scikit_learn(data, labels):
    '''
    '''
    from sklearn import linear_model
    lasso_model = linear_model.Lasso(alpha=0.0001)
    lasso_model.fit(X=data, y=labels)
    return lasso_model


def get_test_model(n_features=100):
    '''
    Return a linear function with n_features random 
    coefficients plus noise
    '''
    coeffs = np.random.normal(size=n_features)
    def func(x):
        return np.dot(x, coeffs)
    func.coeffs = coeffs
    return func


def get_dataset(func, n_features=100, n_datapoints=1e4):
    '''
    Generate a test set with the given dimensions,
    using a test model.
    Returns:
        input_data - n_features x n_datapoints
        output_data - n_datapoints
    '''
    input_data = np.random.random((n_datapoints, n_features))
    output_data = func(input_data)
    return input_data, output_data

# --------------------------------------