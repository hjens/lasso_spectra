import numpy as np
import pylab as pl
import tensorflow as tf


def fit_lasso(data, labels, alpha):
    '''
    Fit a Lasso model using tensorflow
    '''
    # Setup placeholders and variables
    num_datapoints = data.shape[0]
    num_features = data.shape[1]
    x = tf.placeholder(tf.float32, [None, num_features])
    y_ = tf.placeholder(tf.float32, [None, 1])
    coeffs = tf.Variable(tf.random_normal(shape=[num_features, 1]))
    bias = tf.Variable(tf.random_normal(shape=[1]))

    # Prediction
    #y = tf.matmul(x, coeffs) + bias
    y = tf.sigmoid(tf.matmul(x, coeffs) + bias)

    # Cost function
    lasso_cost = tf.reduce_sum(tf.pow(y-y_, 2))/(2.*num_datapoints) + \
                alpha*tf.reduce_sum(tf.abs(coeffs))


    # Minimizer
    NUM_STEPS = 2000
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_step = optimizer.minimize(lasso_cost)

    # Fit the model
    init = tf.initialize_all_variables()
    cost_history = np.zeros(NUM_STEPS)
    sess = tf.Session()
    sess.run(init)

    for i in range(NUM_STEPS):
        if i % 100 == 0:
            print 'Step:', i
        sess.run(train_step, feed_dict={x: data, y_: labels})
        cost_history[i] = sess.run(lasso_cost, feed_dict={x: data,
                y_:labels})

    return sess.run(coeffs), cost_history


def normalize_dataset(dataset):
    '''
    Dataset should have the shape (n_datapoints, n_features)
    '''
    norm_data = np.zeros_like(dataset)
    for i in range(dataset.shape[1]):
        norm_data[:,i] = dataset[:,i]-dataset[:,i].mean()
        norm_data[:,i] /= norm_data[:,i].std()
    return norm_data



# ---------- For testing ----------------

def fit_lasso_scikit_learn(data, labels, alpha):
    '''
    '''
    from sklearn import linear_model
    lasso_model = linear_model.Lasso(alpha=alpha)
    lasso_model.fit(X=data, y=labels)
    return lasso_model


def get_test_model(n_features=100):
    '''
    Return a linear function with n_features random 
    coefficients plus noise
    '''
    coeffs = np.random.normal(size=(n_features, 1))
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
    input_data = np.random.random((n_datapoints, n_features))*10.
    output_data = func(input_data)
    return input_data, output_data

# --------------------------------------

